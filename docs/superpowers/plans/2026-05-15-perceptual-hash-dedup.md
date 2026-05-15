# Perceptual Hash Deduplication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a perceptual image hash (aHash) layer to `split_pdf` that flags suspected duplicate documents without suppressing any output.

**Architecture:** Two new helper functions (`_perceptual_hash`, `_hamming_distance`) are added after `_render_page_fitz`. A new `_write_phash_report` function handles console + file output. Both the streaming dedup path and the non-streaming dedup path are extended to track perceptual hashes in a small dict and collect suspected-duplicate pairs, which are reported at the end of each `split_pdf` call. No output is ever suppressed.

**Tech Stack:** Python 3.11, PyMuPDF (`fitz`) 1.27, Pillow (`PIL.Image`) — both already in `pyproject.toml`.

---

## File Map

| File | Change |
|---|---|
| `src/pdf_extractor/image_splitter.py` | Add `_PHASH_THRESHOLD`, `_hamming_distance`, `_perceptual_hash` after `_render_page_fitz` (~line 557); add `_write_phash_report` just before `split_pdf` (~line 1337); extend streaming dedup path (~lines 1400–1490); extend non-streaming dedup path (~lines 1548–1599) |
| `tests/test_image_splitter.py` | Add tests for all three new helpers |

---

## Task 1: Add `_hamming_distance` and `_perceptual_hash`

**Files:**
- Modify: `src/pdf_extractor/image_splitter.py` (insert after `_render_page_fitz`, ~line 557)
- Test: `tests/test_image_splitter.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_image_splitter.py`:

```python
from pdf_extractor.image_splitter import (
    _hamming_distance,
    _perceptual_hash,
    _PHASH_THRESHOLD,
)


# --- _hamming_distance ---

def test_hamming_distance_identical():
    assert _hamming_distance(0, 0) == 0


def test_hamming_distance_one_bit():
    assert _hamming_distance(0b0, 0b1) == 1


def test_hamming_distance_all_64_bits():
    assert _hamming_distance(0, 0xFFFFFFFFFFFFFFFF) == 64


def test_hamming_distance_symmetric():
    assert _hamming_distance(0xAB, 0xCD) == _hamming_distance(0xCD, 0xAB)


# --- _perceptual_hash helpers ---

def _make_mock_fitz_page(rgb_value: int = 128):
    """Return a mock fitz page that renders as a solid rgb_value color at 64x83 px."""
    mock_pix = MagicMock()
    mock_pix.width = 64
    mock_pix.height = 83
    mock_pix.samples = bytes([rgb_value] * (64 * 83 * 3))
    mock_page = MagicMock()
    mock_page.rect.width = 100.0
    mock_page.get_pixmap.return_value = mock_pix
    return mock_page


def _make_mock_fitz_doc(page: object):
    mock_doc = MagicMock()
    mock_doc.__getitem__ = MagicMock(return_value=page)
    return mock_doc


# --- _perceptual_hash ---

def test_perceptual_hash_returns_int():
    h = _perceptual_hash(_make_mock_fitz_doc(_make_mock_fitz_page()), 0)
    assert isinstance(h, int)


def test_perceptual_hash_deterministic():
    doc = _make_mock_fitz_doc(_make_mock_fitz_page(120))
    assert _perceptual_hash(doc, 0) == _perceptual_hash(doc, 0)


def test_perceptual_hash_zero_width_returns_none():
    mock_page = MagicMock()
    mock_page.rect.width = 0
    assert _perceptual_hash(_make_mock_fitz_doc(mock_page), 0) is None


def test_perceptual_hash_exception_returns_none():
    mock_doc = MagicMock()
    mock_doc.__getitem__ = MagicMock(side_effect=RuntimeError("render failed"))
    assert _perceptual_hash(mock_doc, 0) is None


def test_perceptual_hash_identical_images_zero_distance():
    doc = _make_mock_fitz_doc(_make_mock_fitz_page(100))
    h1 = _perceptual_hash(doc, 0)
    h2 = _perceptual_hash(doc, 0)
    assert _hamming_distance(h1, h2) == 0


def test_perceptual_hash_similar_images_within_threshold():
    """Slight brightness difference stays within the similarity threshold."""
    doc_a = _make_mock_fitz_doc(_make_mock_fitz_page(rgb_value=120))
    doc_b = _make_mock_fitz_doc(_make_mock_fitz_page(rgb_value=130))
    h_a = _perceptual_hash(doc_a, 0)
    h_b = _perceptual_hash(doc_b, 0)
    assert _hamming_distance(h_a, h_b) <= _PHASH_THRESHOLD


def test_phash_threshold_value():
    assert _PHASH_THRESHOLD == 10
```

- [ ] **Step 2: Run the tests to confirm they fail**

```
cd c:\Users\nafsha44\source\repos\pdfExtractor
.venv/Scripts/python -m pytest tests/test_image_splitter.py -k "hamming or perceptual or phash_threshold" -v
```

Expected: `ImportError` or `FAILED` — the names don't exist yet.

- [ ] **Step 3: Add `_PHASH_THRESHOLD`, `_hamming_distance`, and `_perceptual_hash` to `image_splitter.py`**

Insert the following block immediately after `_render_page_fitz` ends (~line 557, the line reading `return None`):

```python
_PHASH_THRESHOLD = 10


def _hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _perceptual_hash(fitz_doc, page_idx: int) -> int | None:
    """Average hash (aHash) of a page rendered at 64-pixel width.

    Renders at ~64 px wide (much smaller than OCR renders), converts to an
    8×8 grayscale thumbnail, and returns a 64-bit integer where bit i is 1
    if pixel i is >= the mean pixel value.  Returns None on any failure.
    """
    try:
        page = fitz_doc[page_idx]
        w_pt = page.rect.width
        if w_pt <= 0:
            return None
        scale = 64 / w_pt
        mat = __import__("fitz").Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        del pix
        thumb = img.convert("L").resize((8, 8), Image.LANCZOS)
        pixels = list(thumb.getdata())
        mean = sum(pixels) / len(pixels)
        bits = [1 if p >= mean else 0 for p in pixels]
        return sum(b << i for i, b in enumerate(bits))
    except Exception:
        return None
```

- [ ] **Step 4: Run the tests — expect them to pass**

```
.venv/Scripts/python -m pytest tests/test_image_splitter.py -k "hamming or perceptual or phash_threshold" -v
```

Expected: all new tests `PASSED`.

- [ ] **Step 5: Run full suite — expect no regressions**

```
.venv/Scripts/python -m pytest tests/ -q
```

Expected: `71 passed` (all existing tests still pass).

- [ ] **Step 6: Commit**

```
git add src/pdf_extractor/image_splitter.py tests/test_image_splitter.py
git commit -m "feat: add _perceptual_hash and _hamming_distance helpers"
```

---

## Task 2: Add `_write_phash_report`

**Files:**
- Modify: `src/pdf_extractor/image_splitter.py` (insert just before `split_pdf`, ~line 1337)
- Test: `tests/test_image_splitter.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_image_splitter.py`:

```python
import datetime
from pdf_extractor.image_splitter import _write_phash_report


def test_write_phash_report_no_hits_silent(tmp_path, capsys):
    _write_phash_report([], tmp_path, Path("test.pdf"))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert not (tmp_path / "suspected_duplicates.txt").exists()


def test_write_phash_report_creates_file(tmp_path):
    hits = [(0, 1, 7, "RETAIL CONTRACT", "RETAIL CONTRACT", 5, 42)]
    _write_phash_report(hits, tmp_path, Path("sample.pdf"))
    assert (tmp_path / "suspected_duplicates.txt").exists()


def test_write_phash_report_file_content(tmp_path):
    hits = [(0, 1, 7, "RETAIL CONTRACT", "RETAIL CONTRACT", 5, 42)]
    _write_phash_report(hits, tmp_path, Path("sample.pdf"))
    content = (tmp_path / "suspected_duplicates.txt").read_text()
    assert "sample.pdf" in content
    assert "Group 0" in content
    assert "Group 1" in content
    assert "page 5" in content
    assert "page 42" in content
    assert "RETAIL CONTRACT" in content
    assert "[distance: 7]" in content


def test_write_phash_report_console_output(tmp_path, capsys):
    hits = [(0, 1, 7, "RETAIL CONTRACT", "RETAIL CONTRACT", 5, 42)]
    _write_phash_report(hits, tmp_path, Path("sample.pdf"))
    out = capsys.readouterr().out
    assert "SUSPECTED DUPLICATES" in out
    assert "distance: 7" in out


def test_write_phash_report_multiple_hits(tmp_path):
    hits = [
        (0, 2, 5, "CREDIT APPLICATION", "CREDIT APPLICATION", 10, 90),
        (1, 3, 8, "RETAIL CONTRACT", "RETAIL CONTRACT", 20, 100),
    ]
    _write_phash_report(hits, tmp_path, Path("batch.pdf"))
    content = (tmp_path / "suspected_duplicates.txt").read_text()
    assert content.count("distance:") == 2
```

You will also need `from pathlib import Path` at the top of the test file if it is not already imported.

- [ ] **Step 2: Run the tests to confirm they fail**

```
.venv/Scripts/python -m pytest tests/test_image_splitter.py -k "write_phash_report" -v
```

Expected: `ImportError` or `FAILED`.

- [ ] **Step 3: Add `_write_phash_report` to `image_splitter.py`**

Insert the following block immediately before `def split_pdf(` (~line 1337). The tuple type comment shows the field order:
`(a_group_idx, b_group_idx, distance, title_a, title_b, page_a_1based, page_b_1based)`

```python
def _write_phash_report(
    phash_hits: list[tuple[int, int, int, str, str, int, int]],
    out_dir: Path,
    source_path: Path,
) -> None:
    """Print suspected-duplicate pairs to console and write suspected_duplicates.txt."""
    if not phash_hits:
        return
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"Suspected duplicates in {source_path.name} — {timestamp}"
    hit_lines = [
        f"  Group {a_idx} (page {page_a}) \"{title_a}\" ~ "
        f"Group {b_idx} (page {page_b}) \"{title_b}\" [distance: {dist}]"
        for a_idx, b_idx, dist, title_a, title_b, page_a, page_b in phash_hits
    ]
    print("\n--- SUSPECTED DUPLICATES (perceptual hash) ---")
    for line in hit_lines:
        print(line)
    report_path = out_dir / "suspected_duplicates.txt"
    report_path.write_text("\n".join([header] + hit_lines), encoding="utf-8")
    print(f"  (written to {report_path})")
```

- [ ] **Step 4: Run the tests — expect them to pass**

```
.venv/Scripts/python -m pytest tests/test_image_splitter.py -k "write_phash_report" -v
```

Expected: all new tests `PASSED`.

- [ ] **Step 5: Run full suite**

```
.venv/Scripts/python -m pytest tests/ -q
```

Expected: `71 passed`.

- [ ] **Step 6: Commit**

```
git add src/pdf_extractor/image_splitter.py tests/test_image_splitter.py
git commit -m "feat: add _write_phash_report for console + file suspected-duplicate output"
```

---

## Task 3: Integrate into the streaming dedup path

The streaming path runs when `_OVERLAP_CHUNK_PAGES > 0` (the default for large files). It processes groups incrementally. We open a fitz_doc for rendering first pages and compare each written group's hash against all previously written groups.

**Files:**
- Modify: `src/pdf_extractor/image_splitter.py` (~lines 1400–1490)

- [ ] **Step 1: Open `fitz_doc` at the start of the streaming dedup block**

Find the comment `# Stream dedup + writes as groups are emitted` (~line 1400). The very next line is `import hashlib`. Insert two lines **before** `import hashlib`:

```python
            import fitz as _fitz_stream
            fitz_doc = _fitz_stream.open(str(path))
```

`fitz_doc` is initialised to `None` at the top of `split_pdf` and the existing `finally` block already calls `fitz_doc.close()` if not `None` — so the file handle is always released, even if the streaming loop raises.

- [ ] **Step 2: Add `seen_phashes` and `phash_hits` to the streaming dedup state block**

Find the block that initialises dedup state (the lines that define `seen_hashes`, `seen_semantic_keys`, etc., ~lines 1414–1422). Add two new lines at the end of that block:

```python
            seen_phashes: dict[int, tuple[int, str, int]] = {}
            phash_hits: list[tuple[int, int, int, str, str, int, int]] = []
```

So the full block now reads:

```python
            seen_hashes = set()
            seen_semantic_keys: set[str] = set()
            hash_to_groups = {}
            semantic_to_groups: dict[str, list[tuple[int, list[int]]]] = {}
            semantic_dedup_hits: list[tuple[int, int, str]] = []
            seen_phashes: dict[int, tuple[int, str, int]] = {}
            phash_hits: list[tuple[int, int, int, str, str, int, int]] = []
```

- [ ] **Step 3: Add perceptual hash tracking after each group is written**

Find the line `gc.collect()` that appears at the end of the group-write block inside the streaming for-loop (~line 1465). It follows `first_group_fallback = None`. Insert the perceptual hash block immediately **before** `gc.collect()`:

```python
                ph = _perceptual_hash(fitz_doc, group[0])
                if ph is not None:
                    for stored_hash, (stored_idx, stored_title, stored_page) in seen_phashes.items():
                        dist = _hamming_distance(ph, stored_hash)
                        if dist <= _PHASH_THRESHOLD:
                            phash_hits.append((stored_idx, group_idx, dist, stored_title, title or "", stored_page, group[0] + 1))
                    seen_phashes[ph] = (group_idx, title or "", group[0] + 1)
```

- [ ] **Step 4: Emit the phash report before `return written`**

Find the `return written` statement at the end of the streaming branch (~line 1490). Insert `_write_phash_report` call immediately before it:

```python
            _write_phash_report(phash_hits, out_dir, path)
            return written
```

- [ ] **Step 5: Run full suite**

```
.venv/Scripts/python -m pytest tests/ -q
```

Expected: `71 passed`. (No new unit tests for the integration itself — the helpers are already tested and the streaming path requires real PDFs to exercise.)

- [ ] **Step 6: Commit**

```
git add src/pdf_extractor/image_splitter.py
git commit -m "feat: add perceptual hash tracking to streaming dedup path"
```

---

## Task 4: Integrate into the non-streaming dedup path

The non-streaming path collects all groups in `flat_groups` and deduplicates them. The original `fitz_doc` is already closed by the `finally` block before this code runs, so a fresh one is opened and closed around the phash loop.

**Files:**
- Modify: `src/pdf_extractor/image_splitter.py` (~lines 1548–1599)

- [ ] **Step 1: Add the perceptual hash pass after the existing dedup loop**

Find the end of the existing dedup loop. It is the line `if semantic_key is not None: seen_semantic_keys.add(semantic_key)` (~line 1575), which is the last statement inside `for i, group in enumerate(flat_groups)`.

Immediately after that loop ends (at the same indentation level as `for i, group`), insert:

```python
    import fitz as _fitz_ph
    seen_phashes: dict[int, tuple[int, str, int]] = {}
    phash_hits: list[tuple[int, int, int, str, str, int, int]] = []
    _phash_fitz = _fitz_ph.open(str(path))
    try:
        for i, group in enumerate(deduped_groups):
            title = _group_primary_title(group, signals_dict)
            ph = _perceptual_hash(_phash_fitz, group[0])
            if ph is not None:
                for stored_hash, (stored_idx, stored_title, stored_page) in seen_phashes.items():
                    dist = _hamming_distance(ph, stored_hash)
                    if dist <= _PHASH_THRESHOLD:
                        phash_hits.append((stored_idx, i, dist, stored_title, title or "", stored_page, group[0] + 1))
                seen_phashes[ph] = (i, title or "", group[0] + 1)
    finally:
        _phash_fitz.close()
```

- [ ] **Step 2: Emit the phash report after writing the output files**

Find the loop `for group in deduped_groups:` that writes files (~line 1594) and the `return written` at its end (~line 1599). Insert `_write_phash_report` immediately before `return written`:

```python
    _write_phash_report(phash_hits, out_dir, path)
    return written
```

- [ ] **Step 3: Run full suite**

```
.venv/Scripts/python -m pytest tests/ -q
```

Expected: `71 passed`.

- [ ] **Step 4: Commit**

```
git add src/pdf_extractor/image_splitter.py
git commit -m "feat: add perceptual hash tracking to non-streaming dedup path"
```

---

## Task 5: End-to-end smoke test

**Files:** none (read-only verification)

- [ ] **Step 1: Generate the test PDF and run a split**

```
.venv/Scripts/python src/pdf_extractor/genpdf.py
.venv/Scripts/python -c "
from pdf_extractor.cli import main; raise SystemExit(main())" \
  src/pdf_extractor/Data/sample_multi_doc.pdf \
  --split-documents \
  --split-output-dir output/phash-smoke
```

If `sample_multi_doc.pdf` is not available, use any file from `src/pdf_extractor/Data/`.

- [ ] **Step 2: Verify normal output is unchanged**

Confirm that split PDF files are written as before (`[1] -> ...` lines printed). No files should be missing compared to a run without this change.

- [ ] **Step 3: Verify `suspected_duplicates.txt` behaviour**

If duplicates were detected: `suspected_duplicates.txt` should exist in the output dir and contain at least one `[distance:` line.

If no duplicates: the file should **not** exist (not an empty file).

- [ ] **Step 4: Run full suite one final time**

```
.venv/Scripts/python -m pytest tests/ -q
```

Expected: `71 passed`.
