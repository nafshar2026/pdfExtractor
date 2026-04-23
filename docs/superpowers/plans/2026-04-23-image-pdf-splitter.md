# Image PDF Splitter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the `split` CLI command to split mixed text/image PDFs into one PDF per logical document, using PaddleOCR to detect document boundaries on image-only pages.

**Architecture:** A new `image_splitter.py` module handles OCR-based boundary detection. A unified `split_pdf` function classifies pages as text or image, processes contiguous runs with the appropriate strategy (existing marker detection for text, OCR title detection for image), and writes all output to the same directory. The existing `extractor.py` is unchanged.

**Tech Stack:** `pypdf` (existing), `paddleocr`, `paddlepaddle`, `Pillow`, `numpy` (PaddleOCR transitive dep)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/pdf_extractor/image_splitter.py` | OCR helpers, `PageSignal`, `analyze_page`, grouping logic, unified `split_pdf` |
| Create | `tests/test_image_splitter.py` | All tests for the new module |
| Modify | `pyproject.toml` | Add `paddleocr`, `paddlepaddle`, `Pillow` dependencies |
| Modify | `src/pdf_extractor/__init__.py` | Export `split_pdf` |
| Modify | `src/pdf_extractor/cli.py` | Import `split_pdf`; update `--split-documents` call and help text |

`extractor.py` is **not modified**. Its private helpers (`_detect_starts`, `_extract_title`, `_page_lines`, `_sanitize_filename`) are imported directly from it by `image_splitter.py` — single-underscore names are accessible within the same package.

---

## Task 1: Add dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update pyproject.toml dependencies**

Replace the `[project]` `dependencies` list:

```toml
dependencies = [
    "pypdf>=5.4.0",
    "paddlepaddle>=2.6.0",
    "paddleocr>=2.7.0",
    "Pillow>=10.0.0",
]
```

- [ ] **Step 2: Install new dependencies**

```bash
.venv/Scripts/pip install -e .[dev]
```

Expected: pip resolves and installs `paddlepaddle`, `paddleocr`, `Pillow`, and their dependencies. PaddleOCR will download model weights (~100 MB) on first OCR call, not at install time.

- [ ] **Step 3: Verify existing tests still pass**

```bash
.venv/Scripts/python -m pytest tests/ -v
```

Expected: all existing tests PASS

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add paddleocr and Pillow dependencies"
```

---

## Task 2: PageSignal dataclass + OCR helpers

**Files:**
- Create: `src/pdf_extractor/image_splitter.py`
- Create: `tests/test_image_splitter.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_image_splitter.py`:

```python
"""Tests for image-based PDF splitting via PaddleOCR title detection."""

from pdf_extractor.image_splitter import PageSignal, _extract_ocr_texts


def test_page_signal_new_doc():
    sig = PageSignal(classification="NEW_DOC", title_text="ST-556", page_num_in_doc=None)
    assert sig.classification == "NEW_DOC"
    assert sig.title_text == "ST-556"
    assert sig.page_num_in_doc is None


def test_page_signal_continuation():
    sig = PageSignal(classification="CONTINUATION", title_text=None, page_num_in_doc=3)
    assert sig.classification == "CONTINUATION"
    assert sig.page_num_in_doc == 3


def test_extract_ocr_texts_normal():
    ocr_result = [
        [
            [[[0,0],[100,0],[100,20],[0,20]], ("ST-556 State Tax", 0.99)],
            [[[0,25],[200,25],[200,45],[0,45]], ("Transaction Return", 0.95)],
        ]
    ]
    assert _extract_ocr_texts(ocr_result) == ["ST-556 State Tax", "Transaction Return"]


def test_extract_ocr_texts_empty_result():
    assert _extract_ocr_texts([[]]) == []


def test_extract_ocr_texts_none():
    assert _extract_ocr_texts(None) == []


def test_extract_ocr_texts_none_inner():
    assert _extract_ocr_texts([[None]]) == []
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/Scripts/python -m pytest tests/test_image_splitter.py -v
```

Expected: `ModuleNotFoundError: No module named 'pdf_extractor.image_splitter'`

- [ ] **Step 3: Create image_splitter.py with PageSignal and helpers**

Create `src/pdf_extractor/image_splitter.py`:

```python
"""Image-based PDF splitting using PaddleOCR title and continuation detection."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from pypdf import PdfReader, PdfWriter

from .extractor import (
    _detect_starts,
    _extract_title,
    _page_lines,
    _sanitize_filename,
)

_CONTINUATION_RE = re.compile(r"Page\s+(\d+)\s+of\s+\d+", re.IGNORECASE)
_TITLE_MAX_CHARS = 60
_TOP_STRIP_FRACTION = 0.25
_BOTTOM_STRIP_FRACTION = 0.15
_TEXT_PAGE_MIN_CHARS = 50

_ocr_instance: PaddleOCR | None = None


def _get_ocr() -> PaddleOCR:
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
    return _ocr_instance


@dataclass(slots=True)
class PageSignal:
    classification: Literal["NEW_DOC", "CONTINUATION", "AMBIGUOUS"]
    title_text: str | None
    page_num_in_doc: int | None


def _extract_ocr_texts(ocr_result: list | None) -> list[str]:
    if not ocr_result:
        return []
    page_result = ocr_result[0]
    if not page_result:
        return []
    texts = []
    for line in page_result:
        if line and len(line) >= 2 and line[1]:
            texts.append(line[1][0])
    return texts
```

- [ ] **Step 4: Run to verify tests pass**

```bash
.venv/Scripts/python -m pytest tests/test_image_splitter.py::test_page_signal_new_doc tests/test_image_splitter.py::test_page_signal_continuation tests/test_image_splitter.py::test_extract_ocr_texts_normal tests/test_image_splitter.py::test_extract_ocr_texts_empty_result tests/test_image_splitter.py::test_extract_ocr_texts_none tests/test_image_splitter.py::test_extract_ocr_texts_none_inner -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/pdf_extractor/image_splitter.py tests/test_image_splitter.py
git commit -m "feat: add PageSignal dataclass and OCR text extraction helper"
```

---

## Task 3: analyze_page

**Files:**
- Modify: `src/pdf_extractor/image_splitter.py`
- Modify: `tests/test_image_splitter.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_image_splitter.py`:

```python
from unittest.mock import MagicMock, patch
from PIL import Image as PILImage
from pdf_extractor.image_splitter import analyze_page


def _make_ocr_result(texts: list[str]) -> list:
    """Build a PaddleOCR-format result from a list of text strings."""
    lines = []
    for text in texts:
        bbox = [[0, 0], [100, 0], [100, 20], [0, 20]]
        lines.append([bbox, (text, 0.99)])
    return [lines]


def _make_image_page(pil_image: PILImage.Image) -> MagicMock:
    img_obj = MagicMock()
    img_obj.image = pil_image
    page = MagicMock()
    page.images = [img_obj]
    return page


def _blank_image(width: int = 612, height: int = 792) -> PILImage.Image:
    return PILImage.new("RGB", (width, height), color=(255, 255, 255))


def test_analyze_page_continuation_detected():
    page = _make_image_page(_blank_image())
    # Bottom strip: "Page 3 of 6"; top strip: short title text
    mock_ocr = MagicMock()
    mock_ocr.ocr.side_effect = [
        _make_ocr_result(["Page 3 of 6"]),   # bottom strip
        _make_ocr_result(["Some Title"]),     # top strip (not reached, continuation wins)
    ]
    with patch("pdf_extractor.image_splitter._get_ocr", return_value=mock_ocr):
        sig = analyze_page(page)
    assert sig.classification == "CONTINUATION"
    assert sig.page_num_in_doc == 3
    assert sig.title_text is None


def test_analyze_page_title_detected():
    page = _make_image_page(_blank_image())
    mock_ocr = MagicMock()
    mock_ocr.ocr.side_effect = [
        _make_ocr_result([]),                     # bottom strip: no continuation
        _make_ocr_result(["ST-556 State Tax"]),   # top strip: short title
    ]
    with patch("pdf_extractor.image_splitter._get_ocr", return_value=mock_ocr):
        sig = analyze_page(page)
    assert sig.classification == "NEW_DOC"
    assert sig.title_text == "ST-556 State Tax"
    assert sig.page_num_in_doc is None


def test_analyze_page_ambiguous_long_top_text():
    page = _make_image_page(_blank_image())
    long_text = "x" * 61
    mock_ocr = MagicMock()
    mock_ocr.ocr.side_effect = [
        _make_ocr_result([]),           # bottom: no continuation
        _make_ocr_result([long_text]),  # top: too long to be a title
    ]
    with patch("pdf_extractor.image_splitter._get_ocr", return_value=mock_ocr):
        sig = analyze_page(page)
    assert sig.classification == "AMBIGUOUS"


def test_analyze_page_no_images():
    page = MagicMock()
    page.images = []
    sig = analyze_page(page)
    assert sig.classification == "AMBIGUOUS"
    assert sig.title_text is None


def test_analyze_page_page_1_of_n_is_not_continuation():
    """'Page 1 of N' must NOT trigger CONTINUATION — only X > 1 does."""
    page = _make_image_page(_blank_image())
    mock_ocr = MagicMock()
    mock_ocr.ocr.side_effect = [
        _make_ocr_result(["Page 1 of 6"]),    # bottom: page 1 of 6 → ignored
        _make_ocr_result(["Title Here"]),      # top: short title → NEW_DOC
    ]
    with patch("pdf_extractor.image_splitter._get_ocr", return_value=mock_ocr):
        sig = analyze_page(page)
    assert sig.classification == "NEW_DOC"
    assert sig.title_text == "Title Here"
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/Scripts/python -m pytest tests/test_image_splitter.py::test_analyze_page_continuation_detected -v
```

Expected: `AttributeError` — `analyze_page` not defined yet

- [ ] **Step 3: Implement analyze_page and _page_to_pil**

Append to `src/pdf_extractor/image_splitter.py`:

```python
def _page_to_pil(pdf_page) -> Image.Image | None:
    images = pdf_page.images
    if not images:
        return None
    img_obj = images[0]
    if img_obj.image is not None:
        return img_obj.image.convert("RGB")
    try:
        return Image.open(io.BytesIO(img_obj.data)).convert("RGB")
    except Exception:
        return None


def analyze_page(pdf_page) -> PageSignal:
    pil_image = _page_to_pil(pdf_page)
    if pil_image is None:
        return PageSignal(classification="AMBIGUOUS", title_text=None, page_num_in_doc=None)

    ocr = _get_ocr()
    width, height = pil_image.size

    bottom_strip = pil_image.crop((0, int(height * (1 - _BOTTOM_STRIP_FRACTION)), width, height))
    bottom_result = ocr.ocr(np.array(bottom_strip), cls=False)
    bottom_texts = _extract_ocr_texts(bottom_result)

    for text in bottom_texts:
        m = _CONTINUATION_RE.search(text)
        if m and int(m.group(1)) > 1:
            return PageSignal(
                classification="CONTINUATION",
                title_text=None,
                page_num_in_doc=int(m.group(1)),
            )

    top_strip = pil_image.crop((0, 0, width, int(height * _TOP_STRIP_FRACTION)))
    top_result = ocr.ocr(np.array(top_strip), cls=False)
    top_texts = _extract_ocr_texts(top_result)

    if top_texts:
        first_text = top_texts[0].strip()
        if first_text and len(first_text) <= _TITLE_MAX_CHARS:
            return PageSignal(
                classification="NEW_DOC",
                title_text=first_text,
                page_num_in_doc=None,
            )

    return PageSignal(classification="AMBIGUOUS", title_text=None, page_num_in_doc=None)
```

- [ ] **Step 4: Run to verify tests pass**

```bash
.venv/Scripts/python -m pytest tests/test_image_splitter.py -k "analyze_page" -v
```

Expected: all 5 `analyze_page` tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/pdf_extractor/image_splitter.py tests/test_image_splitter.py
git commit -m "feat: implement analyze_page with OCR-based title and continuation detection"
```

---

## Task 4: _group_image_pages

**Files:**
- Modify: `src/pdf_extractor/image_splitter.py`
- Modify: `tests/test_image_splitter.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_image_splitter.py`:

```python
from pdf_extractor.image_splitter import _group_image_pages


def _sig(cls, title=None, page_num=None):
    return PageSignal(classification=cls, title_text=title, page_num_in_doc=page_num)


def test_group_single_new_doc():
    signals = [(0, _sig("NEW_DOC", "Title A"))]
    assert _group_image_pages(signals) == [[0]]


def test_group_new_doc_then_continuation():
    signals = [
        (0, _sig("NEW_DOC", "Title A")),
        (1, _sig("CONTINUATION", page_num=2)),
        (2, _sig("CONTINUATION", page_num=3)),
    ]
    assert _group_image_pages(signals) == [[0, 1, 2]]


def test_group_two_consecutive_new_docs():
    """Two NEW_DOC pages back-to-back must become separate documents."""
    signals = [
        (10, _sig("NEW_DOC", "Title A")),
        (11, _sig("NEW_DOC", "Title B")),
    ]
    assert _group_image_pages(signals) == [[10], [11]]


def test_group_ambiguous_extends_current():
    signals = [
        (0, _sig("NEW_DOC", "Title")),
        (1, _sig("AMBIGUOUS")),
        (2, _sig("AMBIGUOUS")),
    ]
    assert _group_image_pages(signals) == [[0, 1, 2]]


def test_group_leading_ambiguous_forms_own_group():
    """AMBIGUOUS pages before any NEW_DOC form their own group."""
    signals = [
        (0, _sig("AMBIGUOUS")),
        (1, _sig("NEW_DOC", "Title")),
    ]
    assert _group_image_pages(signals) == [[0], [1]]


def test_group_orphan_continuation_forms_own_group():
    """CONTINUATION with no preceding group starts one (defensive)."""
    signals = [
        (0, _sig("CONTINUATION", page_num=2)),
        (1, _sig("NEW_DOC", "Title")),
    ]
    assert _group_image_pages(signals) == [[0], [1]]


def test_group_empty_input():
    assert _group_image_pages([]) == []
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/Scripts/python -m pytest tests/test_image_splitter.py -k "group" -v
```

Expected: `ImportError` — `_group_image_pages` not defined yet

- [ ] **Step 3: Implement _group_image_pages**

Append to `src/pdf_extractor/image_splitter.py`:

```python
def _group_image_pages(
    signals: list[tuple[int, PageSignal]],
) -> list[list[int]]:
    if not signals:
        return []

    groups: list[list[int]] = []
    current: list[int] = []

    for abs_idx, signal in signals:
        if signal.classification == "CONTINUATION":
            if current:
                current.append(abs_idx)
            else:
                current = [abs_idx]
        elif signal.classification == "NEW_DOC":
            if current:
                groups.append(current)
            current = [abs_idx]
        else:  # AMBIGUOUS
            if current:
                current.append(abs_idx)
            else:
                current = [abs_idx]

    if current:
        groups.append(current)

    return groups
```

- [ ] **Step 4: Run to verify tests pass**

```bash
.venv/Scripts/python -m pytest tests/test_image_splitter.py -k "group" -v
```

Expected: all 7 grouping tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/pdf_extractor/image_splitter.py tests/test_image_splitter.py
git commit -m "feat: implement _group_image_pages document boundary logic"
```

---

## Task 5: _sanitize_image_title + _write_image_group

**Files:**
- Modify: `src/pdf_extractor/image_splitter.py`
- Modify: `tests/test_image_splitter.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_image_splitter.py`:

```python
from pdf_extractor.image_splitter import _sanitize_image_title


def test_sanitize_image_title_replaces_spaces_with_underscores():
    assert _sanitize_image_title("ST-556 State Tax") == "ST-556_State_Tax"


def test_sanitize_image_title_strips_special_chars():
    assert _sanitize_image_title("Form: A/B") == "Form_AB"


def test_sanitize_image_title_empty_falls_back():
    assert _sanitize_image_title("!!!") == "Untitled"


def test_sanitize_image_title_strips_leading_trailing_underscores():
    assert _sanitize_image_title(" Hello ") == "Hello"
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/Scripts/python -m pytest tests/test_image_splitter.py -k "sanitize" -v
```

Expected: `ImportError` — `_sanitize_image_title` not defined

- [ ] **Step 3: Implement _sanitize_image_title**

Append to `src/pdf_extractor/image_splitter.py`:

```python
def _sanitize_image_title(title: str) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", title).strip()
    cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
    return cleaned or "Untitled"
```

- [ ] **Step 4: Run sanitize tests**

```bash
.venv/Scripts/python -m pytest tests/test_image_splitter.py -k "sanitize" -v
```

Expected: all 4 PASS

- [ ] **Step 5: Write tests for _write_image_group**

Append to `tests/test_image_splitter.py`:

```python
from pypdf import PdfReader, PdfWriter
from pdf_extractor.image_splitter import _write_image_group


def _make_real_reader(tmp_path, num_pages: int = 3) -> PdfReader:
    """Creates a minimal real PDF with num_pages blank pages."""
    pdf_path = tmp_path / "source.pdf"
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=612, height=792)
    with pdf_path.open("wb") as f:
        writer.write(f)
    return PdfReader(str(pdf_path))


def test_write_image_group_title_based_name(tmp_path):
    reader = _make_real_reader(tmp_path, num_pages=3)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    signals = {0: PageSignal("NEW_DOC", "ST-556 State Tax", None)}
    used_names: dict = {}

    dest = _write_image_group(reader, [0, 1], signals, out_dir, used_names)

    assert dest.name == "ST-556_State_Tax.pdf"
    assert dest.exists()
    result = PdfReader(str(dest))
    assert len(result.pages) == 2


def test_write_image_group_fallback_name(tmp_path):
    reader = _make_real_reader(tmp_path, num_pages=5)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    signals = {2: PageSignal("AMBIGUOUS", None, None)}
    used_names: dict = {}

    dest = _write_image_group(reader, [2, 3, 4], signals, out_dir, used_names)

    assert dest.name == "pages_3-5.pdf"


def test_write_image_group_duplicate_name_gets_suffix(tmp_path):
    reader = _make_real_reader(tmp_path, num_pages=4)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    signals_a = {0: PageSignal("NEW_DOC", "Form ABC", None)}
    signals_b = {2: PageSignal("NEW_DOC", "Form ABC", None)}
    used_names: dict = {}

    dest_a = _write_image_group(reader, [0, 1], signals_a, out_dir, used_names)
    dest_b = _write_image_group(reader, [2, 3], signals_b, out_dir, used_names)

    assert dest_a.name == "Form_ABC.pdf"
    assert dest_b.name == "Form_ABC (2).pdf"
```

- [ ] **Step 6: Run to verify failure**

```bash
.venv/Scripts/python -m pytest tests/test_image_splitter.py -k "write_image_group" -v
```

Expected: `ImportError` — `_write_image_group` not defined

- [ ] **Step 7: Implement _write_image_group**

Append to `src/pdf_extractor/image_splitter.py`:

```python
def _write_image_group(
    reader: PdfReader,
    group: list[int],
    signals: dict[int, PageSignal],
    out_dir: Path,
    used_names: dict[str, int],
) -> Path:
    first_idx = group[0]
    signal = signals.get(first_idx)
    raw_title = signal.title_text if signal and signal.title_text else None

    if raw_title:
        base_name = _sanitize_image_title(raw_title)
    else:
        start_page = first_idx + 1
        end_page = group[-1] + 1
        base_name = f"pages_{start_page}-{end_page}" if len(group) > 1 else f"page_{start_page}"

    used_names[base_name] = used_names.get(base_name, 0) + 1
    suffix = "" if used_names[base_name] == 1 else f" ({used_names[base_name]})"
    destination = out_dir / f"{base_name}{suffix}.pdf"

    writer = PdfWriter()
    for page_idx in group:
        writer.add_page(reader.pages[page_idx])

    with destination.open("wb") as handle:
        writer.write(handle)

    return destination
```

- [ ] **Step 8: Run all tests so far**

```bash
.venv/Scripts/python -m pytest tests/test_image_splitter.py -v
```

Expected: all tests PASS

- [ ] **Step 9: Commit**

```bash
git add src/pdf_extractor/image_splitter.py tests/test_image_splitter.py
git commit -m "feat: add title sanitization and image group PDF writer"
```

---

## Task 6: _split_image_run + _split_text_run

**Files:**
- Modify: `src/pdf_extractor/image_splitter.py`
- Modify: `tests/test_image_splitter.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_image_splitter.py`:

```python
from unittest.mock import patch, MagicMock
from pdf_extractor.image_splitter import _split_image_run, _split_text_run


def test_split_image_run_produces_one_file_per_group(tmp_path):
    reader = _make_real_reader(tmp_path, num_pages=4)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    used_names: dict = {}

    signals = [
        PageSignal("NEW_DOC", "Doc One", None),
        PageSignal("CONTINUATION", None, 2),
        PageSignal("NEW_DOC", "Doc Two", None),
        PageSignal("AMBIGUOUS", None, None),
    ]

    with patch("pdf_extractor.image_splitter.analyze_page", side_effect=signals):
        written = _split_image_run(reader, 0, 4, out_dir, used_names)

    assert len(written) == 2
    assert written[0].name == "Doc_One.pdf"
    assert written[1].name == "Doc_Two.pdf"
    result_0 = PdfReader(str(written[0]))
    assert len(result_0.pages) == 2
    result_1 = PdfReader(str(written[1]))
    assert len(result_1.pages) == 2


def test_split_text_run_splits_by_markers(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    pdf_path = tmp_path / "source.pdf"
    with pdf_path.open("wb") as f:
        writer.write(f)
    reader = PdfReader(str(pdf_path))

    page_texts = [
        "DOCUMENT 1 OF 2\nApplicable Law\nsome content here to exceed fifty chars minimum",
        "continued content page two here with enough text to count as substantial",
        "DOCUMENT 2 OF 2\nOdometer Disclosure Statement\ncontent here enough chars",
    ]
    used_names: dict = {}

    written = _split_text_run(reader, page_texts, 0, 3, out_dir, used_names)

    assert len(written) == 2
    r0 = PdfReader(str(written[0]))
    assert len(r0.pages) == 2
    r1 = PdfReader(str(written[1]))
    assert len(r1.pages) == 1
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/Scripts/python -m pytest tests/test_image_splitter.py -k "split_image_run or split_text_run" -v
```

Expected: `ImportError` — functions not defined yet

- [ ] **Step 3: Implement _split_image_run and _split_text_run**

Append to `src/pdf_extractor/image_splitter.py`:

```python
def _split_text_run(
    reader: PdfReader,
    all_page_texts: list[str],
    start: int,
    end: int,
    out_dir: Path,
    used_names: dict[str, int],
) -> list[Path]:
    run_texts = all_page_texts[start:end]
    local_starts = _detect_starts(run_texts)

    written: list[Path] = []
    for i, local_start in enumerate(local_starts):
        local_end = local_starts[i + 1] if i + 1 < len(local_starts) else len(run_texts)
        if all(not run_texts[j].strip() for j in range(local_start, local_end)):
            continue

        first_text = run_texts[local_start]
        lines = _page_lines(first_text)
        title = _extract_title(lines, -1, f"Document {len(written) + 1}")
        base_name = _sanitize_filename(title)

        used_names[base_name] = used_names.get(base_name, 0) + 1
        suffix = "" if used_names[base_name] == 1 else f" ({used_names[base_name]})"
        destination = out_dir / f"{base_name}{suffix}.pdf"

        writer = PdfWriter()
        for page_num in range(start + local_start, start + local_end):
            writer.add_page(reader.pages[page_num])

        with destination.open("wb") as handle:
            writer.write(handle)

        written.append(destination)

    return written


def _split_image_run(
    reader: PdfReader,
    start: int,
    end: int,
    out_dir: Path,
    used_names: dict[str, int],
) -> list[Path]:
    signals_list: list[tuple[int, PageSignal]] = []
    signals_dict: dict[int, PageSignal] = {}

    for abs_idx in range(start, end):
        signal = analyze_page(reader.pages[abs_idx])
        signals_list.append((abs_idx, signal))
        signals_dict[abs_idx] = signal

    groups = _group_image_pages(signals_list)

    written: list[Path] = []
    for group in groups:
        dest = _write_image_group(reader, group, signals_dict, out_dir, used_names)
        written.append(dest)

    return written
```

- [ ] **Step 4: Run to verify tests pass**

```bash
.venv/Scripts/python -m pytest tests/test_image_splitter.py -k "split_image_run or split_text_run" -v
```

Expected: both tests PASS

- [ ] **Step 5: Run full test suite**

```bash
.venv/Scripts/python -m pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/pdf_extractor/image_splitter.py tests/test_image_splitter.py
git commit -m "feat: implement _split_text_run and _split_image_run"
```

---

## Task 7: Unified split_pdf function

**Files:**
- Modify: `src/pdf_extractor/image_splitter.py`
- Modify: `tests/test_image_splitter.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_image_splitter.py`:

```python
from pdf_extractor.image_splitter import split_pdf


def _make_mixed_pdf(tmp_path, text_pages: int, image_pages: int) -> Path:
    """Creates a PDF where the first N pages have text and the rest are blank (image-like)."""
    pdf_path = tmp_path / "mixed.pdf"
    writer = PdfWriter()
    for i in range(text_pages):
        page = writer.add_blank_page(width=612, height=792)
    for _ in range(image_pages):
        writer.add_blank_page(width=612, height=792)
    with pdf_path.open("wb") as f:
        writer.write(f)
    return pdf_path


def test_split_pdf_raises_for_missing_file(tmp_path):
    from pdf_extractor.image_splitter import split_pdf
    import pytest
    with pytest.raises(FileNotFoundError):
        split_pdf(tmp_path / "nonexistent.pdf", tmp_path / "out")


def test_split_pdf_creates_output_dir(tmp_path):
    source = tmp_path / "source.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with source.open("wb") as f:
        writer.write(f)
    out_dir = tmp_path / "new_out_dir"

    fake_signal = PageSignal("NEW_DOC", "MyDoc", None)
    with patch("pdf_extractor.image_splitter.analyze_page", return_value=fake_signal):
        split_pdf(source, out_dir)

    assert out_dir.exists()


def test_split_pdf_routes_image_pages_through_analyze(tmp_path):
    """Blank pages (no text) must route through analyze_page, not text splitter."""
    source = tmp_path / "source.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    with source.open("wb") as f:
        writer.write(f)
    out_dir = tmp_path / "out"

    signals = [
        PageSignal("NEW_DOC", "Doc One", None),
        PageSignal("NEW_DOC", "Doc Two", None),
    ]
    with patch("pdf_extractor.image_splitter.analyze_page", side_effect=signals) as mock_analyze:
        result = split_pdf(source, out_dir)

    assert mock_analyze.call_count == 2
    assert len(result) == 2
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/Scripts/python -m pytest tests/test_image_splitter.py -k "test_split_pdf" -v
```

Expected: `ImportError` — `split_pdf` not defined

- [ ] **Step 3: Implement split_pdf**

Append to `src/pdf_extractor/image_splitter.py`:

```python
def split_pdf(pdf_path: str | Path, output_dir: str | Path) -> list[Path]:
    path = Path(pdf_path)
    out_dir = Path(output_dir)

    if not path.exists():
        raise FileNotFoundError(f"Input PDF not found: {path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(path))
    total = len(reader.pages)

    page_texts = [(reader.pages[i].extract_text() or "").strip() for i in range(total)]
    page_is_text = [len(t) >= _TEXT_PAGE_MIN_CHARS for t in page_texts]

    runs: list[tuple[str, int, int]] = []
    if total > 0:
        start = 0
        mode = "text" if page_is_text[0] else "image"
        for i in range(1, total):
            cur_mode = "text" if page_is_text[i] else "image"
            if cur_mode != mode:
                runs.append((mode, start, i))
                start = i
                mode = cur_mode
        runs.append((mode, start, total))

    written: list[Path] = []
    used_names: dict[str, int] = {}
    doc_idx = 0

    for run_mode, run_start, run_end in runs:
        if run_mode == "text":
            new_docs = _split_text_run(reader, page_texts, run_start, run_end, out_dir, used_names)
        else:
            new_docs = _split_image_run(reader, run_start, run_end, out_dir, used_names)

        for doc in new_docs:
            doc_idx += 1
            print(f"[{doc_idx}] → {doc.name}")

        written.extend(new_docs)

    return written
```

- [ ] **Step 4: Run to verify tests pass**

```bash
.venv/Scripts/python -m pytest tests/test_image_splitter.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Run full suite**

```bash
.venv/Scripts/python -m pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/pdf_extractor/image_splitter.py tests/test_image_splitter.py
git commit -m "feat: implement unified split_pdf handling text and image page runs"
```

---

## Task 8: Wire into CLI and __init__

**Files:**
- Modify: `src/pdf_extractor/__init__.py`
- Modify: `src/pdf_extractor/cli.py`
- Modify: `tests/test_image_splitter.py`

- [ ] **Step 1: Write the failing CLI test**

Append to `tests/test_image_splitter.py`:

```python
from unittest.mock import patch
from pdf_extractor.cli import main


def test_cli_split_documents_calls_split_pdf(tmp_path, monkeypatch):
    source = tmp_path / "source.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with source.open("wb") as f:
        writer.write(f)
    out_dir = tmp_path / "split"

    monkeypatch.setattr(
        "sys.argv",
        ["pdf-extractor", str(source), "--split-documents", "--split-output-dir", str(out_dir)],
    )

    with patch("pdf_extractor.cli.split_pdf") as mock_split:
        mock_split.return_value = [out_dir / "Doc.pdf"]
        exit_code = main()

    mock_split.assert_called_once_with(source, str(out_dir))
    assert exit_code == 0
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/Scripts/python -m pytest tests/test_image_splitter.py::test_cli_split_documents_calls_split_pdf -v
```

Expected: FAIL — `cli.py` still calls `split_pdf_by_internal_documents`

- [ ] **Step 3: Update cli.py**

In `src/pdf_extractor/cli.py`, replace the import block at the top:

```python
from .extractor import (
    ExtractedDocument,
    extract_pdf,
    find_pdf_files,
    split_pdf_by_internal_documents,
)
```

with:

```python
from .extractor import (
    ExtractedDocument,
    extract_pdf,
    find_pdf_files,
)
from .image_splitter import split_pdf
```

Then update the `--split-documents` help text in `build_parser()`. Replace:

```python
    parser.add_argument(
        "--split-documents",
        action="store_true",
        help="Split one PDF into one PDF per internal 'DOCUMENT X OF Y' section",
    )
```

with:

```python
    parser.add_argument(
        "--split-documents",
        action="store_true",
        help="Split one PDF into one PDF per logical document (handles text and scanned image pages)",
    )
```

Then in `main()`, replace:

```python
        written_files = split_pdf_by_internal_documents(
            pdf_files[0],
            args.split_output_dir,
        )
```

with:

```python
        written_files = split_pdf(pdf_files[0], args.split_output_dir)
```

- [ ] **Step 4: Update __init__.py**

Replace the contents of `src/pdf_extractor/__init__.py`:

```python
"""Public package exports for the PDF extraction library."""

from .extractor import ExtractedDocument, ExtractionResult, extract_pdf, find_pdf_files
from .image_splitter import split_pdf

__all__ = [
    "ExtractedDocument",
    "ExtractionResult",
    "extract_pdf",
    "find_pdf_files",
    "split_pdf",
]
```

- [ ] **Step 5: Run full test suite**

```bash
.venv/Scripts/python -m pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 6: Smoke test against Sample.pdf**

```bash
.venv/Scripts/python -m pdf_extractor.cli src/pdf_extractor/Data/Sample.pdf --split-documents --split-output-dir src/pdf_extractor/Data/split
```

Expected: progress lines printed like `[1] → Applicable_Law.pdf`, files written to `Data/split/`. PaddleOCR will download model weights on first run (~100 MB) — this is expected.

- [ ] **Step 7: Commit**

```bash
git add src/pdf_extractor/__init__.py src/pdf_extractor/cli.py tests/test_image_splitter.py
git commit -m "feat: wire split_pdf into CLI and public package API"
```
