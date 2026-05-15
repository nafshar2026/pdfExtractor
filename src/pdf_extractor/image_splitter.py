"""Unified PDF splitting: text extraction for digital pages, PaddleOCR for scanned pages."""

from __future__ import annotations

import gc
import io
import logging
import os
import re
import shutil
from pathlib import Path

from PIL import Image
from pypdf import PdfReader, PdfWriter

from .extractor import (
    _sanitize_filename,
)
from .title_detection import (
    _title_key,
)
from .page_analysis import (
    PageSignal,
    analyze_page,
    _shutdown_ocr_pool,
    _analyze_text_page,
    _analyze_image_page,
)
from .overlap_splitter import (
    analyze_chunk_file_signals,
    chunk_document_groups,
    fixed_page_chunks,
    iter_windowed_groups_from_chunk_files,
    windowed_groups_from_chunk_files,
    windowed_groups_from_signals,
    write_fixed_chunk_files,
)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


# Optional chunk size (in source pages) for post-group processing.
# Chunking is applied only between already-detected document groups, so it never
# splits a detected document into detached pieces.
_CHUNK_MAX_PAGES = max(0, _env_int("PDF_EXTRACTOR_CHUNK_MAX_PAGES", 0))

# Optional fixed chunk size (in source pages) for overlap-window grouping.
# When enabled, groups are produced from sliding two-chunk windows:
#   (chunk1+chunk2), (chunk2+chunk3), ...
# and only groups whose first page belongs to the left chunk are emitted,
# except the final window which emits all remaining groups. This keeps
# boundary-spanning documents together without loading the whole document graph
# at once.
_OVERLAP_CHUNK_PAGES = max(0, _env_int("PDF_EXTRACTOR_OVERLAP_CHUNK_PAGES", 0))


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


def _group_image_pages(
    signals: list[tuple[int, PageSignal]],
) -> list[list[int]]:
    """Group per-page signals into lists of page indices, one list per logical document.

    Grouping rules (applied in order):

    - **NEW_DOC**: Close the current group (if any) and start a fresh one.
      Exception 1: if the current group consists entirely of AMBIGUOUS pages (no
      anchor signal has been seen yet), those pages are prepended to the new
      NEW_DOC group rather than saved as a separate file.  This handles forms
      where the title page comes second and the first page is a cover/back side
      with no detectable title.
      Exception 2 (same-title continuation): if the new page's title matches the
      first titled page of the current group AND the new page has no page-number
      context (``page_num_in_doc is None`` — meaning title heuristic only, not a
      DOCUMENT marker or "Page 1 of N"), the page is treated as part of the current
      document rather than starting a new one.  This handles multi-page forms that
      repeat their document name as a header on every page with no "Page N of M"
      footer, which would otherwise produce one output file per page.
    - **CONTINUATION**: Append to the current group.  If ``total_pages_in_doc``
      changes between consecutive CONTINUATION pages (i.e. the declared total
      shifts from, say, 4 to 6) a new group is started — this handles back-to-back
      multi-page forms that share the same "Page N of M" footer style.
    - **AMBIGUOUS**: Append to the current group, or start a new single-page group
      if no current group exists yet.

    The first page of the PDF is always covered: if the very first signal is
    AMBIGUOUS or CONTINUATION, it still initialises a group rather than being dropped.

    Args:
        signals: Ordered list of ``(absolute_page_index, PageSignal)`` tuples
                 covering every page in the PDF.

    Returns:
        List of groups, where each group is a list of zero-based page indices.
        The groups appear in document order and together cover all input indices.
    """
    if not signals:
        return []

    # Bridge isolated OCR misses in pagination: when a page with no page-number
    # context sits between CONTINUATION pages whose numbers differ by 2 and share
    # the same total (e.g., 2/6, ?, 4/6), treat the middle page as continuation.
    normalized_signals = list(signals)
    for i in range(1, len(normalized_signals) - 1):
        prev_sig = normalized_signals[i - 1][1]
        abs_idx, cur_sig = normalized_signals[i]
        next_sig = normalized_signals[i + 1][1]

        if cur_sig.classification == "CONTINUATION" or cur_sig.page_num_in_doc is not None:
            continue
        if prev_sig.classification != "CONTINUATION" or next_sig.classification != "CONTINUATION":
            continue
        if prev_sig.page_num_in_doc is None or next_sig.page_num_in_doc is None:
            continue
        if prev_sig.total_pages_in_doc is None or next_sig.total_pages_in_doc is None:
            continue
        if prev_sig.total_pages_in_doc != next_sig.total_pages_in_doc:
            continue
        if (next_sig.page_num_in_doc - prev_sig.page_num_in_doc) != 2:
            continue

        normalized_signals[i] = (
            abs_idx,
            PageSignal(
                classification="CONTINUATION",
                title_text=None,
                page_num_in_doc=prev_sig.page_num_in_doc + 1,
                total_pages_in_doc=prev_sig.total_pages_in_doc,
            ),
        )

    groups: list[list[int]] = []
    current: list[int] = []
    current_total: int | None = None
    current_has_anchor: bool = False  # True once group contains a NEW_DOC or CONTINUATION page
    current_first_title: str | None = None  # title of the first titled page in the current group

    for abs_idx, signal in normalized_signals:
        if signal.classification == "CONTINUATION":
            current_has_anchor = True
            if (current and
                    current_total is not None and
                    signal.total_pages_in_doc is not None and
                    signal.total_pages_in_doc != current_total):
                groups.append(current)
                current = [abs_idx]
                current_has_anchor = True
                current_first_title = None
            elif current:
                current.append(abs_idx)
            else:
                current = [abs_idx]
            current_total = signal.total_pages_in_doc
        elif signal.classification == "NEW_DOC":
            # Same-title continuation: a page whose title matches the first title of the
            # current group, and which carries no page-number context, is treated as a
            # continuation of that document.  Covers multi-page forms that repeat their
            # name as a page header on every page without any "Page N of M" footer.
            # The page_num_in_doc is None guard ensures DOCUMENT N OF N pages and
            # explicit "Page 1 of N" pages are never silently merged.
            if (signal.title_text is not None
                    and current_first_title is not None
                    and _title_key(signal.title_text) == _title_key(current_first_title)
                    and signal.page_num_in_doc is None
                    and current):
                current.append(abs_idx)
            elif current and current_has_anchor:
                # Current group has real content — save it and start fresh.
                groups.append(current)
                current = [abs_idx]
                current_first_title = signal.title_text
            elif current and not current_has_anchor:
                # Current group is all-AMBIGUOUS — prepend it to this NEW_DOC group
                # rather than saving a separate untitled file.
                current.append(abs_idx)
                current_first_title = signal.title_text
            else:
                current = [abs_idx]
                current_first_title = signal.title_text
            current_total = signal.total_pages_in_doc
            current_has_anchor = True
        else:  # AMBIGUOUS
            if current:
                current.append(abs_idx)
            else:
                current = [abs_idx]
                current_has_anchor = False
            current_total = None

    if current:
        groups.append(current)

    return groups


def _sanitize_image_title(title: str) -> str:
    """Convert an OCR-extracted title into a valid, filesystem-safe filename stem.

    Strips all characters that are neither word characters, spaces, nor hyphens,
    then collapses whitespace runs to single underscores.  The result uses
    underscores (not spaces) to match the style expected by ``_write_image_group``.

    Args:
        title: Raw title string as returned by ``_select_best_title()``.

    Returns:
        Sanitized filename stem, e.g. ``"Retail_Installment_Contract"``.
        Falls back to ``"Untitled"`` for empty or all-punctuation input.
    """
    cleaned = re.sub(r"[^\w\s-]", "", title).strip()
    cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
    return cleaned or "Untitled"


def _write_image_group(
    reader: PdfReader,
    group: list[int],
    signals: dict[int, PageSignal],
    out_dir: Path,
    used_names: dict[str, int],
) -> Path:
    """Write a group of PDF pages to a single output file and return its path.

    The output filename is derived from the title detected on the first page of
    the group.  When no title is available the name falls back to
    ``"page_N"`` (single page) or ``"pages_N-M"`` (multi-page range).

    Duplicate base names within the same run are disambiguated by appending
    ``" (2)"``, ``" (3)"``, etc. — matching the convention used by
    ``split_pdf_by_internal_documents`` in extractor.py.

    Args:
        reader:     Open ``PdfReader`` for the source PDF.
        group:      Ordered list of zero-based page indices belonging to this document.
        signals:    Map of page index → ``PageSignal`` for title lookup.
        out_dir:    Directory where the output file will be written.
        used_names: Mutable counter dict shared across all groups in the same split
                    run; updated in-place to track name collisions.

    Returns:
        Path to the written PDF file.
    """
    # For paginated groups, only trust a title found on page 1 of that document.
    # Mid-sequence titles are often section headers/field labels when OCR misses
    # one footer line (e.g., page 3 in a 1..6 run).
    raw_title = None
    has_pagination = any(
        (signals.get(idx) and signals[idx].page_num_in_doc is not None)
        for idx in group
    )

    if has_pagination:
        for idx in group:
            sig = signals.get(idx)
            if sig and sig.page_num_in_doc == 1 and sig.title_text:
                raw_title = sig.title_text
                break
    else:
        # Non-paginated groups keep the prior behavior: first available title.
        for idx in group:
            sig = signals.get(idx)
            if sig and sig.title_text:
                raw_title = sig.title_text
                break

    if raw_title:
        base_name = _sanitize_image_title(raw_title)
    else:
        start_page = group[0] + 1
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


def _group_primary_title(group: list[int], signals: dict[int, PageSignal]) -> str | None:
    """Return the preferred title for a grouped document.

    Mirrors naming logic used when writing output files: for paginated groups,
    trust the title on page 1 of the logical document; otherwise use the first
    available title in the group.
    """
    has_pagination = any(
        (signals.get(idx) and signals[idx].page_num_in_doc is not None)
        for idx in group
    )

    if has_pagination:
        for idx in group:
            sig = signals.get(idx)
            if sig and sig.page_num_in_doc == 1 and sig.title_text:
                return sig.title_text
    else:
        for idx in group:
            sig = signals.get(idx)
            if sig and sig.title_text:
                return sig.title_text

    return None


def _semantic_title_key(raw_title: str | None) -> str | None:
    """Build a dedup key from title text that survives punctuation/noise.

    Rules:
    - remove trailing parenthetical copy markers like "(2)"
    - keep only alphanumeric characters
    - uppercase for case-insensitive comparison
    """
    if not raw_title:
        return None

    cleaned = re.sub(r"\s*\(\d+\)\s*$", "", raw_title).strip()
    key = _title_key(cleaned)
    return key or None


def _group_declared_total(group: list[int], signals: dict[int, PageSignal]) -> int | None:
    """Return declared total pages (N in Page X of N) when present."""
    for idx in group:
        sig = signals.get(idx)
        if sig and sig.total_pages_in_doc is not None:
            return sig.total_pages_in_doc
    return None


def _chunk_document_groups(groups: list[list[int]], max_pages: int) -> list[list[list[int]]]:
    return chunk_document_groups(groups, max_pages)


def _fixed_page_chunks(total_pages: int, chunk_pages: int) -> list[tuple[int, int]]:
    return fixed_page_chunks(total_pages, chunk_pages)


def _windowed_groups_from_signals(
    signals: list[tuple[int, PageSignal]],
    chunk_pages: int,
) -> list[list[int]]:
    return windowed_groups_from_signals(
        signals,
        chunk_pages,
        group_pages_fn=_group_image_pages,
    )


def _write_fixed_chunk_files(
    reader: PdfReader,
    chunks: list[tuple[int, int]],
    chunk_dir: Path,
) -> list[Path]:
    return write_fixed_chunk_files(reader, chunks, chunk_dir)


def _analyze_chunk_file_signals(
    chunk_path: Path,
    abs_start: int,
) -> list[tuple[int, PageSignal]]:
    return analyze_chunk_file_signals(
        chunk_path,
        abs_start,
        analyze_page_fn=analyze_page,
        shutdown_ocr_pool_fn=_shutdown_ocr_pool,
    )


def _windowed_groups_from_chunk_files(
    chunk_ranges: list[tuple[int, int]],
    chunk_paths: list[Path],
) -> tuple[list[list[int]], dict[int, PageSignal]]:
    return windowed_groups_from_chunk_files(
        chunk_ranges,
        chunk_paths,
        group_pages_fn=_group_image_pages,
        analyze_chunk_file_signals_fn=_analyze_chunk_file_signals,
    )


def _iter_windowed_groups_from_chunk_files(
    chunk_ranges: list[tuple[int, int]],
    chunk_paths: list[Path],
):
    return iter_windowed_groups_from_chunk_files(
        chunk_ranges,
        chunk_paths,
        group_pages_fn=_group_image_pages,
        analyze_chunk_file_signals_fn=_analyze_chunk_file_signals,
    )


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


def split_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    *,
    verbose: bool = False,
    chunk_max_pages: int | None = None,
) -> list[Path]:
    """Unified PDF splitter: analyzes every page for document boundary signals,
    groups them, and writes one PDF per logical document.

    Text pages (≥50 extracted chars) are analyzed via pattern matching on the
    extracted text.  Image/scanned pages fall back to PaddleOCR.  The grouping
    logic is identical regardless of page type.

    Args:
        pdf_path:   Path to the source PDF.
        output_dir: Directory where split output files are written.
        verbose:    When True, prints a per-page signal table to stdout after
                analysis — useful for diagnosing mis-splits on new files.
        chunk_max_pages:
                Optional page budget for batching whole document groups.
                Chunk boundaries are inserted only between groups.
    """
    path = Path(pdf_path)
    out_dir = Path(output_dir)

    if not path.exists():
        raise FileNotFoundError(f"Input PDF not found: {path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(path))

    fitz_doc = None

    signals_dict: dict[int, PageSignal] = {}
    groups: list[list[int]] = []

    try:
        overlap_chunk_pages = _OVERLAP_CHUNK_PAGES

        if overlap_chunk_pages > 0:
            chunks = _fixed_page_chunks(len(reader.pages), overlap_chunk_pages)

            # Persist chunk PDFs under the output directory so the source/input
            # location stays untouched and operators can inspect chunk artifacts.
            chunk_dir = out_dir / "_overlap_chunks"
            if chunk_dir.exists():
                shutil.rmtree(chunk_dir)
            chunk_dir.mkdir(parents=True, exist_ok=True)

            chunk_paths = _write_fixed_chunk_files(reader, chunks, chunk_dir)
            windows = max(0, len(chunks) - 1)

            if verbose:
                print("\n--- OVERLAP WINDOW REPORT ---")
                print(f"Fixed chunk size: {overlap_chunk_pages}")
                print(f"Chunks: {len(chunks)}")
                print(f"Windows: {windows}")
                print(f"Chunk files directory: {chunk_dir}")
                print("Raw chunk PDFs were written to disk and processed in sliding pairs.")
                print("Per-page signal table is skipped in overlap mode to preserve memory.")

            # Stream dedup + writes as groups are emitted so outputs appear
            # progressively; avoid holding all groups/signals until the end.
            import fitz as _fitz_stream
            fitz_doc = _fitz_stream.open(str(path))
            import hashlib

            def group_hash(group):
                writer = PdfWriter()
                for idx in group:
                    writer.add_page(reader.pages[idx])
                buf = io.BytesIO()
                writer.write(buf)
                h = hashlib.sha256(buf.getvalue()).hexdigest()
                del writer, buf
                return h

            seen_hashes = set()
            seen_semantic_keys: set[str] = set()
            hash_to_groups = {}
            semantic_to_groups: dict[str, list[tuple[int, list[int]]]] = {}
            semantic_dedup_hits: list[tuple[int, int, str]] = []
            seen_phashes: dict[int, tuple[int, str, int]] = {}
            phash_hits: list[tuple[int, int, int, str, str, int, int]] = []

            used_names: dict[str, int] = {}
            written: list[Path] = []
            groups_before_dedup = 0

            emitted_any = False
            first_group_fallback: tuple[list[int], dict[int, PageSignal]] | None = None

            for group_idx, (group, group_signals) in enumerate(
                _iter_windowed_groups_from_chunk_files(chunks, chunk_paths)
            ):
                groups_before_dedup += 1
                h = group_hash(group)
                if h not in hash_to_groups:
                    hash_to_groups[h] = []
                hash_to_groups[h].append((group_idx, group))

                if first_group_fallback is None:
                    first_group_fallback = (group, group_signals)

                if h in seen_hashes:
                    continue

                title = _group_primary_title(group, group_signals)
                title_key = _semantic_title_key(title)
                total = _group_declared_total(group, group_signals)
                semantic_key = None

                if title_key and (total is not None or len(group) > 1):
                    semantic_key = f"{title_key}|{total if total is not None else len(group)}"
                    semantic_to_groups.setdefault(semantic_key, []).append((group_idx, group))

                if semantic_key is not None and semantic_key in seen_semantic_keys:
                    first_idx = semantic_to_groups[semantic_key][0][0]
                    semantic_dedup_hits.append((first_idx, group_idx, semantic_key))
                    continue

                seen_hashes.add(h)
                if semantic_key is not None:
                    seen_semantic_keys.add(semantic_key)

                dest = _write_image_group(reader, group, group_signals, out_dir, used_names)
                written.append(dest)
                print(f"[{len(written)}] -> {dest.name}", flush=True)
                emitted_any = True
                first_group_fallback = None  # no longer needed once at least one group is written
                ph = _perceptual_hash(fitz_doc, group[0])
                if ph is not None:
                    for stored_hash, (stored_idx, stored_title, stored_page) in seen_phashes.items():
                        dist = _hamming_distance(ph, stored_hash)
                        if dist <= _PHASH_THRESHOLD:
                            phash_hits.append((stored_idx, group_idx, dist, stored_title, title or "", stored_page, group[0] + 1))
                    seen_phashes[ph] = (group_idx, title or "", group[0] + 1)
                gc.collect()

            # If no group passed dedup, forcibly write the first group seen as a fallback
            if not emitted_any and first_group_fallback is not None:
                group, group_signals = first_group_fallback
                dest = _write_image_group(reader, group, group_signals, out_dir, used_names)
                written.append(dest)
                print(f"[{len(written)}] -> {dest.name}", flush=True)
                if verbose:
                    print("\n--- PATCH: No groups emitted, wrote fallback output from first chunk ---")

            if verbose:
                print(f"\n--- DEDUPLICATION REPORT ---")
                print(f"Groups before dedup: {groups_before_dedup}")
                print(f"Groups after dedup: {len(written)}")
                print(f"Unique hashes: {len(seen_hashes)}")
                for h, group_list in sorted(hash_to_groups.items(), key=lambda x: -len(x[1])):
                    if len(group_list) > 1:
                        print(f"  DUPLICATE {h[:8]}... appears {len(group_list)} times: groups {[g[0] for g in group_list]}")
                for first_idx, dup_idx, semantic_key in semantic_dedup_hits:
                    print(
                        "  SEMANTIC DUPLICATE "
                        f"groups {first_idx} and {dup_idx} via key {semantic_key[:40]}..."
                    )

            _write_phash_report(phash_hits, out_dir, path)
            return written

        else:
            import fitz
            fitz_doc = fitz.open(str(path))
            signals_list: list[tuple[int, PageSignal]] = []
            for i in range(len(reader.pages)):
                signal = analyze_page(reader.pages[i], fitz_doc=fitz_doc, page_idx=i)
                signals_list.append((i, signal))
                signals_dict[i] = signal
                gc.collect()  # release OCR buffers from the processed page

            if verbose:
                print(f"\n{'Page':>5}  {'Classification':<15}  {'PgNum':>5}  {'Total':>5}  Title")
                print("-" * 80)
                for i, sig in signals_list:
                    title_str = (sig.title_text or "")[:40]
                    pg = str(sig.page_num_in_doc) if sig.page_num_in_doc is not None else "-"
                    tot = str(sig.total_pages_in_doc) if sig.total_pages_in_doc is not None else "-"
                    print(f"{i + 1:>5}  {sig.classification:<15}  {pg:>5}  {tot:>5}  {title_str}")
                print()

            groups = _group_image_pages(signals_list)

            overlap_chunk_pages = 0
    finally:
        if fitz_doc is not None:
            fitz_doc.close()
        _shutdown_ocr_pool()

    effective_chunk_max = _CHUNK_MAX_PAGES if chunk_max_pages is None else max(0, chunk_max_pages)
    grouped_chunks = _chunk_document_groups(groups, effective_chunk_max)

    if verbose and effective_chunk_max > 0:
        print("\n--- CHUNKING REPORT ---")
        print(f"Chunk max pages: {effective_chunk_max}")
        print(f"Chunks: {len(grouped_chunks)}")
        for idx, chunk in enumerate(grouped_chunks, start=1):
            pages = sum(len(g) for g in chunk)
            print(f"  Chunk {idx}: {len(chunk)} group(s), {pages} page(s)")

    # Hash-based deduplication across the whole file (not only back-to-back groups).
    import hashlib

    def group_hash(group):
        writer = PdfWriter()
        for idx in group:
            writer.add_page(reader.pages[idx])
        buffer = io.BytesIO()
        writer.write(buffer)
        return hashlib.sha256(buffer.getvalue()).hexdigest()

    seen_hashes = set()
    seen_semantic_keys: set[str] = set()
    deduped_groups = []
    hash_to_groups = {}
    semantic_to_groups: dict[str, list[tuple[int, list[int]]]] = {}
    semantic_dedup_hits: list[tuple[int, int, str]] = []
    flat_groups = [group for chunk in grouped_chunks for group in chunk]

    for i, group in enumerate(flat_groups):
        h = group_hash(group)
        if h not in hash_to_groups:
            hash_to_groups[h] = []
        hash_to_groups[h].append((i, group))
        if h not in seen_hashes:
            title = _group_primary_title(group, signals_dict)
            title_key = _semantic_title_key(title)
            total = _group_declared_total(group, signals_dict)
            semantic_key = None

            # Only use semantic dedup when title + page-count context exists.
            # This avoids over-merging unrelated single-page forms with generic titles.
            if title_key and (total is not None or len(group) > 1):
                semantic_key = f"{title_key}|{total if total is not None else len(group)}"
                semantic_to_groups.setdefault(semantic_key, []).append((i, group))

            if semantic_key is not None and semantic_key in seen_semantic_keys:
                first_idx = semantic_to_groups[semantic_key][0][0]
                semantic_dedup_hits.append((first_idx, i, semantic_key))
                continue

            deduped_groups.append(group)
            seen_hashes.add(h)
            if semantic_key is not None:
                seen_semantic_keys.add(semantic_key)

    import fitz as _fitz_ph
    seen_phashes: dict[int, tuple[int, str, int]] = {}
    phash_hits: list[tuple[int, int, int, str, str, int, int]] = []
    _phash_fitz = _fitz_ph.open(str(path))
    try:
        for deduped_idx, group in enumerate(deduped_groups):
            title = _group_primary_title(group, signals_dict)
            ph = _perceptual_hash(_phash_fitz, group[0])
            if ph is not None:
                for stored_hash, (stored_idx, stored_title, stored_page) in seen_phashes.items():
                    dist = _hamming_distance(ph, stored_hash)
                    if dist <= _PHASH_THRESHOLD:
                        phash_hits.append((stored_idx, deduped_idx, dist, stored_title, title or "", stored_page, group[0] + 1))
                seen_phashes[ph] = (deduped_idx, title or "", group[0] + 1)
    finally:
        _phash_fitz.close()

    if verbose:
        print(f"\n--- DEDUPLICATION REPORT ---")
        print(f"Groups before dedup: {len(flat_groups)}")
        print(f"Groups after dedup: {len(deduped_groups)}")
        print(f"Unique hashes: {len(seen_hashes)}")
        for h, group_list in sorted(hash_to_groups.items(), key=lambda x: -len(x[1])):
            if len(group_list) > 1:
                print(f"  DUPLICATE {h[:8]}... appears {len(group_list)} times: groups {[g[0] for g in group_list]}")
        for first_idx, dup_idx, semantic_key in semantic_dedup_hits:
            print(
                "  SEMANTIC DUPLICATE "
                f"groups {first_idx} and {dup_idx} via key {semantic_key[:40]}..."
            )

    written: list[Path] = []
    used_names: dict[str, int] = {}

    for group in deduped_groups:
        dest = _write_image_group(reader, group, signals_dict, out_dir, used_names)
        written.append(dest)
        print(f"[{len(written)}] -> {dest.name}", flush=True)

    _write_phash_report(phash_hits, out_dir, path)
    return written
