"""Image-based PDF splitting using PaddleOCR title and continuation detection."""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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


_logger = logging.getLogger(__name__)


def _page_to_pil(pdf_page) -> Image.Image | None:
    images = pdf_page.images
    if not images:
        return None
    # Use only the first embedded image. For typical scanned-document PDFs each page
    # is a single raster image; additional images (logos, stamps) are ignored.
    img_obj = images[0]
    if img_obj.image is not None:
        return img_obj.image.convert("RGB")
    try:
        return Image.open(io.BytesIO(img_obj.data)).convert("RGB")
    except Exception as exc:
        _logger.debug("_page_to_pil: could not decode image data: %s", exc)
        return None


def analyze_page(pdf_page) -> PageSignal:
    pil_image = _page_to_pil(pdf_page)
    if pil_image is None:
        return PageSignal(classification="AMBIGUOUS", title_text=None, page_num_in_doc=None)

    ocr = _get_ocr()
    width, height = pil_image.size
    if width == 0 or height == 0:
        return PageSignal(classification="AMBIGUOUS", title_text=None, page_num_in_doc=None)

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


def _sanitize_image_title(title: str) -> str:
    """Sanitize a document title into a safe filename stem.

    - Removes non-alphanumeric/non-dash characters
    - Replaces spaces with underscores
    - Strips leading/trailing underscores
    - Returns "Untitled" for empty result

    Args:
        title: Raw title text extracted from OCR

    Returns:
        Safe filename stem (e.g., "ST-556_State_Tax")
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
    """Extract and write a group of pages to a PDF.

    Names the output based on:
    1. The title from the first page's PageSignal (if present)
    2. Fallback: "pages_X-Y" (1-based page numbers)

    Handles duplicate names by appending " (2)", " (3)", etc.

    Args:
        reader: Source PDF reader
        group: List of absolute page indices to extract
        signals: Dict mapping page index to PageSignal
        out_dir: Output directory Path
        used_names: Dict tracking name usage counts (modified in-place)

    Returns:
        Path to the written PDF file
    """
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


def _split_text_run(
    reader: PdfReader,
    all_page_texts: list[str],
    start: int,
    end: int,
    out_dir: Path,
    used_names: dict[str, int],
) -> list[Path]:
    """Split a run of pages by text-based document boundaries.

    Detects document starts within the slice using _detect_starts (which works on local indices),
    then adjusts to absolute page indices when writing to the PDF.

    Args:
        reader: Source PDF reader
        all_page_texts: Full list of page texts (indexed 0 to len-1)
        start: Start index (inclusive) into all_page_texts
        end: End index (exclusive) into all_page_texts
        out_dir: Output directory Path
        used_names: Dict tracking name usage counts (modified in-place)

    Returns:
        List of Paths to written PDF files
    """
    run_texts = all_page_texts[start:end]
    local_starts = _detect_starts(run_texts)

    written: list[Path] = []
    for i, local_start in enumerate(local_starts):
        local_end = local_starts[i + 1] if i + 1 < len(local_starts) else len(run_texts)
        if all(not run_texts[j].strip() for j in range(local_start, local_end)):
            continue

        first_text = run_texts[local_start]
        lines = _page_lines(first_text)
        title = _extract_title(lines, -1, f"Document {len(written) + 1}")  # -1: no DOCUMENT marker index in this context
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
    """Split a run of pages by image-based signals (titles and page numbers).

    Analyzes each page in the range [start, end) to detect document boundaries,
    groups them by _group_image_pages, and writes each group as a separate PDF.

    Args:
        reader: Source PDF reader
        start: Start index (inclusive) into reader.pages
        end: End index (exclusive) into reader.pages
        out_dir: Output directory Path
        used_names: Dict tracking name usage counts (modified in-place)

    Returns:
        List of Paths to written PDF files
    """
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


def split_pdf(pdf_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Unified PDF splitter handling both text and image page runs.

    Classifies each page as "text" (≥50 chars) or "image" (<50 chars) by extracting text,
    groups consecutive same-mode pages into runs, and delegates to _split_text_run or
    _split_image_run accordingly.

    Args:
        pdf_path: Path to input PDF file
        output_dir: Path to output directory (created if not exists)

    Returns:
        List of Paths to written PDF files (in order)

    Raises:
        FileNotFoundError: If pdf_path does not exist
    """
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
