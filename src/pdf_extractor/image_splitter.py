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
