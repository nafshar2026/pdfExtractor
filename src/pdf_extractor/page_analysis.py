"""Per-page boundary signal analysis for text and scanned PDF pages."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

from .extractor import _DOC_MARKER_RE, _extract_title, _page_lines
from .ocr_runtime import OcrRuntime
from .title_detection import (
    _BOTTOM_STRIP_FRACTION,
    _TOP_STRIP_FRACTION,
    _extract_text_title,
    _infer_content_title,
    _looks_like_form_code_title,
    _select_best_title,
    _title_appears_top_and_footer,
)

logging.getLogger("ppocr").setLevel(logging.ERROR)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


# Matches "Page 3 of 6" in any page type — used to identify continuations and first pages.
# OCR commonly collapses spaces in pagination text (e.g. "PAGE5OF6", "Page 1 of6").
_CONTINUATION_RE = re.compile(r"Page\s*(\d+)\s*of\s*(\d+)", re.IGNORECASE)

# Maximum width (pixels) for images passed to PaddleOCR.  High-resolution scans
# (600 DPI letter page = 5100 px wide) are downscaled to this before OCR so that
# PaddlePaddle's inference buffers stay within the container memory limit.
# Default 1500 px gives ~150 DPI equivalent — more than sufficient for text
# recognition on standard forms.
_OCR_MAX_WIDTH_DEFAULT = 1500

# A page with at least this many extracted characters is treated as a digital text page;
# fewer characters means the page is blank or image-only and falls back to OCR.
_TEXT_PAGE_MIN_CHARS = 50

# Optional global cap for rendered OCR width to trade speed/accuracy for memory.
_OCR_MAX_WIDTH = max(400, _env_int("PDF_EXTRACTOR_OCR_MAX_WIDTH", _OCR_MAX_WIDTH_DEFAULT))


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# When enabled, OCR runs in a dedicated subprocess so memory can be reclaimed by
# recycling the worker process instead of accumulating in the main process.
_OCR_ISOLATED = _env_flag("PDF_EXTRACTOR_OCR_ISOLATED", default=False)

# Number of OCR calls before recycling the isolated worker. 0 disables recycle.
# Keep this conservative by default for memory-constrained jobs.
_OCR_RECYCLE_CALLS = max(0, _env_int("PDF_EXTRACTOR_OCR_RECYCLE_CALLS", 6))

# Retry count when the isolated worker dies mid-inference.
_OCR_POOL_RETRIES = max(0, _env_int("PDF_EXTRACTOR_OCR_POOL_RETRIES", 1))

_OCR_RUNTIME = OcrRuntime(
    isolated=_OCR_ISOLATED,
    recycle_calls=_OCR_RECYCLE_CALLS,
    pool_retries=_OCR_POOL_RETRIES,
)


def _shutdown_ocr_pool() -> None:
    _OCR_RUNTIME.shutdown()


def _ocr_infer(image_array: np.ndarray):
    """Run OCR via the shared runtime manager."""
    return _OCR_RUNTIME.infer(image_array)


@dataclass(slots=True)
class PageSignal:
    """Boundary classification for a single PDF page.

    Produced by `analyze_page()` and consumed by `_group_image_pages()` to
    determine how consecutive pages are grouped into logical documents.

    Attributes:
        classification:      "NEW_DOC" — this page starts a new logical document.
                             "CONTINUATION" — this page continues the current document
                               (a "Page N of M" footer confirmed it is not page 1).
                             "AMBIGUOUS" — no boundary signal was detected; the page
                               is attached to whichever document came immediately before.
        title_text:          Document title extracted from this page, or None.  Only
                             populated for NEW_DOC pages; used as the output filename.
        page_num_in_doc:     The page number within its logical document (e.g. 2 for
                             "Page 2 of 4"), or None when the marker is absent.
        total_pages_in_doc:  Total page count declared by the "Page N of M" marker, or
                             None when the marker is absent.  Used by `_group_image_pages`
                             to detect mid-sequence document switches.
    """

    classification: Literal["NEW_DOC", "CONTINUATION", "AMBIGUOUS"]
    title_text: str | None
    page_num_in_doc: int | None
    total_pages_in_doc: int | None = None


def _extract_ocr_texts(ocr_result: list | None) -> list[str]:
    """Flatten PaddleOCR output to a plain list of recognised text strings.

    PaddleOCR returns a nested structure:
      ``[[[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, confidence)], ...]]``
    This function extracts only the text strings in line order, discarding
    bounding-box coordinates and confidence scores.

    Args:
        ocr_result: Raw return value from ``PaddleOCR.ocr()``, or None.

    Returns:
        Ordered list of recognised text strings.  Empty list if the input is
        None or if PaddleOCR found nothing on the image strip.
    """
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


def _render_page_fitz(fitz_doc, page_idx: int) -> Image.Image | None:
    """Render a PDF page to a PIL Image using PyMuPDF at a controlled width.

    Unlike pypdf's image-extraction approach, PyMuPDF renders on demand and
    does not cache decompressed pixel data in the document object.  This keeps
    memory bounded regardless of how many pages have been processed.

    The output width is capped at _OCR_MAX_WIDTH so that PaddlePaddle's
    inference buffers stay within the container memory limit.

    Args:
        fitz_doc:  Open ``fitz.Document`` for the source PDF.
        page_idx:  Zero-based page index.

    Returns:
        Rendered page as an RGB PIL Image, or None on failure.
    """
    try:
        page = fitz_doc[page_idx]
        w_pt = page.rect.width
        if w_pt <= 0:
            return None
        scale = _OCR_MAX_WIDTH / w_pt
        mat = __import__("fitz").Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        del pix
        return img
    except Exception as exc:
        _logger.debug("_render_page_fitz: failed for page %d: %s", page_idx, exc)
        return None


def _analyze_text_page(text: str, fitz_doc=None, page_idx: int = 0) -> PageSignal:
    """Analyze a digitally-extracted text page for document boundary signals."""
    lines = _page_lines(text)

    # DOCUMENT N OF N marker → explicit boundary embedded by the document assembly system.
    if _DOC_MARKER_RE.search(text):
        marker_idx = next(
            (i for i, ln in enumerate(lines) if _DOC_MARKER_RE.search(ln)), -1
        )
        raw_title = _extract_title(lines, marker_idx, "")
        return PageSignal(
            classification="NEW_DOC",
            title_text=raw_title or None,
            page_num_in_doc=1,
        )

    # Page N of M — search the full text; forms place this in headers, footers, or
    # sidebars, so restricting to the last N lines causes misses.
    for line in lines:
        m = _CONTINUATION_RE.search(line)
        if m:
            page_num = int(m.group(1))
            total = int(m.group(2))
            if page_num > 1:
                return PageSignal(
                    classification="CONTINUATION",
                    title_text=None,
                    page_num_in_doc=page_num,
                    total_pages_in_doc=total,
                )
            if page_num == 1:
                # prefer_last: complex forms have disclosure boxes before the contract name
                title = _extract_text_title(lines, prefer_last=True, max_words=12)
                return PageSignal(
                    classification="NEW_DOC",
                    title_text=title,
                    page_num_in_doc=1,
                    total_pages_in_doc=total if total > 1 else None,
                )

    # No explicit page markers — a detectable title signals a new document;
    # no title means this page belongs to whatever came before.
    inferred = _infer_content_title(text)
    if inferred:
        return PageSignal(classification="NEW_DOC", title_text=inferred, page_num_in_doc=None)
    title = _extract_text_title(lines)
    if title:
        return PageSignal(classification="NEW_DOC", title_text=title, page_num_in_doc=None)
    return PageSignal(classification="AMBIGUOUS", title_text=None, page_num_in_doc=None)


def _analyze_image_page(fitz_doc, page_idx: int) -> PageSignal:
    """Analyze a scanned/image page using PaddleOCR.

    Uses PyMuPDF to render the page at a controlled resolution so that
    pypdf's image-cache does not accumulate over hundreds of pages.
    Large intermediate objects are explicitly deleted as soon as they are
    no longer needed so gc.collect() can reclaim them immediately.

    Args:
        fitz_doc:  Open ``fitz.Document`` for the source PDF.
        page_idx:  Zero-based index of the page to analyze.
    """
    pil_image = _render_page_fitz(fitz_doc, page_idx)
    if pil_image is None:
        return PageSignal(classification="AMBIGUOUS", title_text=None, page_num_in_doc=None)

    width, height = pil_image.size
    if width == 0 or height == 0:
        del pil_image
        return PageSignal(classification="AMBIGUOUS", title_text=None, page_num_in_doc=None)

    # --- Bottom strip: page-number footer detection ---
    bottom_strip = pil_image.crop((0, int(height * (1 - _BOTTOM_STRIP_FRACTION)), width, height))
    bottom_arr = np.array(bottom_strip)
    del bottom_strip
    bottom_result = _ocr_infer(bottom_arr)
    del bottom_arr
    bottom_texts = _extract_ocr_texts(bottom_result)

    page_one_total: int | None = None
    for text in bottom_texts:
        m = _CONTINUATION_RE.search(text)
        if m:
            page_num = int(m.group(1))
            total = int(m.group(2))
            if page_num > 1:
                del pil_image, bottom_result
                return PageSignal(
                    classification="CONTINUATION",
                    title_text=None,
                    page_num_in_doc=page_num,
                    total_pages_in_doc=total,
                )
            if page_num == 1 and total > 1:
                page_one_total = total

    # --- Top strip: title detection ---
    top_strip = pil_image.crop((0, 0, width, int(height * _TOP_STRIP_FRACTION)))
    top_arr = np.array(top_strip)
    del top_strip
    top_result = _ocr_infer(top_arr)
    del top_arr
    top_texts = _extract_ocr_texts(top_result)

    # Only include actual footer content (pagination "Page N of M") in suppression list,
    # not all bottom-strip text which may include body content from middle of page.
    # This prevents disclaimer text from being treated as form-code footers.
    footer_texts_for_title = [t for t in bottom_texts if _CONTINUATION_RE.search(t)]
    ocr_combined = " ".join(bottom_texts + top_texts)

    # Fallback: if bottom-strip OCR missed pagination, scan full-page OCR text
    # for Page N of M before relying on title-only splitting.
    full_result = None
    if page_one_total is None:
        full_arr = np.array(pil_image)
        full_result = _ocr_infer(full_arr)
        del full_arr
        full_texts = _extract_ocr_texts(full_result)
        for text in full_texts:
            m = _CONTINUATION_RE.search(text)
            if not m:
                continue
            page_num = int(m.group(1))
            total = int(m.group(2))
            if page_num > 1:
                del bottom_result, top_result, pil_image, full_result
                return PageSignal(
                    classification="CONTINUATION",
                    title_text=None,
                    page_num_in_doc=page_num,
                    total_pages_in_doc=total,
                )
            if page_num == 1 and total > 1:
                page_one_total = total
                break

    del bottom_texts, top_texts
    inferred = _infer_content_title(ocr_combined)
    if inferred:
        del bottom_result, top_result, pil_image
        if full_result is not None:
            del full_result
        return PageSignal(
            classification="NEW_DOC",
            title_text=inferred,
            page_num_in_doc=1 if page_one_total else None,
            total_pages_in_doc=page_one_total,
        )

    best_title = _select_best_title(top_result, footer_texts=footer_texts_for_title)

    if best_title and _looks_like_form_code_title(best_title):
        if full_result is None:
            full_arr = np.array(pil_image)
            full_result = _ocr_infer(full_arr)
            del full_arr
        if _title_appears_top_and_footer(full_result, best_title):
            best_title = None

    # Fallback: "Page 1 of N" confirmed but top-strip OCR missed the title
    # (common when the worker recycles mid-chunk). Re-scan the full page and
    # restrict to the top 35% by bounding-box centre-Y before title detection.
    if best_title is None and page_one_total is not None:
        if full_result is None:
            full_arr = np.array(pil_image)
            full_result = _ocr_infer(full_arr)
            del full_arr
        if full_result and full_result[0]:
            ph = max(
                (float(max(pt[1] for pt in line[0]))
                 for line in full_result[0]
                 if line and len(line) >= 2 and line[0]),
                default=1.0,
            )
            top_band = [
                line for line in full_result[0]
                if line and len(line) >= 2 and line[0]
                and (float(sum(pt[1] for pt in line[0])) / len(line[0])) <= ph * 0.35
            ]
            if top_band:
                best_title = _select_best_title([top_band], footer_texts=footer_texts_for_title)

    del pil_image
    del bottom_result, top_result
    if full_result is not None:
        del full_result

    if best_title:
        return PageSignal(
            classification="NEW_DOC",
            title_text=best_title,
            page_num_in_doc=1 if page_one_total else None,
            total_pages_in_doc=page_one_total,
        )

    if page_one_total is not None:
        return PageSignal(
            classification="NEW_DOC",
            title_text=None,
            page_num_in_doc=1,
            total_pages_in_doc=page_one_total,
        )

    return PageSignal(classification="AMBIGUOUS", title_text=None, page_num_in_doc=None)


def analyze_page(pdf_page, fitz_doc=None, page_idx: int = 0) -> PageSignal:
    """Unified page analyzer: text extraction for digital pages, OCR for scanned pages.

    The boundary signals (Page N of M, DOCUMENT N OF N) are structural and apply
    regardless of whether the page content is digitally encoded or image-scanned.

    Args:
        pdf_page:  pypdf PageObject (used for text extraction).
        fitz_doc:  Open fitz.Document for the same PDF (used for image rendering).
        page_idx:  Zero-based index of this page (passed to fitz_doc).
    """
    text = (pdf_page.extract_text() or "").strip()
    if len(text) >= _TEXT_PAGE_MIN_CHARS:
        return _analyze_text_page(text, fitz_doc=fitz_doc, page_idx=page_idx)
    return _analyze_image_page(fitz_doc, page_idx)
