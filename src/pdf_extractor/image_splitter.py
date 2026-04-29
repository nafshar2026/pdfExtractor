"""Unified PDF splitting: text extraction for digital pages, PaddleOCR for scanned pages."""

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

# Suppress noisy per-call warnings from PaddleOCR's internal logger.
logging.getLogger("ppocr").setLevel(logging.ERROR)

from .extractor import (
    _DOC_MARKER_RE,
    _extract_title,
    _page_lines,
    _sanitize_filename,
)

_CONTINUATION_RE = re.compile(r"Page\s+(\d+)\s+of\s+(\d+)", re.IGNORECASE)
_TITLE_MAX_CHARS = 60
_TOP_STRIP_FRACTION = 0.25
_BOTTOM_STRIP_FRACTION = 0.15
_TEXT_PAGE_MIN_CHARS = 50

# Patterns used by _extract_text_title to skip non-title ALL-CAPS lines.
_TEXT_TITLE_SKIP_RE = re.compile(
    r"FORM\s+NO\.|[:(#@©()]|\bLLC\b|\bINC\b|\bCORP\b|\bLTD\b|\bINCORPORATED\b",
    re.IGNORECASE,
)
# "A. " or "1. " style section headers within a document body.
_TEXT_TITLE_SECTION_RE = re.compile(r"^([A-Za-z]|\d+)\.\s")

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
    total_pages_in_doc: int | None = None


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


_MIN_TITLE_SCORE = 50.0


def _select_best_title(ocr_result: list | None) -> str | None:
    """Pick the best title candidate from OCR results.

    Scores each text by font height (box_height) with a multiplier that strongly
    favours multi-word strings and penalises single words (which tend to be logos,
    field labels, or column headers rather than document titles).  Strings that look
    like addresses, field labels, or long disclaimers are excluded entirely.

    Returns None when no candidate clears _MIN_TITLE_SCORE, which causes the caller
    to classify the page as AMBIGUOUS.  AMBIGUOUS pages attach to the previous
    document group rather than starting a new one — the intended behaviour for pages
    that have no discernible title (e.g. the back side of a multi-copy form).
    """
    if not ocr_result or not ocr_result[0]:
        return None

    best_text: str | None = None
    best_score: float = 0.0

    for line in ocr_result[0]:
        if not line or len(line) < 2 or not line[1]:
            continue
        text = line[1][0].strip()
        if not text or len(text) < 4 or len(text) > _TITLE_MAX_CHARS:
            continue
        # Field labels, form numbers with metadata, addresses
        if any(c in text for c in ":(#@"):
            continue
        # Strings with 7+ digits are addresses, phone numbers, or VINs — not titles.
        # Threshold is deliberately high so form numbers (ST-556, DEAL 321130, DT 523)
        # survive; phone numbers (847-882-8400 = 10d) and zip+4 strings do not.
        if sum(c.isdigit() for c in text) >= 7:
            continue
        words = text.split()
        n = len(words)
        # Long sentences are disclaimers, not titles
        if n > 8:
            continue
        # Garbled OCR produces runs of all-lowercase words (e.g. "se nag tor stot
        # codns").  Genuine titles are headed or all-caps; reject candidates with
        # 2+ all-lowercase words of length > 2.
        if sum(1 for w in words if w.islower() and len(w) > 2) >= 2:
            continue

        try:
            box = line[0]
            bh = max(pt[1] for pt in box) - min(pt[1] for pt in box)
        except (IndexError, TypeError):
            bh = 0

        # Multi-word text scores 2× higher; single words are heavily penalised
        # so that dealer logos and column headers don't beat real titles.
        multiplier = 2.0 if n >= 2 else 0.3
        score = bh * multiplier

        if score > best_score:
            best_score, best_text = score, text

    # If every surviving candidate is a weak column header or form-field label,
    # treat the page as untitled so it merges with the previous document.
    return best_text if best_score >= _MIN_TITLE_SCORE else None


_logger = logging.getLogger(__name__)


def _page_to_pil(pdf_page) -> Image.Image | None:
    images = pdf_page.images
    if not images:
        return None

    # Pick the largest image by pixel area — pages may embed small logos or stamps
    # before the main scan, so images[0] is not necessarily the page scan.
    best: Image.Image | None = None
    best_area = 0
    for img_obj in images:
        try:
            pil = img_obj.image.convert("RGB") if img_obj.image is not None else Image.open(io.BytesIO(img_obj.data)).convert("RGB")
        except Exception as exc:
            _logger.debug("_page_to_pil: could not decode image: %s", exc)
            continue
        w, h = pil.size
        if w * h > best_area:
            best, best_area = pil, w * h

    return best


def _extract_text_title(lines: list[str], *, prefer_last: bool = False) -> str | None:
    """Return an ALL-CAPS document title from the first half of page lines.

    prefer_last=False (default): returns the *first* qualifying candidate — correct
    for standalone pages where the title is the leading ALL-CAPS line.

    prefer_last=True: returns the *last* qualifying candidate in the first half —
    needed for complex forms (e.g. Retail Installment Contracts) where pypdf reads
    a FEDERAL TRUTH-IN-LENDING DISCLOSURES box before the actual contract name.
    Use this when a "Page 1 of N" marker confirms we are on the first page of a
    multi-page form.
    """
    half = max(len(lines) // 2, 10)
    result: str | None = None
    for line in lines[:half]:
        stripped = line.rstrip("*").strip()
        if len(stripped) < 4 or len(stripped) > _TITLE_MAX_CHARS:
            continue
        if stripped.endswith("."):
            continue
        # Skip non-ASCII lines — bilingual forms duplicate titles in a second language
        # with accented/encoded characters that produce garbled filenames.
        if any(ord(c) > 127 for c in stripped):
            continue
        if _TEXT_TITLE_SKIP_RE.search(stripped):
            continue
        if _TEXT_TITLE_SECTION_RE.match(stripped):
            continue
        if _DOC_MARKER_RE.search(stripped):
            continue
        words = stripped.split()
        if not (3 <= len(words) <= 8):
            continue
        # Address lines start with a house number (all-digit first token).
        if words[0].isdigit():
            continue
        # Must have at least one word with ≥3 alphabetic characters (rules out "DT 5/23").
        if not any(sum(c.isalpha() for c in w) >= 3 for w in words):
            continue
        # Addresses and phone numbers have ≥7 digits.
        if sum(c.isdigit() for c in stripped) >= 7:
            continue
        if stripped.upper() == stripped and any(c.isalpha() for c in stripped):
            if prefer_last:
                result = stripped  # keep scanning — last match wins
            else:
                return _sanitize_filename(stripped)  # first match wins
    return _sanitize_filename(result) if result else None


def _analyze_text_page(text: str) -> PageSignal:
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
                title = _extract_text_title(lines, prefer_last=True)
                return PageSignal(
                    classification="NEW_DOC",
                    title_text=title,
                    page_num_in_doc=1,
                    total_pages_in_doc=total if total > 1 else None,
                )

    # No explicit page markers — a detectable title signals a new document;
    # no title means this page belongs to whatever came before.
    title = _extract_text_title(lines)
    if title:
        return PageSignal(classification="NEW_DOC", title_text=title, page_num_in_doc=None)
    return PageSignal(classification="AMBIGUOUS", title_text=None, page_num_in_doc=None)


def _analyze_image_page(pdf_page) -> PageSignal:
    """Analyze a scanned/image page using PaddleOCR."""
    pil_image = _page_to_pil(pdf_page)
    if pil_image is None:
        return PageSignal(classification="AMBIGUOUS", title_text=None, page_num_in_doc=None)

    ocr = _get_ocr()
    width, height = pil_image.size
    if width == 0 or height == 0:
        return PageSignal(classification="AMBIGUOUS", title_text=None, page_num_in_doc=None)

    bottom_strip = pil_image.crop((0, int(height * (1 - _BOTTOM_STRIP_FRACTION)), width, height))
    bottom_result = ocr.ocr(np.array(bottom_strip))
    bottom_texts = _extract_ocr_texts(bottom_result)

    page_one_total: int | None = None
    for text in bottom_texts:
        m = _CONTINUATION_RE.search(text)
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
            if page_num == 1 and total > 1:
                page_one_total = total

    top_strip = pil_image.crop((0, 0, width, int(height * _TOP_STRIP_FRACTION)))
    top_result = ocr.ocr(np.array(top_strip))
    best_title = _select_best_title(top_result)

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


def analyze_page(pdf_page) -> PageSignal:
    """Unified page analyzer: text extraction for digital pages, OCR for scanned pages.

    The boundary signals (Page N of M, DOCUMENT N OF N) are structural and apply
    regardless of whether the page content is digitally encoded or image-scanned.
    """
    text = (pdf_page.extract_text() or "").strip()
    if len(text) >= _TEXT_PAGE_MIN_CHARS:
        return _analyze_text_page(text)
    return _analyze_image_page(pdf_page)


def _group_image_pages(
    signals: list[tuple[int, PageSignal]],
) -> list[list[int]]:
    if not signals:
        return []

    groups: list[list[int]] = []
    current: list[int] = []
    current_total: int | None = None

    for abs_idx, signal in signals:
        if signal.classification == "CONTINUATION":
            if (current and
                    current_total is not None and
                    signal.total_pages_in_doc is not None and
                    signal.total_pages_in_doc != current_total):
                groups.append(current)
                current = [abs_idx]
            elif current:
                current.append(abs_idx)
            else:
                current = [abs_idx]
            current_total = signal.total_pages_in_doc
        elif signal.classification == "NEW_DOC":
            if current:
                groups.append(current)
            current = [abs_idx]
            current_total = signal.total_pages_in_doc
        else:  # AMBIGUOUS
            if current:
                current.append(abs_idx)
            else:
                current = [abs_idx]
            current_total = None

    if current:
        groups.append(current)

    return groups


def _sanitize_image_title(title: str) -> str:
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


def split_pdf(pdf_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Unified PDF splitter: analyzes every page for document boundary signals,
    groups them, and writes one PDF per logical document.

    Text pages (≥50 extracted chars) are analyzed via pattern matching on the
    extracted text.  Image/scanned pages fall back to PaddleOCR.  The grouping
    logic is identical regardless of page type.
    """
    path = Path(pdf_path)
    out_dir = Path(output_dir)

    if not path.exists():
        raise FileNotFoundError(f"Input PDF not found: {path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(path))

    signals_list: list[tuple[int, PageSignal]] = []
    signals_dict: dict[int, PageSignal] = {}

    for i in range(len(reader.pages)):
        signal = analyze_page(reader.pages[i])
        signals_list.append((i, signal))
        signals_dict[i] = signal

    groups = _group_image_pages(signals_list)

    written: list[Path] = []
    used_names: dict[str, int] = {}

    for group in groups:
        dest = _write_image_group(reader, group, signals_dict, out_dir, used_names)
        written.append(dest)

    return written
