"""Unified PDF splitting: text extraction for digital pages, PaddleOCR for scanned pages."""

from __future__ import annotations

import gc
import io
import logging
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
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

# Matches "Page 3 of 6" in any page type — used to identify continuations and first pages.
# OCR commonly collapses spaces in pagination text (e.g. "PAGE5OF6", "Page 1 of6").
_CONTINUATION_RE = re.compile(r"Page\s*(\d+)\s*of\s*(\d+)", re.IGNORECASE)

# OCR title candidates longer than this are likely disclaimers or body text, not titles.
# Increased from 60 to 100 to accommodate multi-word contract titles like
# "MOTOR VEHICLE RETAIL INSTALLMENT SALES CONTRACT - SIMPLE FINANCE CHARGE" (69 chars).
_TITLE_MAX_CHARS = 100

# Fraction of page height scanned from the top for title detection on image pages.
_TOP_STRIP_FRACTION = 0.25

# Fraction of page height scanned from the bottom for "Page N of M" on image pages.
_BOTTOM_STRIP_FRACTION = 0.15

# Maximum width (pixels) for images passed to PaddleOCR.  High-resolution scans
# (600 DPI letter page = 5100 px wide) are downscaled to this before OCR so that
# PaddlePaddle's inference buffers stay within the container memory limit.
# Default 1500 px gives ~150 DPI equivalent — more than sufficient for text
# recognition on standard forms.
_OCR_MAX_WIDTH_DEFAULT = 1500

# A page with at least this many extracted characters is treated as a digital text page;
# fewer characters means the page is blank or image-only and falls back to OCR.
_TEXT_PAGE_MIN_CHARS = 50

# For digital text pages, only consider lines from the upper portion of the page
# when using layout-aware title extraction to avoid mid-page boxed labels.
_TEXT_TITLE_TOP_FRACTION = 0.30

# Patterns used by _extract_text_title to skip non-title ALL-CAPS lines.
_TEXT_TITLE_SKIP_RE = re.compile(
    r"FORM\s+NO\.|[:(#@©()]|\bLLC\b|\bINC\b|\bCORP\b|\bLTD\b|\bINCORPORATED\b",
    re.IGNORECASE,
)
# "A. " or "1. " style section headers within a document body.
_TEXT_TITLE_SECTION_RE = re.compile(r"^([A-Za-z]|\d+)\.\s")
_TITLE_DOCWORD_RE = re.compile(
    r"\b(CONTRACT|APPLICATION|DISCLOSURE|STATEMENT|AGREEMENT|NOTICE|AUTHORIZATION|ORDER|LEASE|ADDENDUM|INVOICE|ODOMETER|WARRANTY|INSURANCE|FINANCE|RETAIL)\b",
    re.IGNORECASE,
)
_TITLE_STRONG_DOCWORD_RE = re.compile(
    r"\b(CONTRACT|APPLICATION|AGREEMENT|ORDER|LEASE|ADDENDUM|INVOICE|ODOMETER)\b",
    re.IGNORECASE,
)
_TITLE_WEAK_HEADER_RE = re.compile(
    r"\b(DISCLOSURE|DISCLOSURES|DISCLAIMER|WARRANTY|WARRANTIES|NOTICE)\b",
    re.IGNORECASE,
)

# DealerTrack credit application detection.
# DT prints a version footer ("DT 6/17", "DT 5/23") on every page of their forms;
# the credit app is the only one that also carries the incomplete-applications notice.
# The DT pattern is intentionally loose ("DT" + any digit) to survive OCR misreads.
_DT_FOOTER_RE = re.compile(r"\bDT\s*\d", re.IGNORECASE)
_DT_CREDIT_APP_RE = re.compile(r"INCOMPLETE\s+APPLICATIONS\s+WILL\s+NOT\s+BE\s+PROCESSED", re.IGNORECASE)


def _infer_content_title(text: str) -> str | None:
    """Return a canonical title when structural heuristics cannot detect one.

    Currently handles one known case: DealerTrack credit application forms,
    which carry no ALL-CAPS title but do carry a unique DT version footer and
    an 'INCOMPLETE APPLICATIONS WILL NOT BE PROCESSED' instruction line.
    Both signals must be present to avoid false positives on other DT forms.

    Works on both digitally-extracted text and OCR-joined text from image pages.
    """
    if _DT_FOOTER_RE.search(text) and _DT_CREDIT_APP_RE.search(text):
        return "Credit Application"
    return None

# Module-level singleton; avoids reloading the ~200 MB PaddleOCR model on every page.
_ocr_instance: PaddleOCR | None = None
_ocr_pool: ProcessPoolExecutor | None = None
_ocr_pool_calls = 0


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


# When enabled, OCR runs in a dedicated subprocess so memory can be reclaimed by
# recycling the worker process instead of accumulating in the main process.
_OCR_ISOLATED = _env_flag("PDF_EXTRACTOR_OCR_ISOLATED", default=False)

# Number of OCR calls before recycling the isolated worker. 0 disables recycle.
_OCR_RECYCLE_CALLS = max(0, _env_int("PDF_EXTRACTOR_OCR_RECYCLE_CALLS", 40))

# Retry count when the isolated worker dies mid-inference.
_OCR_POOL_RETRIES = max(0, _env_int("PDF_EXTRACTOR_OCR_POOL_RETRIES", 1))

# Optional global cap for rendered OCR width to trade speed/accuracy for memory.
_OCR_MAX_WIDTH = max(400, _env_int("PDF_EXTRACTOR_OCR_MAX_WIDTH", _OCR_MAX_WIDTH_DEFAULT))


def _get_ocr() -> PaddleOCR:
    """Return the process-wide PaddleOCR singleton, initialising it on first call.

    Angle classification is disabled because deal-jacket pages are always upright.
    Initialisation takes several seconds and loads ~200 MB of model weights, so the
    instance is created once and reused for every subsequent page in the same run.
    """
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
    return _ocr_instance


_worker_ocr_instance = None


def _worker_run_ocr(image_array: np.ndarray):
    """Run OCR in worker process with a process-local PaddleOCR singleton."""
    global _worker_ocr_instance
    if _worker_ocr_instance is None:
        _worker_ocr_instance = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
    return _worker_ocr_instance.ocr(image_array)


def _get_ocr_pool() -> ProcessPoolExecutor:
    global _ocr_pool
    if _ocr_pool is None:
        # Use spawn so worker memory is isolated from parent state across platforms.
        mp_ctx = multiprocessing.get_context("spawn")
        _ocr_pool = ProcessPoolExecutor(max_workers=1, mp_context=mp_ctx)
    return _ocr_pool


def _shutdown_ocr_pool() -> None:
    global _ocr_pool, _ocr_pool_calls
    if _ocr_pool is not None:
        _ocr_pool.shutdown(wait=True, cancel_futures=False)
        _ocr_pool = None
    _ocr_pool_calls = 0


def _ocr_infer(image_array: np.ndarray):
    """Run OCR either in-process or in an isolated subprocess worker."""
    global _ocr_pool_calls

    if not _OCR_ISOLATED:
        return _get_ocr().ocr(image_array)

    # Ensure stable pickling/transfer to worker process.
    arr = np.ascontiguousarray(image_array)

    attempts = _OCR_POOL_RETRIES + 1
    for attempt in range(attempts):
        try:
            pool = _get_ocr_pool()
            result = pool.submit(_worker_run_ocr, arr).result()
            _ocr_pool_calls += 1

            if _OCR_RECYCLE_CALLS > 0 and _ocr_pool_calls >= _OCR_RECYCLE_CALLS:
                _shutdown_ocr_pool()

            return result
        except BrokenProcessPool:
            # Worker died (commonly OOM). Recreate pool and retry a bounded number of times.
            _shutdown_ocr_pool()
            if attempt >= attempts - 1:
                raise


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


def _normalize_detected_title(text: str) -> str:
    """Normalize detected titles to remove common leading id noise.

    Example: ``H18554BUSINESS OR COMMERCIAL`` -> ``BUSINESS OR COMMERCIAL``.
    """
    normalized = re.sub(r"(?<=\d)(?=[A-Z])", " ", text, count=1).strip()
    words = normalized.split()
    # Drop only short ID-like prefixes (e.g. "H18554") and keep meaningful
    # form-code tokens like "LAW553" so footer suppression can match them.
    if len(words) >= 2:
        first = words[0]
        digits = sum(c.isdigit() for c in first)
        letters = sum(c.isalpha() for c in first)
        if digits >= 3 and letters <= 2:
            normalized = " ".join(words[1:])
    return normalized


def _filter_layout_noise_for_title(ocr_result: list | None) -> list | None:
    """Filter OCR lines that are likely boxed/table field labels.

    This is fully dynamic and page-local: no fixed coordinates are used.
    """
    if not ocr_result or not ocr_result[0]:
        return ocr_result

    page_lines = ocr_result[0]
    parsed: list[dict] = []
    for idx, line in enumerate(page_lines):
        if not line or len(line) < 2 or not line[1]:
            continue
        try:
            box = line[0]
            x0 = float(min(pt[0] for pt in box))
            y0 = float(min(pt[1] for pt in box))
            x1 = float(max(pt[0] for pt in box))
            y1 = float(max(pt[1] for pt in box))
        except (TypeError, ValueError, IndexError):
            continue
        text = str(line[1][0]).strip()
        parsed.append({
            "idx": idx,
            "line": line,
            "text": text,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "w": max(1.0, x1 - x0),
            "h": max(1.0, y1 - y0),
            "cy": (y0 + y1) / 2.0,
        })

    if not parsed:
        return ocr_result

    page_width = max(item["x1"] for item in parsed)
    filtered_lines: list = []

    for item in parsed:
        words = item["text"].split()
        short_words = sum(1 for w in words if len(w) <= 3)
        width_ratio = item["w"] / max(1.0, page_width)

        row_density = 0
        for other in parsed:
            if abs(other["cy"] - item["cy"]) <= max(item["h"], other["h"]) * 0.8:
                row_density += 1

        # Dense rows of short labels are usually boxed form fields, not document titles.
        is_label_row = (
            row_density >= 4
            and len(words) <= 4
            and short_words >= max(1, len(words) - 1)
            and width_ratio < 0.45
        )
        if not is_label_row:
            filtered_lines.append(item["line"])

    return [filtered_lines]


# Minimum score (box_height × word-count multiplier) a candidate must reach to be
# accepted as a document title.  Candidates below this threshold are treated as
# column headers or field labels, and the page is classified as AMBIGUOUS so it
# merges with the previous document group instead of starting a new one.
_MIN_TITLE_SCORE = 50.0


def _looks_like_form_code_title(text: str) -> bool:
    """Heuristic for short alphanumeric form-code strings (not real titles)."""
    if not text:
        return False
    has_alpha = any(c.isalpha() for c in text)
    has_digit = any(c.isdigit() for c in text)
    has_sep = any(c in text for c in "-/")
    return has_alpha and has_digit and has_sep and len(text.split()) <= 8


def _title_appears_top_and_footer(ocr_result: list | None, candidate: str) -> bool:
    """Return True when the same normalized text appears near top and footer."""
    if not ocr_result or not ocr_result[0] or not candidate:
        return False

    key = _title_key(_normalize_detected_title(candidate))
    if not key:
        return False

    page_height_est = 1.0
    for line in ocr_result[0]:
        if not line or len(line) < 2:
            continue
        box = line[0]
        try:
            page_height_est = max(page_height_est, float(max(pt[1] for pt in box)))
        except (TypeError, ValueError, IndexError):
            continue

    seen_top = False
    seen_bottom = False
    for line in ocr_result[0]:
        if not line or len(line) < 2 or not line[1]:
            continue
        try:
            text = _normalize_detected_title(line[1][0].strip())
            box = line[0]
            cy = (max(pt[1] for pt in box) + min(pt[1] for pt in box)) / 2.0
        except (TypeError, ValueError, IndexError):
            continue

        text_key = _title_key(text)
        if not (_is_footer_variant(key, text_key) or _is_footer_variant(text_key, key)):
            continue
        if cy <= page_height_est * 0.35:
            seen_top = True
        if cy >= page_height_est * 0.70:
            seen_bottom = True
        if seen_top and seen_bottom:
            return True

    return False


def _select_best_title(ocr_result: list | None, *, footer_texts: list[str] | None = None) -> str | None:
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

    footer_keys = {
        _title_key(_normalize_detected_title(t.strip()))
        for t in (footer_texts or [])
        if t and _title_key(_normalize_detected_title(t.strip()))
    }

    # Remove layout-driven field-label noise first (dynamic per page).
    ocr_result = _filter_layout_noise_for_title(ocr_result)
    if not ocr_result or not ocr_result[0]:
        return None

    page_height_est = 1.0
    page_width_est = 1.0
    for line in ocr_result[0]:
        if not line or len(line) < 2:
            continue
        box = line[0]
        try:
            page_height_est = max(page_height_est, float(max(pt[1] for pt in box)))
            page_width_est = max(page_width_est, float(max(pt[0] for pt in box)))
        except (TypeError, ValueError, IndexError):
            continue

    best_text: str | None = None
    best_score: float = 0.0

    for line in ocr_result[0]:
        if not line or len(line) < 2 or not line[1]:
            continue
        text = _normalize_detected_title(line[1][0].strip())
        if not text or len(text) < 4 or len(text) > _TITLE_MAX_CHARS:
            continue
        words = text.split()
        n = len(words)
        text_key = _title_key(text)
        if any(_is_footer_variant(text_key, footer_key) for footer_key in footer_keys):
            continue
        # Field labels, form numbers with metadata, addresses
        if any(c in text for c in ":(#@"):
            continue
        # Strings with 7+ digits are addresses, phone numbers, or VINs — not titles.
        # Threshold is deliberately high so form numbers (ST-556, DEAL 321130, DT 523)
        # survive; phone numbers (847-882-8400 = 10d) and zip+4 strings do not.
        if sum(c.isdigit() for c in text) >= 7:
            continue
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
            bw = max(pt[0] for pt in box) - min(pt[0] for pt in box)
            cy = (max(pt[1] for pt in box) + min(pt[1] for pt in box)) / 2.0
        except (IndexError, TypeError):
            bh = 0
            bw = 0
            cy = page_height_est

        # Multi-word text scores 2× higher; single words are heavily penalised
        # so that dealer logos and column headers don't beat real titles.
        multiplier = 2.0 if n >= 2 else 0.3
        width_bonus = 0.7 + 0.6 * (bw / max(1.0, page_width_est))
        top_bonus = 0.7 + 0.6 * max(0.0, 1.0 - (cy / max(1.0, page_height_est)))
        score = bh * multiplier * width_bonus * top_bonus

        if score > best_score:
            best_score, best_text = score, text

    # If every surviving candidate is a weak column header or form-field label,
    # treat the page as untitled so it merges with the previous document.
    return _normalize_detected_title(best_text) if best_score >= _MIN_TITLE_SCORE and best_text else None


_logger = logging.getLogger(__name__)


def _title_key(title: str) -> str:
    """Normalise a title for same-document comparison.

    Strips everything except letters and digits, then uppercases the result so
    that OCR variations in spacing, punctuation, and capitalisation all hash to
    the same key.  Examples that must compare equal:
      "VEHICLE BUYERS ORDER"  →  "VEHICLEBUYERSORDER"
      "VEHICLEBUYERS ORDER"   →  "VEHICLEBUYERSORDER"  (missing space)
      "VEhICLE BUYERS ORDER"  →  "VEHICLEBUYERSORDER"  (mixed-case OCR)
    """
    return re.sub(r"[^A-Z0-9]", "", title.upper())


def _is_footer_variant(candidate_key: str, footer_key: str) -> bool:
    """True when footer_key is candidate_key with extra trailing/leading noise.

    Example: LAW553TXARBEA421 matches LAW553TXARBEA421V1PAGE1OF6.
    """
    if not candidate_key or not footer_key:
        return False
    if candidate_key == footer_key:
        return True
    if len(candidate_key) >= 10 and candidate_key in footer_key:
        return True
    return False


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


def _extract_text_title(
    lines: list[str],
    *,
    prefer_last: bool = False,
    max_words: int = 8,
    require_doc_word: bool = False,
) -> str | None:
    """Return a document title from the first half of page lines.

    Two passes are made in priority order:

    **Pass 1 — ALL-CAPS** (primary, searches first half of page):
    The strongest signal.  Most printed forms put their title in full capitals.
    prefer_last=False (default): returns the *first* qualifying candidate.
    prefer_last=True: returns the *last* candidate in the first half — needed for
    complex forms (e.g. Retail Installment Contracts) where pypdf reads a
    FEDERAL TRUTH-IN-LENDING DISCLOSURES box before the actual contract name.

    **Pass 2 — Title Case fallback** (searches only the first 6 meaningful lines):
    Catches forms whose title is mixed-case (e.g. "Credit Application").  Restricted
    to the opening lines because real titles appear near the top of the page;
    field labels and body text that happen to be Title Case appear much later.
    Word count is capped at 4 (tighter than ALL-CAPS) to exclude long field-header
    strings like "Title Last Name First Middle Suffix" (6 words).
    """
    half = max(len(lines) // 2, 10)

    # If a candidate string also appears in the footer region, it is likely a
    # form code/footer label (not a document title).
    footer_window = max(8, len(lines) // 4)
    footer_start = max(0, len(lines) - footer_window)
    footer_keys = {
        _title_key(line.rstrip("*").strip())
        for line in lines[footer_start:]
        if line and _title_key(line.rstrip("*").strip())
    }

    def appears_in_footer(candidate: str) -> bool:
        candidate_key = _title_key(candidate)
        return any(_is_footer_variant(candidate_key, footer_key) for footer_key in footer_keys)

    result: str | None = None
    best_prefer_last: str | None = None
    best_prefer_last_score = float("-inf")

    # Unicode box/checkbox characters to disqualify as titles
    BOX_CHARS = set([
        "☐", "☑", "☒", "□", "■", "▪", "▢", "▣", "▤", "▥", "▦", "▧", "▨", "▩", "⬜", "⬛", "🗹", "🗷", "🗸", "🗵", "🗶"
    ])
    def has_box_char(s):
        return any(c in BOX_CHARS for c in s)

    max_chars = 100 if prefer_last else _TITLE_MAX_CHARS

    for idx, line in enumerate(lines[:half]):
        stripped = line.rstrip("*").strip()
        # Exclude if this line or an adjacent line contains a box/checkbox character
        if has_box_char(stripped):
            continue
        if idx > 0 and has_box_char(lines[idx-1]):
            continue
        if idx+1 < len(lines) and has_box_char(lines[idx+1]):
            continue
        if len(stripped) < 4 or len(stripped) > max_chars:
            continue
        if stripped.endswith("."):
            continue
        # Allow Unicode punctuation (e.g., en dash) but reject non-ASCII letters,
        # which are often bilingual duplicates and produce noisy filenames.
        if any(ord(c) > 127 and c.isalpha() for c in stripped):
            continue
        if _TEXT_TITLE_SKIP_RE.search(stripped):
            continue
        if _TEXT_TITLE_SECTION_RE.match(stripped):
            continue
        if _DOC_MARKER_RE.search(stripped):
            continue
        words = stripped.split()
        if not (3 <= len(words) <= max_words):
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
        if require_doc_word and not _TITLE_DOCWORD_RE.search(stripped):
            continue
        if appears_in_footer(stripped):
            continue
        if stripped.upper() == stripped and any(c.isalpha() for c in stripped):
            if prefer_last:
                score = 0.0
                if _TITLE_STRONG_DOCWORD_RE.search(stripped):
                    score += 6.0
                elif _TITLE_DOCWORD_RE.search(stripped):
                    score += 2.0
                if _TITLE_WEAK_HEADER_RE.search(stripped):
                    score -= 2.5
                if re.search(r"\bOR\b", stripped, re.IGNORECASE):
                    score -= 1.0
                # Mild bias toward later lines when tie-breaking.
                score += idx / max(1, half)
                if score > best_prefer_last_score:
                    best_prefer_last_score = score
                    best_prefer_last = stripped
            else:
                return _sanitize_filename(_normalize_detected_title(stripped))  # first match wins

    if prefer_last and best_prefer_last:
        return _sanitize_filename(_normalize_detected_title(best_prefer_last))
    if result:
        return _sanitize_filename(_normalize_detected_title(result))

    # Pass 2: Title Case fallback — only the first 6 lines, max 4 words.
    # ALL-CAPS strings are excluded: they already failed Pass 1 for a good reason
    # (e.g. "APPLICABLE LAW" — a section label, not a document title).
    for line in lines[:min(half, 6)]:
        stripped = line.rstrip("*").strip()
        if len(stripped) < 4 or len(stripped) > _TITLE_MAX_CHARS:
            continue
        if stripped.endswith("."):
            continue
        if any(ord(c) > 127 and c.isalpha() for c in stripped):
            continue
        if _TEXT_TITLE_SKIP_RE.search(stripped):
            continue
        if _TEXT_TITLE_SECTION_RE.match(stripped):
            continue
        if _DOC_MARKER_RE.search(stripped):
            continue
        words = stripped.split()
        if not (2 <= len(words) <= 4):
            continue
        if words[0].isdigit():
            continue
        if not any(sum(c.isalpha() for c in w) >= 3 for w in words):
            continue
        if sum(c.isdigit() for c in stripped) >= 7:
            continue
        if appears_in_footer(stripped):
            continue
        # Skip ALL-CAPS — those already had their chance in Pass 1.
        if stripped.upper() == stripped:
            continue
        # Lines containing an ALL-CAPS word with ≥3 alpha chars are column headers or
        # field-label rows (e.g. "Make Trim VIN"), not document titles.
        if any(
            all(c.isupper() for c in w if c.isalpha()) and sum(c.isalpha() for c in w) >= 3
            for w in words
        ):
            continue
        # Every alphabetic-starting word must begin with an uppercase letter.
        if all(w[0].isupper() for w in words if w and w[0].isalpha()):
            return _sanitize_filename(_normalize_detected_title(stripped))

    return None


def _extract_text_title_with_layout(fitz_doc, page_idx: int, *, prefer_last: bool = False) -> str | None:
    """Extract title from a digital text page using layout y-position filtering.

    Uses PyMuPDF text lines and keeps only lines whose top is in the upper page
    band. This avoids selecting all-caps labels found in boxed sections located
    mid-page or lower.
    """
    if fitz_doc is None:
        return None

    try:
        page = fitz_doc[page_idx]
        page_height = float(page.rect.height)
        if page_height <= 0:
            return None
        top_limit = page_height * _TEXT_TITLE_TOP_FRACTION
        text_dict = page.get_text("dict")
    except Exception:
        return None

    layout_rows: list[tuple[float, float, str]] = []
    for block in text_dict.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            bbox = line.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            line_top = float(bbox[1])
            line_left = float(bbox[0])
            if line_top > top_limit:
                continue
            spans = line.get("spans", [])
            text = "".join(span.get("text", "") for span in spans).strip()
            if text:
                layout_rows.append((line_top, line_left, text))

    if not layout_rows:
        return None

    layout_rows.sort(key=lambda row: (row[0], row[1]))
    layout_lines = [row[2] for row in layout_rows]

    return _extract_text_title(
        layout_lines,
        prefer_last=prefer_last,
        max_words=12,
        require_doc_word=True,
    )


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


def split_pdf(pdf_path: str | Path, output_dir: str | Path, *, verbose: bool = False) -> list[Path]:
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
    """
    path = Path(pdf_path)
    out_dir = Path(output_dir)

    if not path.exists():
        raise FileNotFoundError(f"Input PDF not found: {path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(path))

    import fitz
    fitz_doc = fitz.open(str(path))

    signals_list: list[tuple[int, PageSignal]] = []
    signals_dict: dict[int, PageSignal] = {}

    try:
        for i in range(len(reader.pages)):
            signal = analyze_page(reader.pages[i], fitz_doc=fitz_doc, page_idx=i)
            signals_list.append((i, signal))
            signals_dict[i] = signal
            gc.collect()  # release OCR buffers from the processed page
    finally:
        fitz_doc.close()
        _shutdown_ocr_pool()

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
    for i, group in enumerate(groups):
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
    
    if verbose:
        print(f"\n--- DEDUPLICATION REPORT ---")
        print(f"Groups before dedup: {len(groups)}")
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

    return written
