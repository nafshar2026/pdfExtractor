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

# Matches "Page 3 of 6" in any page type — used to identify continuations and first pages.
_CONTINUATION_RE = re.compile(r"Page\s+(\d+)\s+of\s+(\d+)", re.IGNORECASE)

# OCR title candidates longer than this are likely disclaimers or body text, not titles.
_TITLE_MAX_CHARS = 60

# Fraction of page height scanned from the top for title detection on image pages.
_TOP_STRIP_FRACTION = 0.25

# Fraction of page height scanned from the bottom for "Page N of M" on image pages.
_BOTTOM_STRIP_FRACTION = 0.15

# A page with at least this many extracted characters is treated as a digital text page;
# fewer characters means the page is blank or image-only and falls back to OCR.
_TEXT_PAGE_MIN_CHARS = 50

# Patterns used by _extract_text_title to skip non-title ALL-CAPS lines.
_TEXT_TITLE_SKIP_RE = re.compile(
    r"FORM\s+NO\.|[:(#@©()]|\bLLC\b|\bINC\b|\bCORP\b|\bLTD\b|\bINCORPORATED\b",
    re.IGNORECASE,
)
# "A. " or "1. " style section headers within a document body.
_TEXT_TITLE_SECTION_RE = re.compile(r"^([A-Za-z]|\d+)\.\s")

# Module-level singleton; avoids reloading the ~200 MB PaddleOCR model on every page.
_ocr_instance: PaddleOCR | None = None


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


# Minimum score (box_height × word-count multiplier) a candidate must reach to be
# accepted as a document title.  Candidates below this threshold are treated as
# column headers or field labels, and the page is classified as AMBIGUOUS so it
# merges with the previous document group instead of starting a new one.
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
    """Extract the dominant embedded image from a PDF page as a PIL Image.

    Many scanned deal-jacket pages embed a single high-resolution scan plus small
    decorative images (logos, stamps, watermarks).  This function picks the image
    with the largest pixel area so that incidental graphics do not win over the
    main page scan.

    Args:
        pdf_page: A pypdf ``PageObject`` from ``PdfReader.pages``.

    Returns:
        The largest embedded image converted to RGB, or None if the page contains
        no decodable image data.
    """
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

    if result:
        return _sanitize_filename(result)

    # Pass 2: Title Case fallback — only the first 6 lines, max 4 words.
    # ALL-CAPS strings are excluded: they already failed Pass 1 for a good reason
    # (e.g. "APPLICABLE LAW" — a section label, not a document title).
    for line in lines[:min(half, 6)]:
        stripped = line.rstrip("*").strip()
        if len(stripped) < 4 or len(stripped) > _TITLE_MAX_CHARS:
            continue
        if stripped.endswith("."):
            continue
        if any(ord(c) > 127 for c in stripped):
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
            return _sanitize_filename(stripped)

    return None


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
    """Group per-page signals into lists of page indices, one list per logical document.

    Grouping rules (applied in order):

    - **NEW_DOC**: Close the current group (if any) and start a fresh one.
      Exception: if the current group consists entirely of AMBIGUOUS pages (no
      anchor signal has been seen yet), those pages are prepended to the new
      NEW_DOC group rather than saved as a separate file.  This handles forms
      where the title page comes second and the first page is a cover/back side
      with no detectable title.
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

    groups: list[list[int]] = []
    current: list[int] = []
    current_total: int | None = None
    current_has_anchor: bool = False  # True once group contains a NEW_DOC or CONTINUATION page

    for abs_idx, signal in signals:
        if signal.classification == "CONTINUATION":
            current_has_anchor = True
            if (current and
                    current_total is not None and
                    signal.total_pages_in_doc is not None and
                    signal.total_pages_in_doc != current_total):
                groups.append(current)
                current = [abs_idx]
                current_has_anchor = True
            elif current:
                current.append(abs_idx)
            else:
                current = [abs_idx]
            current_total = signal.total_pages_in_doc
        elif signal.classification == "NEW_DOC":
            if current and current_has_anchor:
                # Current group has real content — save it and start fresh.
                groups.append(current)
                current = [abs_idx]
            elif current and not current_has_anchor:
                # Current group is all-AMBIGUOUS — prepend it to this NEW_DOC group
                # rather than saving a separate untitled file.
                current.append(abs_idx)
            else:
                current = [abs_idx]
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
    # Scan all pages in the group for the first non-None title.  The titled page
    # is usually first, but leading AMBIGUOUS pages (prepended by _group_image_pages)
    # have no title, so the anchor title may be on a later page.
    raw_title = None
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
