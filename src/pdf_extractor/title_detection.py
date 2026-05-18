"""Title detection heuristics for text and OCR pages."""
from __future__ import annotations

import re

from .extractor import _DOC_MARKER_RE, _sanitize_filename

# OCR title candidates longer than this are likely disclaimers or body text, not titles.
# Increased from 60 to 100 to accommodate multi-word contract titles like
# "MOTOR VEHICLE RETAIL INSTALLMENT SALES CONTRACT - SIMPLE FINANCE CHARGE" (69 chars).
_TITLE_MAX_CHARS = 100

# Fraction of page height scanned from the top for title detection on image pages.
_TOP_STRIP_FRACTION = 0.25

# Fraction of page height scanned from the bottom for "Page N of M" on image pages.
_BOTTOM_STRIP_FRACTION = 0.15

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


# Minimum score thresholds.  Single-word candidates (multiplier=0.3) are held to a
# high bar so that logos and column headers don't qualify.  Multi-word candidates
# (multiplier=2.0) have already passed strict content filters (length, digit count,
# lowercase runs, corporate suffixes, address patterns) so a lower threshold is safe
# and catches small-print form titles that render at modest box heights.
_MIN_TITLE_SCORE = 50.0        # single-word
_MIN_TITLE_SCORE_MULTI = 20.0  # two or more words


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
    best_n: int = 0

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
        # Corporate entity abbreviations are logos/letterheads, not document titles.
        if re.search(r'\b(LLC|INC|CORP|LTD|INCORPORATED)\b', text, re.IGNORECASE):
            continue
        # Address lines: house number as first token (entire token all-digits), or
        # first character of first token is a digit (e.g. "6DATE"), or trailing
        # street/entity suffix.  The entity-type last-word set covers strings like
        # "Limited Liability Company", "S Corporation", "General Partnership" — all
        # of which are entity descriptors or checkbox options, not document headings.
        # "Only" at the end marks option labels ("Card Only", "Cash Only").
        if words[0].isdigit() or words[0][0].isdigit():
            continue
        # Address lines where the house number appears anywhere — as a standalone token
        # ("ADDRESS 2948 GREENBRIAR") or merged by OCR ("Address2948 GREENBRIAR").
        # Any word containing a run of ≥2 consecutive digits signals an address or form code.
        if any(re.search(r'\d{2,}(?![%\w])', w) for w in words):
            continue
        if words[-1].upper().rstrip('.') in {
            'BLVD', 'BOULEVARD', 'AVE', 'AVENUE', 'STREET', 'ROAD',
            'LANE', 'HWY', 'HIGHWAY', 'PKWY', 'PARKWAY',
            'COMPANY', 'CORPORATION', 'PARTNERSHIP', 'ORGANIZATION',
            'ENTITY', 'ASSOCIATION', 'TRUST', 'FOUNDATION',
            'ONLY',
        }:
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
        # codns") AND starts with a lowercase first word.  Title Case titles
        # (e.g. "Application for Texas Title and/or Registration") start with an
        # uppercase word and legitimately contain lowercase function words — only
        # apply the garbled-text filter when the first word is itself lowercase.
        if words[0][0].islower() and sum(1 for w in words if w.islower() and len(w) > 2) >= 2:
            continue
        # OCR sometimes concatenates adjacent form-field labels into one token with
        # embedded CamelCase (e.g. "DescriptionAdd", "LienOther").  Real words in
        # document titles don't carry an uppercase letter past the first character
        # unless the entire word is uppercase (ALL-CAPS).
        if any(len(w) > 4 and not w.isupper() and any(c.isupper() for c in w[1:]) for w in words):
            continue
        # Document titles always have at least one word starting with an upper-case
        # letter.  Phrases like "a check" that are entirely uncapitalized are field
        # values or checkbox labels, not document titles.
        if not any(w and w[0].isupper() for w in words):
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
            best_score, best_text, best_n = score, text, n

    # Multi-word candidates cleared strong content filters; they qualify at a lower
    # score threshold than single-word candidates (logos/column headers).
    threshold = _MIN_TITLE_SCORE_MULTI if best_n >= 2 else _MIN_TITLE_SCORE
    # If every surviving candidate is a weak column header or form-field label,
    # treat the page as untitled so it merges with the previous document.
    return _normalize_detected_title(best_text) if best_score >= threshold and best_text else None


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
    if len(candidate_key) >= 10 and candidate_key in footer_key and len(footer_key) <= 2 * len(candidate_key):
        return True
    return False


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
        # Address lines where the house number appears anywhere — standalone ("ADDRESS 2948
        # GREENBRIAR") or merged by OCR ("Address2948 GREENBRIAR"). Any word containing a
        # run of ≥2 consecutive digits signals an address or form code.
        if any(re.search(r'\d{2,}(?![%\w])', w) for w in words):
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
            # "Title: Subtitle" pattern — try the pre-colon portion as the title.
            if ':' in stripped and not re.search(r'[#@©()]', stripped):
                pre = stripped[:stripped.index(':')].strip()
                pre_words = pre.split()
                if (2 <= len(pre_words) <= 4
                        and _TITLE_DOCWORD_RE.search(pre)
                        and not appears_in_footer(pre)
                        and all(w[0].isupper() for w in pre_words if w and w[0].isalpha())):
                    return _sanitize_filename(_normalize_detected_title(pre))
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
