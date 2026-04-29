"""Core PDF reading utilities and data models used by the CLI and tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import re

from pypdf import PdfReader, PdfWriter


@dataclass(slots=True)
class ExtractionResult:
    """Text content extracted from a single PDF page."""

    page_number: int
    text: str


@dataclass(slots=True)
class ExtractedDocument:
    """All pages extracted from one PDF file, plus file-level metadata."""

    source_path: str
    page_count: int
    metadata: dict[str, Any]
    pages: list[ExtractionResult]

    @property
    def combined_text(self) -> str:
        """Concatenate all non-empty page texts with double newlines between them."""
        return "\n\n".join(page.text for page in self.pages if page.text)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the document to a plain dict suitable for JSON output."""
        payload = asdict(self)
        payload["combined_text"] = self.combined_text
        return payload


# Matches "DOCUMENT 1 OF 5", "DOCUMENT 3 OF 3", etc. — explicit boundary markers
# embedded by document assembly systems in the combined PDF.
_DOC_MARKER_RE = re.compile(r"\bDOCUMENT\s+\d+\s+OF\s+\d+\b", re.IGNORECASE)

# Matches bare page-number lines such as "Page 3" that are not real content.
_PAGE_MARKER_RE = re.compile(r"^Page\s+\d+$", re.IGNORECASE)

# Matches "Page 1 of N" patterns used as a fallback document-start signal.
_PAGE_ONE_OF_RE = re.compile(r"\bPage\s*1\s*of\s*\d+\b", re.IGNORECASE)

# Characters illegal in Windows / POSIX filenames.
_INVALID_FILENAME_RE = re.compile(r'[<>:"/\\|?*]+')

# Normalises runs of whitespace to a single space.
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_metadata(raw_metadata: Any) -> dict[str, Any]:
    """Convert pypdf's raw metadata object to a plain {str: str} dict.

    pypdf metadata keys carry a leading "/" (PDF name syntax).  This strips
    that prefix and converts all values to strings so the result serialises
    cleanly to JSON.

    Args:
        raw_metadata: The object returned by PdfReader.metadata, or None.

    Returns:
        A flat dict of string keys to string values.  Empty dict if input is None.
    """
    if raw_metadata is None:
        return {}

    normalized: dict[str, Any] = {}
    for key, value in dict(raw_metadata).items():
        clean_key = str(key).lstrip("/")
        normalized[clean_key] = None if value is None else str(value)
    return normalized


def _sanitize_filename(value: str) -> str:
    """Strip characters that are illegal in filenames and collapse whitespace.

    Args:
        value: Raw string (e.g. a document title extracted from page text).

    Returns:
        A safe filename stem.  Falls back to "Untitled" for empty input.
    """
    cleaned = _INVALID_FILENAME_RE.sub("", value).strip()
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return cleaned or "Untitled"


def _is_meaningful_line(line: str) -> bool:
    """Return True if a line carries real document content rather than boilerplate.

    Filters out:
    - Empty lines
    - Bare page-number lines ("Page 3")
    - DOCUMENT N OF N boundary markers
    - "Non-authoritative copy" watermark lines
    """
    if not line:
        return False
    if _PAGE_MARKER_RE.match(line):
        return False
    if _DOC_MARKER_RE.search(line):
        return False
    if line.lower().startswith("non-authoritative copy"):
        return False
    return True


def _extract_title(lines: list[str], marker_index: int, fallback: str) -> str:
    """Extract the best title candidate from a list of page text lines.

    Starts searching immediately after the DOCUMENT marker line (marker_index),
    skipping non-meaningful lines.  If nothing useful is found after the marker
    the function performs a second pass over all lines.  ALL-CAPS candidates are
    converted to Title Case for readability.

    Args:
        lines:        All non-empty lines from the page text.
        marker_index: Index of the DOCUMENT N OF N line (-1 to search from the top).
        fallback:     Value returned when no meaningful line is found at all.

    Returns:
        A sanitized filename-safe title string.
    """
    for i in range(marker_index + 1, len(lines)):
        candidate = lines[i].strip()
        if not _is_meaningful_line(candidate):
            continue
        if candidate.isupper():
            candidate = candidate.title()
        return _sanitize_filename(candidate)

    for line in lines:
        candidate = line.strip()
        if _is_meaningful_line(candidate):
            if candidate.isupper():
                candidate = candidate.title()
            return _sanitize_filename(candidate)

    return fallback


def _page_lines(text: str) -> list[str]:
    """Split page text into non-empty, stripped lines.

    Args:
        text: Raw text string extracted from a PDF page.

    Returns:
        List of non-empty strings with leading/trailing whitespace removed.
    """
    return [line.strip() for line in text.splitlines() if line.strip()]


def _detect_starts(page_texts: list[str]) -> list[int]:
    """Identify which page indices begin a new logical document.

    Detection strategy (in priority order):

    1. If any page contains a "DOCUMENT N OF N" marker, only those pages are
       boundaries.  This is the most reliable signal.

    2. Otherwise, pages containing "Page 1 of N" are treated as starts.

    The returned list always includes index 0 so that the first page is covered
    even when no markers are found.

    Args:
        page_texts: Extracted text for every page in the PDF (index 0 = page 1).

    Returns:
        Sorted list of page indices that begin a new document.
    """
    starts: list[int] = []

    for idx, text in enumerate(page_texts):
        if not text:
            continue
        if _DOC_MARKER_RE.search(text):
            starts.append(idx)

    if starts:
        return sorted(set(starts))

    for idx, text in enumerate(page_texts):
        if not text:
            continue
        if _PAGE_ONE_OF_RE.search(text):
            starts.append(idx)

    starts = sorted(set(starts))
    if 0 not in starts:
        starts.insert(0, 0)
    return starts


def extract_pdf(pdf_path: str | Path) -> ExtractedDocument:
    """Extract all page texts and file-level metadata from a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        An ExtractedDocument containing per-page text and metadata.
    """
    path = Path(pdf_path)
    reader = PdfReader(str(path))

    pages: list[ExtractionResult] = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        pages.append(ExtractionResult(page_number=index, text=text))

    return ExtractedDocument(
        source_path=str(path.resolve()),
        page_count=len(reader.pages),
        metadata=_normalize_metadata(reader.metadata),
        pages=pages,
    )


def split_pdf_by_internal_documents(pdf_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Split a text-based PDF into one file per logical document using embedded markers.

    This is the simpler text-only predecessor to image_splitter.split_pdf().
    Prefer split_pdf() for production use as it also handles scanned/image pages.

    Boundary detection uses _detect_starts():
    - "DOCUMENT N OF N" markers take priority.
    - "Page 1 of N" patterns are used as a fallback.

    Args:
        pdf_path:   Path to the source PDF.
        output_dir: Directory where split PDFs will be written (created if absent).

    Returns:
        List of Paths to the written output files (in document order).

    Raises:
        FileNotFoundError: If pdf_path does not exist.
        ValueError:        If no split markers are found or all segments are blank.
    """
    path = Path(pdf_path)
    out_dir = Path(output_dir)

    if not path.exists():
        raise FileNotFoundError(f"Input PDF not found: {path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(path))
    total_pages = len(reader.pages)

    page_texts: list[str] = []
    for page in reader.pages:
        page_texts.append((page.extract_text() or "").strip())

    starts = _detect_starts(page_texts)
    if not starts:
        raise ValueError("No split markers found.")

    starts = sorted(set(i for i in starts if 0 <= i < total_pages))
    if not starts:
        starts = [0]

    written: list[Path] = []
    used_names: dict[str, int] = {}

    for i, start_page in enumerate(starts):
        end_page = starts[i + 1] if i + 1 < len(starts) else total_pages
        if end_page <= start_page:
            continue

        if all(not page_texts[p].strip() for p in range(start_page, end_page)):
            continue

        first_text = page_texts[start_page]
        lines = _page_lines(first_text)
        title = _extract_title(lines, -1, f"Document {len(written) + 1}")
        base_name = _sanitize_filename(title)

        used_names[base_name] = used_names.get(base_name, 0) + 1
        suffix = "" if used_names[base_name] == 1 else f" ({used_names[base_name]})"
        destination = out_dir / f"{base_name}{suffix}.pdf"

        writer = PdfWriter()
        for page_num in range(start_page, end_page):
            writer.add_page(reader.pages[page_num])

        with destination.open("wb") as handle:
            writer.write(handle)

        written.append(destination)

    if not written:
        raise ValueError("No non-empty split documents were produced.")

    return written


def find_pdf_files(input_path: str | Path, recursive: bool = False) -> list[Path]:
    """Locate PDF files under the given path.

    If input_path is a file, returns it directly (raising ValueError if it is
    not a PDF).  If it is a directory, globs for *.pdf (or **/*.pdf when
    recursive=True).

    Args:
        input_path: A PDF file or a directory to search.
        recursive:  Whether to descend into subdirectories.

    Returns:
        Sorted list of PDF Paths found.

    Raises:
        ValueError:        If input_path is a non-PDF file.
        FileNotFoundError: If input_path does not exist.
    """
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path}")
        return [path]

    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(file_path for file_path in path.glob(pattern) if file_path.is_file())
