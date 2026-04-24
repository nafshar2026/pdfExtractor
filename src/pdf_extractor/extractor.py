"""Core PDF reading utilities and data models used by the CLI and tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import re

from pypdf import PdfReader, PdfWriter


@dataclass(slots=True)
class ExtractionResult:
    page_number: int
    text: str


@dataclass(slots=True)
class ExtractedDocument:
    source_path: str
    page_count: int
    metadata: dict[str, Any]
    pages: list[ExtractionResult]

    @property
    def combined_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["combined_text"] = self.combined_text
        return payload


_DOC_MARKER_RE = re.compile(r"\bDOCUMENT\s+\d+\s+OF\s+\d+\b", re.IGNORECASE)
_PAGE_MARKER_RE = re.compile(r"^Page\s+\d+$", re.IGNORECASE)
_PAGE_ONE_OF_RE = re.compile(r"\bPage\s*1\s*of\s*\d+\b", re.IGNORECASE)
_INVALID_FILENAME_RE = re.compile(r'[<>:"/\\|?*]+')
_WHITESPACE_RE = re.compile(r"\s+")
_FORM_LIKE_START_RE = re.compile(
    r"^(ODOMETER DISCLOSURE STATEMENT|ASSIGNMENT OF CREDIT CONTRACT|DT\s+\d+/\d+|FORM NO\.)",
    re.IGNORECASE,
)


def _normalize_metadata(raw_metadata: Any) -> dict[str, Any]:
    if raw_metadata is None:
        return {}

    normalized: dict[str, Any] = {}
    for key, value in dict(raw_metadata).items():
        clean_key = str(key).lstrip("/")
        normalized[clean_key] = None if value is None else str(value)
    return normalized


def _sanitize_filename(value: str) -> str:
    cleaned = _INVALID_FILENAME_RE.sub("", value).strip()
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return cleaned or "Untitled"


def _is_meaningful_line(line: str) -> bool:
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
    return [line.strip() for line in text.splitlines() if line.strip()]


def _detect_starts(page_texts: list[str]) -> list[int]:
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

    # fallback: common form/document headers (except page 0 already included later)
    for idx, text in enumerate(page_texts):
        if idx == 0 or not text:
            continue
        lines = _page_lines(text)
        if not lines:
            continue
        first = lines[0]
        if _FORM_LIKE_START_RE.search(first):
            starts.append(idx)

    starts = sorted(set(starts))
    if 0 not in starts:
        starts.insert(0, 0)
    return starts


def extract_pdf(pdf_path: str | Path) -> ExtractedDocument:
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

    # Ensure sorted unique and bounded
    starts = sorted(set(i for i in starts if 0 <= i < total_pages))
    if not starts:
        starts = [0]

    written: list[Path] = []
    used_names: dict[str, int] = {}

    for i, start_page in enumerate(starts):
        end_page = starts[i + 1] if i + 1 < len(starts) else total_pages
        if end_page <= start_page:
            continue

        # Skip chunks that are fully blank
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
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path}")
        return [path]

    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(file_path for file_path in path.glob(pattern) if file_path.is_file())