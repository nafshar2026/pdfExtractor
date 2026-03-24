"""Core PDF reading utilities and data models used by the CLI and tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader


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


def _normalize_metadata(raw_metadata: Any) -> dict[str, Any]:
    if raw_metadata is None:
        return {}

    normalized: dict[str, Any] = {}
    for key, value in dict(raw_metadata).items():
        clean_key = str(key).lstrip("/")
        if value is None:
            normalized[clean_key] = None
        else:
            normalized[clean_key] = str(value)
    return normalized


def extract_pdf(pdf_path: str | Path) -> ExtractedDocument:
    path = Path(pdf_path)
    reader = PdfReader(str(path))

    pages: list[ExtractionResult] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(ExtractionResult(page_number=index, text=text.strip()))

    return ExtractedDocument(
        source_path=str(path.resolve()),
        page_count=len(reader.pages),
        metadata=_normalize_metadata(reader.metadata),
        pages=pages,
    )


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
