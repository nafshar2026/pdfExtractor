"""Public package exports for the PDF extraction library."""

from .extractor import ExtractedDocument, ExtractionResult, extract_pdf, find_pdf_files

__all__ = [
    "ExtractedDocument",
    "ExtractionResult",
    "extract_pdf",
    "find_pdf_files",
]
