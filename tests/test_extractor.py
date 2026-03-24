"""Tests for PDF discovery, extraction behavior, and output naming."""

from pathlib import Path

from pdf_extractor.cli import _build_output_path
from pdf_extractor.extractor import extract_pdf, find_pdf_files


class FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class FakeReader:
    def __init__(self, _: str) -> None:
        self.pages = [FakePage("Page 1"), FakePage("Page 2")]
        self.metadata = {"/Title": "Example"}


def test_extract_pdf(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr("pdf_extractor.extractor.PdfReader", FakeReader)

    document = extract_pdf(pdf_path)

    assert document.page_count == 2
    assert document.metadata == {"Title": "Example"}
    assert document.combined_text == "Page 1\n\nPage 2"


def test_find_pdf_files_non_recursive(tmp_path: Path) -> None:
    top_level_pdf = tmp_path / "top.pdf"
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested_pdf = nested_dir / "nested.pdf"

    top_level_pdf.write_bytes(b"pdf")
    nested_pdf.write_bytes(b"pdf")

    result = find_pdf_files(tmp_path)

    assert result == [top_level_pdf]


def test_find_pdf_files_recursive(tmp_path: Path) -> None:
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested_pdf = nested_dir / "nested.pdf"
    nested_pdf.write_bytes(b"pdf")

    result = find_pdf_files(tmp_path, recursive=True)

    assert result == [nested_pdf]


def test_build_output_path() -> None:
    output_path = _build_output_path(Path("output"), "source.pdf", "json")

    assert output_path == Path("output/source.json")