"""Command-line interface for finding PDFs and writing extracted output files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .extractor import ExtractedDocument, extract_pdf, find_pdf_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract text and metadata from PDF files.",
    )
    parser.add_argument("input_path", help="PDF file or directory containing PDF files")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where extracted files will be written",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format for extracted content",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for PDFs when the input path is a directory",
    )
    return parser


def _build_output_path(output_dir: Path, source_path: str, output_format: str) -> Path:
    extension = ".txt" if output_format == "text" else ".json"
    return output_dir / f"{Path(source_path).stem}{extension}"


def _write_output(document: ExtractedDocument, output_dir: Path, output_format: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = _build_output_path(output_dir, document.source_path, output_format)

    if output_format == "json":
        destination.write_text(
            json.dumps(document.to_dict(), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
    else:
        destination.write_text(document.combined_text, encoding="utf-8")

    return destination


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        pdf_files = find_pdf_files(args.input_path, recursive=args.recursive)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    if not pdf_files:
        parser.error("No PDF files found in the given input path.")

    output_dir = Path(args.output_dir)
    written_files: list[Path] = []

    for pdf_file in pdf_files:
        document = extract_pdf(pdf_file)
        written_files.append(_write_output(document, output_dir, args.format))

    for written_file in written_files:
        print(written_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
