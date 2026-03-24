# PDF Extractor

This file documents what the project does, how to install it, and how to run the CLI.

Small Python CLI for extracting text and metadata from PDF files.

## Features

- Extract text from a single PDF or a full directory.
- Export each PDF as plain text or JSON.
- Include document metadata and page-level content in JSON output.
- Skip image-only pages gracefully when no embedded text is present.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

## Usage

Extract a single PDF into a text file:

```powershell
pdf-extractor input.pdf --output-dir output
```

Extract all PDFs in a folder recursively as JSON:

```powershell
pdf-extractor .\pdfs --output-dir output --format json --recursive
```

Run tests:

```powershell
pytest
```

## Output formats

- `text`: Writes one `.txt` file per input PDF.
- `json`: Writes one `.json` file per input PDF containing metadata, page count, and page text.

## Notes

- This project extracts embedded text. It does not perform OCR on scanned PDFs.
- For scanned documents, the next step would be integrating OCR with a tool like Tesseract.
