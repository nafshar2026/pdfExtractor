# pdfExtractor — Claude Code Context

## What This Project Does

Splits a multi-document PDF (e.g. an auto-finance deal jacket) into one PDF per logical
document. Handles two page types:

- **Text pages** — pypdf extracts text directly; document boundaries are found via
  `DOCUMENT N OF N` markers, `Page 1 of N` patterns, and known form headers.
- **Image/scanned pages** — PaddleOCR reads the top strip for a title and the bottom
  strip for a `Page N of N` continuation marker. The OCR singleton is initialized once
  per process to avoid repeated model loading.

## Repository Layout

```
src/pdf_extractor/
    extractor.py        # Core text extraction, boundary detection, data models
    image_splitter.py   # PaddleOCR-based image page analysis + unified split_pdf()
    cli.py              # argparse CLI wired to both modes
    genpdf.py           # Utility: generates test PDFs via reportlab

tests/
    test_extractor.py
    test_image_splitter.py

src/pdf_extractor/Data/   # NOT in git — input PDFs go here
output/                   # NOT in git — split output goes here
```

## Key Commands

```bash
# One-time setup
py -3.11 -m venv .venv
.venv/Scripts/pip install -e ".[dev]"

# Split a PDF into separate documents
.venv/Scripts/python -c "from pdf_extractor.cli import main; raise SystemExit(main())" \
    src/pdf_extractor/Data/Sample-1.pdf \
    --split-documents \
    --split-output-dir output/split-1

# Run tests
.venv/Scripts/python -m pytest tests/ -v

# Generate the MuleSoft test PDF (text-based, multi-document)
.venv/Scripts/python src/pdf_extractor/genpdf.py
```

## Architecture Notes

### `split_pdf()` in image_splitter.py — unified entry point
1. Extracts text from every page with pypdf.
2. Classifies each page as `"text"` (≥ 50 chars) or `"image"` (< 50 chars).
3. Groups consecutive same-mode pages into runs.
4. Delegates text runs → `_split_text_run()` (uses `_detect_starts` from extractor.py).
5. Delegates image runs → `_split_image_run()` (calls PaddleOCR per page).

### PaddleOCR usage
- Singleton via `_get_ocr()` — initialised once, reused across all pages.
- Per page: crops **top 25%** for title detection, **bottom 15%** for `Page N of N`.
- `PageSignal` dataclass carries: `classification`, `title_text`, `page_num_in_doc`.
- `_group_image_pages()` groups signals into document runs; `CONTINUATION` pages attach
  to the current group, `NEW_DOC` starts a new one, `AMBIGUOUS` falls through to current.

### Naming output files
- Image pages: OCR title from first page → `_sanitize_image_title()`.
- Text pages: first meaningful line after boundary marker → `_sanitize_filename()`.
- Duplicate names get ` (2)`, ` (3)` suffixes.

## Input Data

The full source file was too large to process in one session, so it was split into 8
files (`Sample-1.pdf` through `Sample-8.pdf`), stored in `src/pdf_extractor/Data/`.
None of these are checked into git. `Sample-1.pdf` (50 pages) was successfully processed
on 2026-04-24, producing 39 output documents.

## Known Issues / Next Steps

- Some OCR titles are partial (e.g. `Contract_No.pdf` instead of the full form name).
  The OCR picks the first text in the top strip; a footer copyright line can win if it
  appears before the actual title.
- `Dt 523.pdf` vs `DT 523 © 2023 Dealertrack, Inc...` — same form, different OCR reads
  depending on whether the top strip captured the title or the footer bleed.
- Still to process: `Sample-2.pdf` through `Sample-8.pdf`.
- Consider adding a `--verbose` flag that prints each page's `PageSignal` classification
  to help diagnose mis-splits.
