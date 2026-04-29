# pdfExtractor — Claude Code Context

## What This Project Does

Splits a multi-document PDF (e.g. an auto-finance deal jacket scanned at a Stellantis
dealership) into one PDF per logical document.  Each page is analyzed for structural
signals — page-numbering footers, embedded document markers, or a prominent title —
and the results are grouped into document runs before writing output files.

---

## Repository Layout

```
src/pdf_extractor/
    extractor.py        # Core pypdf utilities and data models
    image_splitter.py   # Unified split_pdf() entry point + all boundary-detection logic
    cli.py              # argparse CLI wired to split_pdf()
    genpdf.py           # Utility: generates text-based test PDFs via reportlab

tests/
    test_extractor.py
    test_image_splitter.py

src/pdf_extractor/Data/   # NOT in git — input PDFs go here
output/                   # NOT in git — split output goes here
```

---

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

# Run tests (53 tests)
.venv/Scripts/python -m pytest tests/ -v

# Generate the sample multi-document test PDF (text-based)
.venv/Scripts/python src/pdf_extractor/genpdf.py
```

---

## Architecture

### Unified pipeline (`split_pdf` in `image_splitter.py`)

Every page goes through `analyze_page()` regardless of whether it contains digital
text or a scanned image.  `analyze_page` decides internally which extraction method
to use:

```
split_pdf()
  for each page:
      analyze_page(pdf_page) → PageSignal
          if pypdf yields ≥ 50 chars  → _analyze_text_page(text)
          else                         → _analyze_image_page(pdf_page)  [PaddleOCR]
  _group_image_pages(signals) → groups of page indices
  for each group:
      _write_image_group(...)  → one PDF file
```

The grouping logic and the boundary signals are **identical for text and image pages**.
Only the extraction method differs.

### PageSignal

Every page produces one `PageSignal(classification, title_text, page_num_in_doc,
total_pages_in_doc)` where `classification` is one of:

| Value | Meaning |
|---|---|
| `NEW_DOC` | This page starts a new document |
| `CONTINUATION` | This page belongs to the current document |
| `AMBIGUOUS` | No boundary detected — attach to the previous document |

### Document boundary rules (in priority order)

These rules apply to **both text and image pages**:

1. **`DOCUMENT N OF N` marker** (text only) — an explicit boundary string embedded by
   the document assembly system.  Always `NEW_DOC`.

2. **`Page 1 of N` footer/header** — signals the first page of a paginated document.
   → `NEW_DOC`.  Carries `total_pages_in_doc=N` so the grouper can detect total-change
   splits even when intermediate pages lack footers.

3. **`Page N of M` (N > 1) footer/header** — continuation of the current document.
   → `CONTINUATION`.  If the "of M" total changes between consecutive CONTINUATION
   signals, the grouper starts a new document (two back-to-back paginated forms with
   no explicit NEW_DOC in between).

4. **Detectable ALL-CAPS title** (no page-number marker present) — the page has no
   explicit pagination but opens with a recognizable document title.  → `NEW_DOC`.
   Title-detection rules are in `_extract_text_title` (text) and `_select_best_title`
   (OCR).

5. **No signal** — page has no marker and no detectable title.  → `AMBIGUOUS`.
   The grouper attaches it to whatever document is currently open.

### `_extract_text_title` heuristic (text pages)

Searches the **first half of the page lines** for an ALL-CAPS phrase that qualifies
as a document title.  Returns `None` (→ AMBIGUOUS) if nothing qualifies.

Filters that **exclude** a line from consideration:
- Fewer than 3 words, or more than 8 words
- Longer than 60 characters
- Ends with a period (disclaimer sentences)
- Contains non-ASCII characters (bilingual form duplicate titles)
- Contains `:(#@©()`  (field labels, copyright notices, parenthetical form numbers)
- Matches `FORM NO.` pattern
- Matches `A. ` or `1. ` section-header prefix
- Matches `DOCUMENT N OF N` (already handled upstream)
- Corporate suffix: `LLC`, `INC`, `CORP`, `LTD`, `INCORPORATED`
- No word with ≥ 3 alphabetic characters (rules out "DT 5/23")
- First token is all-digits (house number → address line)
- ≥ 7 digit characters total (phone numbers, ZIP codes, VINs)

**`prefer_last` flag:** when `Page 1 of N` was found first, `prefer_last=True` is
passed so that required-disclosure boxes (e.g. `FEDERAL TRUTH-IN-LENDING DISCLOSURES`)
that appear before the contract name in pypdf's column-reading order are skipped in
favour of the last qualifying candidate in the first half of lines.

### `_select_best_title` heuristic (OCR / image pages)

Scores each OCR text line as `box_height × multiplier` where:
- `multiplier = 2.0` for multi-word strings (genuine titles)
- `multiplier = 0.3` for single words (logos, column headers)

Minimum score to qualify: **50.0** (box height 25 px multi-word, or 167 px
single-word).

Same exclusion rules as above (digit count, garbled lowercase, length) applied to
the OCR text before scoring.

### Grouping logic (`_group_image_pages`)

- `NEW_DOC` → save the current group, start a new one.
- `CONTINUATION` → append to current group; if `total_pages_in_doc` changes vs. the
  active group's total, split first.
- `AMBIGUOUS` → append to current group (or start one if none is open).

### Output file naming

The first page's `title_text` drives the filename:
- Image pages: `_sanitize_image_title()` — replaces spaces with underscores, strips
  non-alphanumeric except dash.
- Text pages (DOCUMENT marker): `_sanitize_filename()` — strips invalid filename
  characters, preserves spaces.
- Fallback (no title): `pages_X-Y` or `page_X` (1-based page numbers).
- Duplicate names: ` (2)`, ` (3)` suffixes.

---

## Design Constraints

**Do not add hardcoded document names or form identifiers to the boundary-detection
logic.**  Patterns like `"DT 5/23"`, `"ODOMETER DISCLOSURE STATEMENT"`, or
`"ASSIGNMENT OF CREDIT CONTRACT"` must never appear as detection rules.  Only
structural signals (page numbering, DOCUMENT markers, ALL-CAPS title heuristics)
are permitted.  The same code must work for Sample-2 through Sample-8 without
modification.

---

## Input Data

The source file was too large to process in one session and was split into 8 files
(`Sample-1.pdf` through `Sample-8.pdf`), stored in `src/pdf_extractor/Data/`.
None of these are checked into git.

`Sample-1.pdf` (50 pages) was processed on 2026-04-29, producing 32 output documents.
The file contains two complete copies of the same deal jacket (pages 1–30 and 31–50),
so the `(2)` and `(3)` suffixes on many output files are correct.

---

## Known Limitations / Next Steps

- Pages 8–10 of Sample-1 (DT 5/23 credit application, 3 pages, no page-number
  footer) attach to the preceding "Disclosure of 36% Rate Cap" document because they
  have no detectable title.  This is correct per the current rules but may not match
  the expected output.
- `VEhICLE_SERVICE_COnTrACT` — mixed-case OCR artifact in the title; caused by OCR
  reading the form at an angle.  Cosmetic only.
- Still to process: `Sample-2.pdf` through `Sample-8.pdf`.
- Consider a `--verbose` flag that prints each page's `PageSignal` to help diagnose
  mis-splits on new files.
