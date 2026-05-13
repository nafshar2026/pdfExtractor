# Image PDF Splitter — Design Spec

**Date:** 2026-04-23
**Status:** Approved

---

## Problem

The existing `split` command uses `pypdf` text extraction to detect document boundaries via "DOCUMENT X OF Y" and "Page X of Y" markers. It works correctly for text-based PDF pages but produces no output for image-only pages (scanned documents saved as page images), which have zero extractable text.

`Sample.pdf` is a representative input: pages 1–12 are text-based (one dealer packet), pages 13–379 are image-only (many more scanned packets), page 380 is mixed. All pages need to be split into individual document PDFs.

---

## Goal

Extend the `split` command to handle image-only pages by using PaddleOCR to detect document boundaries, producing one PDF per logical document regardless of whether the source pages contain embedded text or scanned images.

---

## Boundary Detection Strategy

Title detection is the **primary signal**. A page with a detectable title near its top is the start of a new document. The "Page X of Y" continuation marker is a **safety override** — if X > 1, the page is a continuation regardless of any title detected.

### Decision logic per page

| Condition | Classification |
|-----------|---------------|
| "Page X of Y" detected AND X > 1 | `CONTINUATION` — always, overrides title |
| Title detected (and not overridden above) | `NEW_DOC` |
| No signal | `AMBIGUOUS` — appended to current document |

### Why "Page 1 of N" is ignored

The total count N reflects what was printed on the original form, not how many pages were actually scanned into this PDF. A page reading "1 of 6" may be immediately followed by another "1 of 7" — both are separate documents. The count is unreliable for grouping.

---

## Architecture

### New module: `src/pdf_extractor/image_splitter.py`

Contains all image-splitting logic. The existing `extractor.py` is unchanged.

#### `PageSignal` dataclass

```python
@dataclass(slots=True)
class PageSignal:
    classification: Literal["NEW_DOC", "CONTINUATION", "AMBIGUOUS"]
    title_text: str | None        # detected title text, if any
    page_num_in_doc: int | None   # X from "Page X of Y", if detected
```

#### `analyze_page(pdf_page) -> PageSignal`

1. Extract the embedded raster image from the PDF page via `pypdf` (`page.images[0]`)
2. Convert to a `Pillow` `Image`
3. **Bottom strip** (bottom 15%): run PaddleOCR, search for `Page\s+(\d+)\s+of\s+(\d+)` — if X > 1 → return `CONTINUATION`
4. **Top strip** (top 25%): run PaddleOCR, check if first detected text block is short (≤ 60 chars) and positioned in the upper half of the strip → title detected → return `NEW_DOC`
5. If neither fires → return `AMBIGUOUS`

Pages where `extract_text()` returns substantial content (≥ 50 chars) are **text-based** and bypass `analyze_page` entirely; the existing marker-based logic handles them.

#### `split_pdf(pdf_path, output_dir) -> list[Path]`

Unified entry point that handles both text-based and image-based pages in a single pass:

```
for each page:
    if page has substantial text:
        apply existing document-marker detection
    else:
        call analyze_page() → PageSignal
        apply boundary logic above
```

Flushes accumulated pages to a PDF whenever a `NEW_DOC` signal arrives. Flushes the final group after the last page.

---

## Output File Naming

1. **Title detected**: sanitize title text → replace non-alphanumeric runs with `_`, strip leading/trailing underscores → `{title}.pdf`
   - Example: `ST-556 State Tax Transaction Return` → `ST-556_State_Tax_Transaction_Return.pdf`
2. **No title** (first page was `AMBIGUOUS`): `pages_{start}-{end}.pdf` using 1-based source page numbers
   - Example: pages 31–33 → `pages_31-33.pdf`

If a file with the same name already exists in the output directory it is overwritten. The output directory is never cleared automatically — that is a manual step outside the tool.

---

## CLI

The existing `split` subcommand is updated to call `split_pdf` instead of `split_pdf_by_internal_documents`:

```
pdf-extractor split <pdf_path> [--output-dir <dir>]
```

Progress is printed per document as each PDF is written:

```
[1] Applicable Law → Applicable_Law.pdf (6 pages)
[2] FORM NO. LAWIL-RATECAP_e → FORM_NO_LAWIL-RATECAP_e.pdf (1 page)
...
[12] ST-556 State Tax Transaction Return → ST-556_State_Tax_Transaction_Return.pdf (2 pages)
```

---

## New Dependencies

Added to `pyproject.toml` under `[project.dependencies]`:

| Package | Purpose |
|---------|---------|
| `paddlepaddle` | PaddleOCR backend (CPU build) |
| `paddleocr` | OCR and text detection |
| `Pillow` | Image cropping and conversion |

---

## Error Handling

- **Page with no embedded images and no text**: skip silently, log a warning with the page number.
- **OCR failure on a page**: treat as `AMBIGUOUS`, continue.
- **Empty output group** (e.g., a group with zero pages): skip without writing a file.
- **Output directory does not exist**: create it with `mkdir(parents=True, exist_ok=True)`.

---

## Testing

- Unit test `analyze_page` by mocking `pypdf` page objects and PaddleOCR responses.
- Integration test `split_pdf` against `Sample.pdf`, asserting the known first 12 pages produce the same 7 files as the existing splitter.
- Test fallback naming (`pages_{start}-{end}.pdf`) for pages with no detected title.
- Test that `CONTINUATION` override suppresses a title on a mid-document page.

---

## Addendum (2026-05-13): Large-file + low-hardware hardening

### Objective

Run large scanned deal-jacket PDFs reliably on the smallest practical container hardware.

### Implemented hardening controls

- Isolated OCR worker process (`PDF_EXTRACTOR_OCR_ISOLATED=1`)
- Worker recycling (`PDF_EXTRACTOR_OCR_RECYCLE_CALLS`)
- Worker crash retry (`PDF_EXTRACTOR_OCR_POOL_RETRIES`)
- Render width cap before OCR (`PDF_EXTRACTOR_OCR_MAX_WIDTH`)

### Validated production profile

- `PDF_EXTRACTOR_OCR_ISOLATED=1`
- `PDF_EXTRACTOR_OCR_RECYCLE_CALLS=6`
- `PDF_EXTRACTOR_OCR_POOL_RETRIES=2`
- `PDF_EXTRACTOR_OCR_MAX_WIDTH=1200`

### Results

- Large files `600157742.pdf` and `600157748.pdf` completed successfully.
- Wildcard Azure batch (`6*`) completed with one combined Excel and split outputs for all matched inputs.

### CLI enhancement

Azure mode now supports glob patterns in `input_path` (`*`, `6*`, `abc*`, `?`, `[]`) and processes matched blobs in one run.
