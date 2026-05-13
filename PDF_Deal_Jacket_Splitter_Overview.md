# PDF Deal Jacket Splitter Overview

## Objective

Process large scanned deal-jacket PDFs reliably on the smallest practical hardware profile, while preserving document boundary accuracy and enabling batch throughput.

## What was implemented

### 1) OCR process isolation

- Added an isolated OCR worker mode in `src/pdf_extractor/image_splitter.py`.
- Controlled by `PDF_EXTRACTOR_OCR_ISOLATED=1`.
- Benefit: OCR memory is isolated from the main process and can be reclaimed by recycling workers.

### 2) Worker recycling

- Added configurable worker recycling via `PDF_EXTRACTOR_OCR_RECYCLE_CALLS`.
- Production profile uses `PDF_EXTRACTOR_OCR_RECYCLE_CALLS=6`.
- Benefit: prevents unbounded memory growth during long runs.

### 3) Retry on OCR worker failure

- Added retry behavior for broken OCR process pools via `PDF_EXTRACTOR_OCR_POOL_RETRIES`.
- Production profile uses `PDF_EXTRACTOR_OCR_POOL_RETRIES=2`.
- Benefit: transient worker crashes or OOM events do not fail the entire job immediately.

### 4) OCR render downscale cap

- Added configurable render width cap via `PDF_EXTRACTOR_OCR_MAX_WIDTH`.
- Production profile uses `PDF_EXTRACTOR_OCR_MAX_WIDTH=1200`.
- Benefit: lowers peak memory use per OCR pass while maintaining usable recognition quality.

### 5) Azure wildcard batch processing

- Added glob pattern support in Azure mode input path (`*`, `6*`, `abc*`, `?`, `[]`) in `src/pdf_extractor/cli.py`.
- Benefit: one run can process many files and generate one consolidated opt-in Excel output.

## Hardened production profile

Set these environment variables in Azure job configuration:

- `PDF_EXTRACTOR_OCR_ISOLATED=1`
- `PDF_EXTRACTOR_OCR_RECYCLE_CALLS=6`
- `PDF_EXTRACTOR_OCR_POOL_RETRIES=2`
- `PDF_EXTRACTOR_OCR_MAX_WIDTH=1200`

## Validation summary

- Large file success: `600157742.pdf`
- Large file success: `600157748.pdf`
- Wildcard batch success: `6*` matched 4 files and completed with one consolidated Excel output.

## Remaining gap

Duplicate split outputs in repeated packets are still emitted with numeric suffixes (for example `(2)`, `(3)`).
A hash-based deduplication stage remains the next enhancement.
