# Changelog

All notable changes to this project are documented in this file.

## 2026-05-13

### Added
- OCR process isolation via `PDF_EXTRACTOR_OCR_ISOLATED` to improve reliability on constrained hardware.
- OCR worker recycling via `PDF_EXTRACTOR_OCR_RECYCLE_CALLS` to limit memory growth over long runs.
- OCR worker retry handling via `PDF_EXTRACTOR_OCR_POOL_RETRIES` for crash/OOM resilience.
- OCR render-width cap via `PDF_EXTRACTOR_OCR_MAX_WIDTH` to reduce peak memory pressure on large scanned PDFs.
- Azure wildcard blob input support in CLI (`*`, `6*`, `abc*`, `?`, `[]`) for one-run batch processing.
- Project overview document [PDF_Deal_Jacket_Splitter_Overview.md](PDF_Deal_Jacket_Splitter_Overview.md) capturing the low-hardware large-file objective and outcomes.

### Changed
- Updated project docs to describe the hardened production profile for large files and batch wildcard execution.

### Validated
- Large-file Azure runs succeeded for `600157742.pdf` and `600157748.pdf` under hardened OCR settings.
- Wildcard batch run (`6*`) succeeded across 4 inputs and produced one consolidated Excel output.
