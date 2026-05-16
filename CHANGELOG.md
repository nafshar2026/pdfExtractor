# Changelog

All notable changes to this project are documented in this file.

## 2026-05-15

### Added
- Windowed chunking via `PDF_EXTRACTOR_OVERLAP_CHUNK_PAGES`: splits very large PDFs into
  overlapping N-page windows so memory stays bounded regardless of source file size.
  Enabled by default in the Docker image (`OVERLAP_CHUNK_PAGES=20`).
- Perceptual hash (aHash) deduplication: after splitting, the first page of each output
  document is fingerprinted and pairs within Hamming distance 10 are reported in
  `suspected_duplicates.txt`. Exact-byte and semantic-title dedup also run automatically.

### Fixed
- Memory scaling on large files: the full source fitz document was held open throughout
  all chunk windows, defeating windowed memory management. Each chunk now opens only the
  small chunk PDF it needs and closes it immediately.
- Title detection — address lines with an interior house number (e.g. "ADDRESS 2948
  GREENBRIAR") are now rejected regardless of token position, catching both standalone
  digits and OCR-merged forms like "Address2948".
- Title detection — Title Case titles (e.g. "Application for Texas Title and/or
  Registration") no longer rejected by the garbled-OCR lowercase filter; the filter now
  only fires when the first word itself is lowercase.
- Title detection — OCR merge artifacts where adjacent form-field labels are concatenated
  into a single CamelCase token (e.g. "DescriptionAdd", "LienOther") are now rejected.
- Docker image: removed PaddleOCR model pre-bake step that caused a segfault in the ACR
  build agent due to a cffi 2.0 / PaddlePaddle 2.6.2 incompatibility.
- `az acr build` in run.ps1 and README: added `--no-wait` to prevent a Unicode encoding
  crash in Azure CLI log streaming on Windows.

### Validated
- `602819077.pdf` (3,900+ pages) completed in 1h 42m on Azure with windowed chunking,
  producing 27 split documents.

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
