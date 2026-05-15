# image_splitter.py Refactor — Design Spec

**Date:** 2026-05-15
**Status:** Approved

---

## Problem

`src/pdf_extractor/image_splitter.py` has grown to 1,686 lines and is responsible for too many things: title-detection heuristics, page analysis, deduplication logic, grouping, output writing, and the top-level `split_pdf` orchestrator. Two modules have already been extracted (`ocr_runtime.py`, `overlap_splitter.py`), but the remaining file is still difficult to navigate and edit.

---

## Goal

Split `image_splitter.py` into three new focused modules plus a reduced orchestrator, without changing any behaviour. Pure code movement — no logic changes.

---

## Approach

**Option B (chosen):** Extract three modules. Each move is independent and tested before the next begins.

---

## Module Map

All files live under `src/pdf_extractor/`.

### `title_detection.py` (~490 lines) — NEW

Everything that reads text or OCR output and decides whether it is a document title.

**Contents moved from `image_splitter.py`:**
- `_infer_content_title`
- `_normalize_detected_title`
- `_filter_layout_noise_for_title`
- `_looks_like_form_code_title`
- `_title_appears_top_and_footer`
- `_select_best_title`
- `_title_key`
- `_is_footer_variant`
- `_extract_text_title`
- `_extract_text_title_with_layout`

**Imports:** `re`, `fitz`, `extractor._DOC_MARKER_RE`, `extractor._page_lines`, `extractor._sanitize_filename`

---

### `page_analysis.py` (~430 lines) — NEW

Everything that turns a single PDF page into a `PageSignal`.

**Contents moved from `image_splitter.py`:**
- `PageSignal` dataclass
- `_shutdown_ocr_pool`, `_ocr_infer` (OCR pool wrappers)
- `_extract_ocr_texts`
- `_render_page_fitz`
- `_analyze_text_page`
- `_analyze_image_page`
- `analyze_page` (public API)

**Imports:** `title_detection` (for `_extract_text_title`, `_extract_text_title_with_layout`, `_select_best_title`), `ocr_runtime.OcrRuntime`, `PIL.Image`, `fitz`, `numpy`

---

### `dedup.py` (~150 lines) — NEW

All deduplication helpers and the suspected-duplicate report.

**Contents moved from `image_splitter.py`:**
- `_PHASH_THRESHOLD`
- `_hamming_distance`
- `_perceptual_hash`
- `_group_primary_title`
- `_semantic_title_key`
- `_group_declared_total`
- `_write_phash_report`

**Imports:** `page_analysis.PageSignal` (type hint), `title_detection._title_key`, `PIL.Image`, `fitz`, `pathlib.Path`, `re`

---

### `image_splitter.py` (reduced to ~450 lines) — MODIFIED

Top-level orchestrator: grouping, output writing, and `split_pdf`.

**Stays here:**
- Env/config helpers (`_env_flag`, `_env_int`) and constants (`_OVERLAP_CHUNK_PAGES`, `_CHUNK_MAX_PAGES`, `_OCR_MAX_WIDTH`)
- `_group_image_pages`
- `_sanitize_image_title`, `_write_image_group`
- Thin wrappers delegating to `overlap_splitter` (`_chunk_document_groups`, `_fixed_page_chunks`, `_windowed_groups_from_signals`, `_write_fixed_chunk_files`, `_analyze_chunk_file_signals`, `_windowed_groups_from_chunk_files`, `_iter_windowed_groups_from_chunk_files`)
- `split_pdf`

**New imports added:** `page_analysis` (PageSignal, analyze_page, _shutdown_ocr_pool), `dedup` (_PHASH_THRESHOLD, _hamming_distance, _perceptual_hash, _group_primary_title, _semantic_title_key, _group_declared_total, _write_phash_report)

---

## Dependency Graph

```
extractor.py          (existing, unchanged)
ocr_runtime.py        (existing, unchanged)
overlap_splitter.py   (existing, unchanged)

title_detection.py    ← extractor, fitz, re
page_analysis.py      ← title_detection, ocr_runtime, PIL, fitz, numpy
dedup.py              ← page_analysis (type hint), title_detection, PIL, fitz, Path, re
image_splitter.py     ← page_analysis, dedup, overlap_splitter, extractor, pypdf, fitz
```

No circular imports. `image_splitter.py` is the only consumer of `page_analysis`, `dedup`, and `title_detection` from within the package (outside of tests).

---

## Test Migration

All 87 existing tests import from `pdf_extractor.image_splitter`. After the refactor, imports are updated to reflect actual module locations:

| Symbol | Updated import |
|---|---|
| `PageSignal`, `analyze_page`, `_extract_ocr_texts`, `_render_page_fitz`, `_analyze_text_page`, `_analyze_image_page` | `pdf_extractor.page_analysis` |
| `_extract_text_title`, `_select_best_title`, `_title_key`, `_normalize_detected_title` | `pdf_extractor.title_detection` |
| `_hamming_distance`, `_perceptual_hash`, `_PHASH_THRESHOLD`, `_write_phash_report` | `pdf_extractor.dedup` |
| `_group_image_pages`, `_windowed_groups_from_signals`, `split_pdf` | `pdf_extractor.image_splitter` (unchanged) |

`pdf_extractor/__init__.py` requires no changes (does not re-export symbols).

---

## Execution Strategy

**One module at a time, test after each:**

1. Extract `title_detection.py` → run `pytest` → commit
2. Extract `page_analysis.py` → run `pytest` → commit
3. Extract `dedup.py` → run `pytest` → commit

Each step is independent. If a step breaks tests, it is fixed before moving to the next module. No logic changes at any step — only file moves and import updates.

---

## Success Criteria

- `image_splitter.py` is ≤ 500 lines
- All 87 tests pass unchanged (imports updated, behaviour identical)
- No circular imports
- Each new module is importable independently
