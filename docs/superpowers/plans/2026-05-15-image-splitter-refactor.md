# image_splitter.py Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `src/pdf_extractor/image_splitter.py` (1,686 lines) into three focused modules — `title_detection.py`, `page_analysis.py`, and `dedup.py` — reducing the main file to ~450 lines.

**Architecture:** Pure code movement, no logic changes. Each task extracts one cluster of functions into a new file, updates `image_splitter.py` to import from it, updates test imports, and verifies the full suite still passes. Tasks are independent and ordered by dependency: title_detection has no internal deps; page_analysis depends on title_detection; dedup depends on both.

**Tech Stack:** Python 3.11. No new dependencies. Existing 87 tests are the verification mechanism throughout.

---

## Baseline

Before starting, confirm the test baseline:

```
cd c:\Users\nafsha44\source\repos\pdfExtractor
.venv/Scripts/python -m pytest tests/ -q
```

Expected: `87 passed`.

---

## File Map

| File | Action | Result |
|---|---|---|
| `src/pdf_extractor/title_detection.py` | Create | ~490 lines of title-detection heuristics |
| `src/pdf_extractor/page_analysis.py` | Create | ~430 lines of page analysis |
| `src/pdf_extractor/dedup.py` | Create | ~150 lines of deduplication helpers |
| `src/pdf_extractor/image_splitter.py` | Shrink | ~450 lines — grouping, output, `split_pdf` |
| `tests/test_image_splitter.py` | Update imports | 7 import lines updated |

---

## Task 1: Extract `title_detection.py`

Moves all title-detection heuristics — constants, regex patterns, and the functions that decide whether a text string is a document title. No callers outside `image_splitter.py` change yet; `image_splitter.py` will import from this new module.

**Files:**
- Create: `src/pdf_extractor/title_detection.py`
- Modify: `src/pdf_extractor/image_splitter.py`
- Modify: `tests/test_image_splitter.py`

---

- [ ] **Step 1: Verify baseline**

```
cd c:\Users\nafsha44\source\repos\pdfExtractor
.venv/Scripts/python -m pytest tests/ -q
```

Expected: `87 passed`. Do not proceed if tests fail.

---

- [ ] **Step 2: Create `src/pdf_extractor/title_detection.py`**

Create the file with this header, then copy the listed symbols verbatim from `image_splitter.py` (do not change any logic):

```python
"""Title detection heuristics for text and OCR pages."""
from __future__ import annotations

import re

from .extractor import _DOC_MARKER_RE, _sanitize_filename
```

**Copy these constants verbatim from `image_splitter.py` (in this order):**
1. `_TITLE_MAX_CHARS`
2. `_TOP_STRIP_FRACTION`
3. `_BOTTOM_STRIP_FRACTION`
4. `_TEXT_TITLE_TOP_FRACTION`
5. `_TEXT_TITLE_SKIP_RE`
6. `_TEXT_TITLE_SECTION_RE`
7. `_TITLE_DOCWORD_RE`
8. `_TITLE_STRONG_DOCWORD_RE`
9. `_TITLE_WEAK_HEADER_RE`
10. `_DT_FOOTER_RE`
11. `_DT_CREDIT_APP_RE`

**Copy these functions verbatim from `image_splitter.py` (in this order):**
1. `_infer_content_title`
2. `_normalize_detected_title`
3. `_filter_layout_noise_for_title`
4. `_looks_like_form_code_title`
5. `_title_appears_top_and_footer`
6. `_select_best_title`
7. `_title_key`
8. `_is_footer_variant`
9. `_extract_text_title`
10. `_extract_text_title_with_layout`

Note: `_extract_text_title_with_layout` uses `fitz_doc[page_idx]` and `page.get_text("dict")` directly — it receives an open fitz document as a parameter and does not import fitz at the module level. No fitz import is needed in `title_detection.py`.

---

- [ ] **Step 3: Update `image_splitter.py` — remove moved code, add import**

**Remove from `image_splitter.py`:**
- All 11 constants listed in Step 2
- All 10 functions listed in Step 2

**Add this import to `image_splitter.py`** (place it after the existing `from .extractor import ...` block):

```python
from .title_detection import (
    _infer_content_title,
    _is_footer_variant,
    _normalize_detected_title,
    _select_best_title,
    _title_key,
    _extract_text_title,
    _extract_text_title_with_layout,
)
```

Note: `_filter_layout_noise_for_title`, `_looks_like_form_code_title`, `_title_appears_top_and_footer`, and the regex constants are only used by functions within `title_detection.py` itself — they do not need to be imported back into `image_splitter.py`.

---

- [ ] **Step 4: Update test imports in `tests/test_image_splitter.py`**

Find line:
```python
from pdf_extractor.image_splitter import _analyze_text_page, _extract_text_title
```

Change to:
```python
from pdf_extractor.image_splitter import _analyze_text_page
from pdf_extractor.title_detection import _extract_text_title
```

---

- [ ] **Step 5: Run full suite**

```
.venv/Scripts/python -m pytest tests/ -q
```

Expected: `87 passed`. If any tests fail, fix before committing.

---

- [ ] **Step 6: Commit**

```
git add src/pdf_extractor/title_detection.py src/pdf_extractor/image_splitter.py tests/test_image_splitter.py
git commit -m "refactor: extract title_detection.py from image_splitter"
```

---

## Task 2: Extract `page_analysis.py`

Moves the `PageSignal` dataclass, OCR pool management, rendering, and the three `analyze_*` functions. After this task, `image_splitter.py` imports `PageSignal`, `analyze_page`, and `_shutdown_ocr_pool` from `page_analysis`.

**Files:**
- Create: `src/pdf_extractor/page_analysis.py`
- Modify: `src/pdf_extractor/image_splitter.py`
- Modify: `tests/test_image_splitter.py`

---

- [ ] **Step 1: Verify baseline**

```
.venv/Scripts/python -m pytest tests/ -q
```

Expected: `87 passed`.

---

- [ ] **Step 2: Create `src/pdf_extractor/page_analysis.py`**

Create the file with this header:

```python
"""Per-page boundary signal analysis for text and scanned PDF pages."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

from .extractor import _DOC_MARKER_RE, _page_lines
from .ocr_runtime import OcrRuntime
from .title_detection import (
    _extract_text_title,
    _extract_text_title_with_layout,
    _infer_content_title,
    _select_best_title,
)

logging.getLogger("ppocr").setLevel(logging.ERROR)
```

Then add this `_env_int` helper (it is duplicated here to keep `page_analysis.py` self-contained; `image_splitter.py` retains its own copy for chunk-size config):

```python
def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default
```

**Copy these constants verbatim from `image_splitter.py` (in this order):**
1. `_OCR_MAX_WIDTH_DEFAULT`
2. `_TEXT_PAGE_MIN_CHARS`
3. `_CONTINUATION_RE`

Then copy these runtime initialisation lines verbatim:

```python
_OCR_MAX_WIDTH = max(400, _env_int("PDF_EXTRACTOR_OCR_MAX_WIDTH", _OCR_MAX_WIDTH_DEFAULT))
```

And the OCR pool setup block verbatim:
```python
# (copy the _env_flag definition, _OCR_ISOLATED, _OCR_RECYCLE_CALLS,
#  _OCR_POOL_RETRIES, _OCR_RUNTIME = OcrRuntime(...) block)
```

Specifically, copy these in order:
- The `_env_flag` function definition
- `_OCR_ISOLATED = _env_flag(...)`
- `_OCR_RECYCLE_CALLS = max(...)`
- `_OCR_POOL_RETRIES = max(...)`
- `_OCR_RUNTIME = OcrRuntime(...)`

**Copy these functions verbatim from `image_splitter.py` (in this order):**
1. `_shutdown_ocr_pool`
2. `_ocr_infer`
3. `PageSignal` (the dataclass)
4. `_extract_ocr_texts`
5. `_render_page_fitz`
6. `_analyze_text_page`
7. `_analyze_image_page`
8. `analyze_page`

---

- [ ] **Step 3: Update `image_splitter.py` — remove moved code, add imports**

**Remove from `image_splitter.py`:**
- `_env_flag` function definition
- `_OCR_MAX_WIDTH_DEFAULT`, `_TEXT_PAGE_MIN_CHARS`, `_CONTINUATION_RE` constants
- `_OCR_ISOLATED`, `_OCR_RECYCLE_CALLS`, `_OCR_POOL_RETRIES`, `_OCR_MAX_WIDTH` lines
- `_OCR_RUNTIME = OcrRuntime(...)` block
- `_shutdown_ocr_pool`, `_ocr_infer`
- `PageSignal` dataclass
- `_extract_ocr_texts`, `_render_page_fitz`
- `_analyze_text_page`, `_analyze_image_page`, `analyze_page`
- The `logging.getLogger("ppocr")...` line (it moves to `page_analysis.py`)
- The `from .ocr_runtime import OcrRuntime` import (no longer needed in `image_splitter.py`)
- The `from dataclasses import dataclass` import (no longer needed)
- The `from typing import Literal` import (no longer needed)
- The `import numpy as np` import (no longer needed — verify before removing)

**Add this import to `image_splitter.py`** (after the `from .title_detection import ...` block):

```python
from .page_analysis import (
    PageSignal,
    analyze_page,
    _shutdown_ocr_pool,
    _extract_ocr_texts,
)
```

Note: `_extract_ocr_texts` is only needed here if `image_splitter.py` calls it directly. Check — if it is only used by the moved functions, omit it from this import. Keep only what `image_splitter.py` itself calls.

Also remove the now-redundant `from .title_detection import` entries for functions that are only called by `page_analysis.py` (i.e. `_infer_content_title`, `_is_footer_variant`, `_normalize_detected_title`, `_select_best_title`, `_extract_text_title`, `_extract_text_title_with_layout`). Keep only functions that `image_splitter.py` itself still calls directly.

---

- [ ] **Step 4: Update test imports in `tests/test_image_splitter.py`**

**Line 3** — change:
```python
from pdf_extractor.image_splitter import PageSignal, _extract_ocr_texts, _windowed_groups_from_signals
```
to:
```python
from pdf_extractor.image_splitter import _windowed_groups_from_signals
from pdf_extractor.page_analysis import PageSignal, _extract_ocr_texts
```

**Line ~43** — change:
```python
from pdf_extractor.image_splitter import analyze_page
```
to:
```python
from pdf_extractor.page_analysis import analyze_page
```

**Line ~330** — change (if not already updated in Task 1):
```python
from pdf_extractor.image_splitter import _analyze_text_page
```
to:
```python
from pdf_extractor.page_analysis import _analyze_text_page
```

---

- [ ] **Step 5: Run full suite**

```
.venv/Scripts/python -m pytest tests/ -q
```

Expected: `87 passed`. Fix any failures before committing.

---

- [ ] **Step 6: Commit**

```
git add src/pdf_extractor/page_analysis.py src/pdf_extractor/image_splitter.py tests/test_image_splitter.py
git commit -m "refactor: extract page_analysis.py from image_splitter"
```

---

## Task 3: Extract `dedup.py`

Moves all deduplication helpers and the phash report writer. After this task `image_splitter.py` imports them from `dedup`.

**Files:**
- Create: `src/pdf_extractor/dedup.py`
- Modify: `src/pdf_extractor/image_splitter.py`
- Modify: `tests/test_image_splitter.py`

---

- [ ] **Step 1: Verify baseline**

```
.venv/Scripts/python -m pytest tests/ -q
```

Expected: `87 passed`.

---

- [ ] **Step 2: Create `src/pdf_extractor/dedup.py`**

Create the file with this header:

```python
"""Deduplication helpers: byte hash, semantic key, perceptual hash, and report writer."""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from .page_analysis import PageSignal
from .title_detection import _title_key
```

**Copy these symbols verbatim from `image_splitter.py` (in this order):**
1. `_PHASH_THRESHOLD`
2. `_hamming_distance`
3. `_perceptual_hash`
4. `_group_primary_title`
5. `_semantic_title_key`
6. `_group_declared_total`
7. `_write_phash_report`

Note: `_perceptual_hash` uses `__import__("fitz").Matrix(...)` inline — no top-level `import fitz` is needed in `dedup.py`.

Note: `_semantic_title_key` calls `_title_key` — that is now imported from `.title_detection` above.

Note: `_group_primary_title` and `_group_declared_total` take `signals: dict[int, PageSignal]` parameters — `PageSignal` is now imported from `.page_analysis` above.

---

- [ ] **Step 3: Update `image_splitter.py` — remove moved code, add import**

**Remove from `image_splitter.py`:**
- `_PHASH_THRESHOLD`
- `_hamming_distance`
- `_perceptual_hash`
- `_group_primary_title`
- `_semantic_title_key`
- `_group_declared_total`
- `_write_phash_report`

**Add this import to `image_splitter.py`** (after the `from .page_analysis import ...` block):

```python
from .dedup import (
    _PHASH_THRESHOLD,
    _hamming_distance,
    _perceptual_hash,
    _group_primary_title,
    _semantic_title_key,
    _group_declared_total,
    _write_phash_report,
)
```

---

- [ ] **Step 4: Update test imports in `tests/test_image_splitter.py`**

**Line ~823** — change:
```python
from pdf_extractor.image_splitter import (
    _hamming_distance,
    _perceptual_hash,
    _PHASH_THRESHOLD,
)
```
to:
```python
from pdf_extractor.dedup import (
    _hamming_distance,
    _perceptual_hash,
    _PHASH_THRESHOLD,
)
```

**Line ~912** — change:
```python
from pdf_extractor.image_splitter import _write_phash_report
```
to:
```python
from pdf_extractor.dedup import _write_phash_report
```

---

- [ ] **Step 5: Run full suite**

```
.venv/Scripts/python -m pytest tests/ -q
```

Expected: `87 passed`. Fix any failures before committing.

---

- [ ] **Step 6: Commit**

```
git add src/pdf_extractor/dedup.py src/pdf_extractor/image_splitter.py tests/test_image_splitter.py
git commit -m "refactor: extract dedup.py from image_splitter"
```

---

## Final Verification

- [ ] **Confirm `image_splitter.py` line count**

```
python -c "print(sum(1 for _ in open('src/pdf_extractor/image_splitter.py')))"
```

Expected: ≤ 500 lines.

- [ ] **Confirm no circular imports**

```
.venv/Scripts/python -c "
from pdf_extractor import title_detection
from pdf_extractor import page_analysis
from pdf_extractor import dedup
from pdf_extractor import image_splitter
print('All imports OK')
"
```

Expected: `All imports OK` with no errors.

- [ ] **Run full suite one final time**

```
.venv/Scripts/python -m pytest tests/ -q
```

Expected: `87 passed`.
