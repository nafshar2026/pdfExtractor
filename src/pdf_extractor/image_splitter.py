"""Image-based PDF splitting using PaddleOCR title and continuation detection."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from pypdf import PdfReader, PdfWriter

from .extractor import (
    _detect_starts,
    _extract_title,
    _page_lines,
    _sanitize_filename,
)

_CONTINUATION_RE = re.compile(r"Page\s+(\d+)\s+of\s+\d+", re.IGNORECASE)
_TITLE_MAX_CHARS = 60
_TOP_STRIP_FRACTION = 0.25
_BOTTOM_STRIP_FRACTION = 0.15
_TEXT_PAGE_MIN_CHARS = 50

_ocr_instance: PaddleOCR | None = None


def _get_ocr() -> PaddleOCR:
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
    return _ocr_instance


@dataclass(slots=True)
class PageSignal:
    classification: Literal["NEW_DOC", "CONTINUATION", "AMBIGUOUS"]
    title_text: str | None
    page_num_in_doc: int | None


def _extract_ocr_texts(ocr_result: list | None) -> list[str]:
    if not ocr_result:
        return []
    page_result = ocr_result[0]
    if not page_result:
        return []
    texts = []
    for line in page_result:
        if line and len(line) >= 2 and line[1]:
            texts.append(line[1][0])
    return texts
