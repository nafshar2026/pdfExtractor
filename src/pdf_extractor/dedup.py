"""Deduplication helpers: byte hash, semantic key, perceptual hash, and report writer."""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from .page_analysis import PageSignal
from .title_detection import _title_key


_PHASH_THRESHOLD = 10


def _hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _perceptual_hash(fitz_doc, page_idx: int) -> int | None:
    """Average hash (aHash) of a page rendered at 64-pixel width.

    Renders at ~64 px wide (much smaller than OCR renders), converts to an
    8×8 grayscale thumbnail, and returns a 64-bit integer where bit i is 1
    if pixel i is >= the mean pixel value.  Returns None on any failure.
    """
    try:
        page = fitz_doc[page_idx]
        w_pt = page.rect.width
        if w_pt <= 0:
            return None
        scale = 64 / w_pt
        mat = __import__("fitz").Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        del pix
        thumb = img.convert("L").resize((8, 8), Image.LANCZOS)
        pixels = list(thumb.getdata())
        mean = sum(pixels) / len(pixels)
        bits = [1 if p >= mean else 0 for p in pixels]
        return sum(b << i for i, b in enumerate(bits))
    except Exception:
        return None


def _group_primary_title(group: list[int], signals: dict[int, PageSignal]) -> str | None:
    """Return the preferred title for a grouped document.

    Mirrors naming logic used when writing output files: for paginated groups,
    trust the title on page 1 of the logical document; otherwise use the first
    available title in the group.
    """
    has_pagination = any(
        (signals.get(idx) and signals[idx].page_num_in_doc is not None)
        for idx in group
    )

    if has_pagination:
        for idx in group:
            sig = signals.get(idx)
            if sig and sig.page_num_in_doc == 1 and sig.title_text:
                return sig.title_text
    else:
        for idx in group:
            sig = signals.get(idx)
            if sig and sig.title_text:
                return sig.title_text

    return None


def _semantic_title_key(raw_title: str | None) -> str | None:
    """Build a dedup key from title text that survives punctuation/noise.

    Rules:
    - remove trailing parenthetical copy markers like "(2)"
    - keep only alphanumeric characters
    - uppercase for case-insensitive comparison
    """
    if not raw_title:
        return None

    cleaned = re.sub(r"\s*\(\d+\)\s*$", "", raw_title).strip()
    key = _title_key(cleaned)
    return key or None


def _group_declared_total(group: list[int], signals: dict[int, PageSignal]) -> int | None:
    """Return declared total pages (N in Page X of N) when present."""
    for idx in group:
        sig = signals.get(idx)
        if sig and sig.total_pages_in_doc is not None:
            return sig.total_pages_in_doc
    return None


def _write_phash_report(
    phash_hits: list[tuple[int, int, int, str, str, int, int]],
    out_dir: Path,
    source_path: Path,
) -> None:
    """Print suspected-duplicate pairs to console and write suspected_duplicates.txt."""
    if not phash_hits:
        return
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"Suspected duplicates in {source_path.name} — {timestamp}"
    hit_lines = [
        f"  Group {a_idx} (page {page_a}) \"{title_a}\" ~ "
        f"Group {b_idx} (page {page_b}) \"{title_b}\" [distance: {dist}]"
        for a_idx, b_idx, dist, title_a, title_b, page_a, page_b in phash_hits
    ]
    print("\n--- SUSPECTED DUPLICATES (perceptual hash) ---")
    for line in hit_lines:
        print(line)
    report_path = out_dir / "suspected_duplicates.txt"
    report_path.write_text("\n".join([header] + hit_lines), encoding="utf-8")
    print(f"  (written to {report_path})")
