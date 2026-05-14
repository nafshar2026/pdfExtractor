from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Callable

from pypdf import PdfReader, PdfWriter


def chunk_document_groups(groups: list[list[int]], max_pages: int) -> list[list[list[int]]]:
    """Batch whole document groups into chunks of approximately max_pages."""
    if max_pages <= 0 or not groups:
        return [groups] if groups else []

    chunks: list[list[list[int]]] = []
    current_chunk: list[list[int]] = []
    current_pages = 0

    for group in groups:
        group_pages = len(group)
        if current_chunk and (current_pages + group_pages) > max_pages:
            chunks.append(current_chunk)
            current_chunk = []
            current_pages = 0

        current_chunk.append(group)
        current_pages += group_pages

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def fixed_page_chunks(total_pages: int, chunk_pages: int) -> list[tuple[int, int]]:
    """Return fixed [start, end) page chunks for a document."""
    if total_pages <= 0:
        return []
    if chunk_pages <= 0:
        return [(0, total_pages)]

    chunks: list[tuple[int, int]] = []
    start = 0
    while start < total_pages:
        end = min(total_pages, start + chunk_pages)
        chunks.append((start, end))
        start = end
    return chunks


def windowed_groups_from_signals(
    signals: list[tuple[int, Any]],
    chunk_pages: int,
    *,
    group_pages_fn: Callable[[list[tuple[int, Any]]], list[list[int]]],
) -> list[list[int]]:
    """Group signals via sliding two-chunk windows."""
    if not signals:
        return []

    if chunk_pages <= 0:
        return group_pages_fn(signals)

    signal_by_idx = {idx: sig for idx, sig in signals}
    total_pages = max(idx for idx, _ in signals) + 1
    chunks = fixed_page_chunks(total_pages, chunk_pages)

    if len(chunks) <= 1:
        return group_pages_fn(signals)

    emitted: list[list[int]] = []
    seen_groups: set[tuple[int, ...]] = set()

    for win_idx in range(len(chunks) - 1):
        left_start = chunks[win_idx][0]
        right_start = chunks[win_idx + 1][0]
        window_end = chunks[win_idx + 1][1]
        is_last_window = win_idx == (len(chunks) - 2)

        window_signals = [
            (i, signal_by_idx[i])
            for i in range(left_start, window_end)
            if i in signal_by_idx
        ]
        window_groups = group_pages_fn(window_signals)

        for group in window_groups:
            if not group:
                continue

            first_idx = group[0]
            first_signal = signal_by_idx.get(first_idx)

            if (
                left_start > 0
                and first_idx == left_start
                and (first_signal is None or first_signal.classification != "NEW_DOC")
            ):
                continue

            if not is_last_window and first_idx >= right_start:
                continue

            key = tuple(group)
            if key in seen_groups:
                continue
            seen_groups.add(key)
            emitted.append(group)

    return emitted


def write_fixed_chunk_files(
    reader: PdfReader,
    chunks: list[tuple[int, int]],
    chunk_dir: Path,
) -> list[Path]:
    """Write fixed page chunks to disk and return their file paths."""
    chunk_paths: list[Path] = []
    for idx, (start, end) in enumerate(chunks, start=1):
        writer = PdfWriter()
        for page_idx in range(start, end):
            writer.add_page(reader.pages[page_idx])

        chunk_path = chunk_dir / f"chunk_{idx:04d}.pdf"
        with chunk_path.open("wb") as handle:
            writer.write(handle)
        chunk_paths.append(chunk_path)

    return chunk_paths


def analyze_chunk_file_signals(
    chunk_path: Path,
    abs_start: int,
    *,
    analyze_page_fn: Callable[..., Any],
    shutdown_ocr_pool_fn: Callable[[], None],
) -> list[tuple[int, Any]]:
    """Analyze one on-disk chunk and return absolute-index page signals."""
    local_reader = PdfReader(str(chunk_path))

    import fitz

    local_fitz = fitz.open(str(chunk_path))
    chunk_signals: list[tuple[int, Any]] = []
    try:
        for local_idx in range(len(local_reader.pages)):
            abs_idx = abs_start + local_idx
            signal = analyze_page_fn(local_reader.pages[local_idx], fitz_doc=local_fitz, page_idx=local_idx)
            chunk_signals.append((abs_idx, signal))
            gc.collect()
    finally:
        local_fitz.close()
        shutdown_ocr_pool_fn()
        gc.collect()

    return chunk_signals


def windowed_groups_from_chunk_files(
    chunk_ranges: list[tuple[int, int]],
    chunk_paths: list[Path],
    *,
    group_pages_fn: Callable[[list[tuple[int, Any]]], list[list[int]]],
    analyze_chunk_file_signals_fn: Callable[[Path, int], list[tuple[int, Any]]],
) -> tuple[list[list[int]], dict[int, Any]]:
    """Group pages via two-chunk sliding windows using on-disk chunk files."""
    if not chunk_paths:
        return [], {}

    if len(chunk_paths) == 1:
        all_signals = analyze_chunk_file_signals_fn(chunk_paths[0], chunk_ranges[0][0])
        return group_pages_fn(all_signals), dict(all_signals)

    emitted_groups: list[list[int]] = []
    emitted_signals: dict[int, Any] = {}
    seen_groups: set[tuple[int, ...]] = set()

    left_start = chunk_ranges[0][0]
    left_signals = analyze_chunk_file_signals_fn(chunk_paths[0], left_start)

    for win_idx in range(len(chunk_paths) - 1):
        right_start = chunk_ranges[win_idx + 1][0]
        right_signals = analyze_chunk_file_signals_fn(chunk_paths[win_idx + 1], right_start)

        window_signals = left_signals + right_signals
        signal_by_idx = dict(window_signals)
        window_groups = group_pages_fn(window_signals)

        is_last_window = win_idx == (len(chunk_paths) - 2)

        for group in window_groups:
            if not group:
                continue

            first_idx = group[0]
            first_signal = signal_by_idx.get(first_idx)

            if (
                left_start > 0
                and first_idx == left_start
                and (first_signal is None or first_signal.classification != "NEW_DOC")
            ):
                continue

            if not is_last_window and first_idx >= right_start:
                continue

            key = tuple(group)
            if key in seen_groups:
                continue
            seen_groups.add(key)
            emitted_groups.append(group)

            for idx in group:
                if idx in signal_by_idx:
                    emitted_signals[idx] = signal_by_idx[idx]

        del window_signals, signal_by_idx, window_groups

        left_start = right_start
        left_signals = right_signals
        gc.collect()

    return emitted_groups, emitted_signals


def iter_windowed_groups_from_chunk_files(
    chunk_ranges: list[tuple[int, int]],
    chunk_paths: list[Path],
    *,
    group_pages_fn: Callable[[list[tuple[int, Any]]], list[list[int]]],
    analyze_chunk_file_signals_fn: Callable[[Path, int], list[tuple[int, Any]]],
):
    """Yield dedup-candidate groups from disk-backed two-chunk windows."""
    if not chunk_paths:
        return

    if len(chunk_paths) == 1:
        all_signals = analyze_chunk_file_signals_fn(chunk_paths[0], chunk_ranges[0][0])
        signal_by_idx = dict(all_signals)
        for group in group_pages_fn(all_signals):
            yield group, {idx: signal_by_idx[idx] for idx in group if idx in signal_by_idx}
        return

    seen_groups: set[tuple[int, ...]] = set()

    left_start = chunk_ranges[0][0]
    left_signals = analyze_chunk_file_signals_fn(chunk_paths[0], left_start)

    for win_idx in range(len(chunk_paths) - 1):
        right_start = chunk_ranges[win_idx + 1][0]
        right_signals = analyze_chunk_file_signals_fn(chunk_paths[win_idx + 1], right_start)

        window_signals = left_signals + right_signals
        signal_by_idx = dict(window_signals)
        window_groups = group_pages_fn(window_signals)

        is_last_window = win_idx == (len(chunk_paths) - 2)
        for group in window_groups:
            if not group:
                continue

            first_idx = group[0]
            first_signal = signal_by_idx.get(first_idx)

            if (
                left_start > 0
                and first_idx == left_start
                and (first_signal is None or first_signal.classification != "NEW_DOC")
            ):
                continue

            if not is_last_window and first_idx >= right_start:
                continue

            key = tuple(group)
            if key in seen_groups:
                continue
            seen_groups.add(key)

            yield group, {idx: signal_by_idx[idx] for idx in group if idx in signal_by_idx}

        del window_signals, signal_by_idx, window_groups

        left_start = right_start
        left_signals = right_signals
        gc.collect()
