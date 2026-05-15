from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np


_worker_ocr_instance = None


def _worker_run_ocr(image_array: np.ndarray):
    """Run OCR in worker process with a process-local PaddleOCR singleton."""
    global _worker_ocr_instance
    if _worker_ocr_instance is None:
        from paddleocr import PaddleOCR
        _worker_ocr_instance = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
    return _worker_ocr_instance.ocr(image_array)


class OcrRuntime:
    """Manage in-process or isolated-subprocess OCR execution."""

    def __init__(self, *, isolated: bool, recycle_calls: int, pool_retries: int):
        self._isolated = isolated
        self._recycle_calls = max(0, recycle_calls)
        self._pool_retries = max(0, pool_retries)
        self._ocr_instance = None
        self._ocr_pool: ProcessPoolExecutor | None = None
        self._ocr_pool_calls = 0

    def _get_ocr(self):
        if self._ocr_instance is None:
            from paddleocr import PaddleOCR
            self._ocr_instance = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
        return self._ocr_instance

    def _get_ocr_pool(self) -> ProcessPoolExecutor:
        if self._ocr_pool is None:
            mp_ctx = multiprocessing.get_context("spawn")
            max_tasks = self._recycle_calls if self._recycle_calls > 0 else None
            self._ocr_pool = ProcessPoolExecutor(
                max_workers=1,
                mp_context=mp_ctx,
                max_tasks_per_child=max_tasks,
            )
        return self._ocr_pool

    def shutdown(self) -> None:
        if self._ocr_pool is not None:
            self._ocr_pool.shutdown(wait=True, cancel_futures=True)
            self._ocr_pool = None
        self._ocr_pool_calls = 0

    def _isolated_infer(self, arr: np.ndarray):
        """Single isolated-subprocess OCR attempt; raises on BrokenProcessPool."""
        attempts = self._pool_retries + 1
        for attempt in range(attempts):
            try:
                pool = self._get_ocr_pool()
                result = pool.submit(_worker_run_ocr, arr).result()
                self._ocr_pool_calls += 1
                if self._recycle_calls > 0 and self._ocr_pool_calls >= self._recycle_calls:
                    self.shutdown()
                return result
            except BrokenProcessPool:
                self.shutdown()
                if attempt >= attempts - 1:
                    raise

    def infer(self, image_array: np.ndarray):
        """Run OCR either in-process or in an isolated subprocess worker.

        In isolated mode, an empty result on the first call after a worker
        recycle (model warm-up) is retried once automatically so that callers
        never receive a silent empty response for a non-blank image strip.
        """
        if not self._isolated:
            return self._get_ocr().ocr(image_array)

        arr = np.ascontiguousarray(image_array)
        result = self._isolated_infer(arr)

        # Empty result right after worker recycle: the new worker spent its first
        # call loading the model and returned nothing. Retry once — the worker is
        # now warmed up and will produce real output.
        if result is None or not result or not result[0]:
            result = self._isolated_infer(arr)

        return result
