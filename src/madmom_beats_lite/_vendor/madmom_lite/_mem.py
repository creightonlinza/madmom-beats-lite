# encoding: utf-8
"""Lightweight stage-level RSS/array-size logging helpers."""

from __future__ import annotations

import resource
import sys

_RSS_SCALE = 1 if sys.platform == "darwin" else 1024


def _format_mb(value: int) -> str:
    return f"{value / (1024.0 * 1024.0):.2f}MB"


class MemoryLogger:
    """Low-overhead logger for coarse memory diagnostics."""

    __slots__ = ("enabled", "_max_rss")

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self._max_rss = 0

    def _rss_bytes(self) -> int:
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * _RSS_SCALE
        if rss > self._max_rss:
            self._max_rss = rss
        return rss

    def log(self, stage: str, **arrays) -> None:
        if not self.enabled:
            return
        parts = []
        total_bytes = 0
        for name, value in arrays.items():
            nbytes = getattr(value, "nbytes", None)
            if nbytes is None:
                continue
            nbytes = int(nbytes)
            total_bytes += nbytes
            parts.append(f"{name}={_format_mb(nbytes)}")
        detail = ", ".join(parts) if parts else "-"
        rss_now = self._rss_bytes()
        print(
            "mbl_mem "
            f"stage={stage} "
            f"arrays=[{detail}] "
            f"arrays_total={_format_mb(total_bytes)} "
            f"rss_current={_format_mb(rss_now)} "
            f"rss_max={_format_mb(self._max_rss)}"
        )
