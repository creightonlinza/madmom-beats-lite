from __future__ import annotations

from typing import Callable

from .types import ProgressEvent

ProgressCallback = Callable[[ProgressEvent], None]


class ProgressReporter:
    def __init__(self, callback: ProgressCallback | None = None) -> None:
        self._callback = callback
        self._last_percent = -1

    def emit(self, percent: int, stage: str, message: str) -> ProgressEvent:
        normalized = max(0, min(100, int(percent)))
        if normalized < self._last_percent:
            normalized = self._last_percent
        event = ProgressEvent(percent=normalized, stage=stage, message=message)
        self._last_percent = normalized
        if self._callback is not None:
            self._callback(event)
        return event
