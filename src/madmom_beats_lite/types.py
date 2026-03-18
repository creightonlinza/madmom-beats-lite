from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProgressEvent:
    percent: int
    stage: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "percent": int(self.percent),
            "stage": self.stage,
            "message": self.message,
        }


@dataclass(frozen=True)
class BeatResult:
    fps: int
    beat_times: list[float]
    beat_numbers: list[int]
    beat_confidences: list[float]
    downbeat_times: list[float]
    downbeat_confidences: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fps": int(self.fps),
            "beat_times": [float(x) for x in self.beat_times],
            "beat_numbers": [int(x) for x in self.beat_numbers],
            "beat_confidences": [float(x) for x in self.beat_confidences],
            "downbeat_times": [float(x) for x in self.downbeat_times],
            "downbeat_confidences": [float(x) for x in self.downbeat_confidences],
        }
