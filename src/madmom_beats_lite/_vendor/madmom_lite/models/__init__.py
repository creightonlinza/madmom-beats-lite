"""Model path definitions for vendored assets."""

from __future__ import annotations

import glob
from pathlib import Path

MODEL_PATH = str(Path(__file__).resolve().parents[3] / "assets")


def models(pattern: str, path: str = MODEL_PATH) -> list[str]:
    return sorted(glob.glob(f"{path}/{pattern}"))


DOWNBEATS_BLSTM = models("downbeats/2016/downbeats_blstm_[1-8].pkl")
