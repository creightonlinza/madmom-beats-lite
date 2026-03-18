from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def ensure_madmom_importable() -> None:
    try:
        import madmom  # noqa: F401
        return
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[2]
        vendor_path = repo_root / "vendor" / "madmom"
        if vendor_path.exists():
            sys.path.insert(0, str(vendor_path))
            import madmom  # noqa: F401
            return
        raise


def derive_confidences(
    activations: np.ndarray,
    beat_times: np.ndarray,
    beat_numbers: np.ndarray,
    fps: int,
) -> np.ndarray:
    if len(beat_times) == 0:
        return np.empty((0,), dtype=np.float32)
    frame_idx = np.rint(beat_times * float(fps)).astype(np.int64)
    frame_idx = np.clip(frame_idx, 0, len(activations) - 1)
    beat_scores = activations[frame_idx, 0]
    downbeat_scores = activations[frame_idx, 1]
    return np.where(beat_numbers == 1, downbeat_scores, beat_scores)


def load_audio_from_npz(path: Path) -> tuple[np.ndarray, int]:
    with np.load(path, allow_pickle=False) as payload:
        for key in ("audio", "samples", "waveform"):
            if key in payload:
                audio = payload[key]
                break
        else:
            raise ValueError("input .npz must contain one of: audio, samples, waveform")

        if "sample_rate" not in payload:
            raise ValueError("input .npz must contain sample_rate")
        sample_rate = int(np.asarray(payload["sample_rate"]).reshape(()).item())

    return audio, sample_rate
