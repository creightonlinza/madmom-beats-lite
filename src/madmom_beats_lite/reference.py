from __future__ import annotations

from typing import Sequence

import numpy as np

from ._shared import derive_confidences, ensure_madmom_importable
from .types import BeatResult

def run_reference_extraction(
    audio: np.ndarray,
    sample_rate: int,
    *,
    fps: int = 100,
    beats_per_bar: Sequence[int] = (3, 4),
) -> BeatResult:
    ensure_madmom_importable()
    from madmom.audio.signal import Signal
    from madmom.features.downbeats import (
        DBNDownBeatTrackingProcessor,
        RNNDownBeatProcessor,
    )

    signal = Signal(audio, sample_rate=int(sample_rate))
    activations = RNNDownBeatProcessor()(signal)
    beats = DBNDownBeatTrackingProcessor(beats_per_bar=list(beats_per_bar), fps=int(fps))(activations)

    if beats.size == 0:
        return BeatResult(
            fps=int(fps),
            beat_times=[],
            beat_numbers=[],
            beat_confidences=[],
            downbeat_times=[],
            downbeat_confidences=[],
        )

    beat_times = beats[:, 0].astype(np.float64)
    beat_numbers = beats[:, 1].astype(np.int64)
    beat_confidences = derive_confidences(activations, beat_times, beat_numbers, int(fps))
    downbeat_mask = beat_numbers == 1

    return BeatResult(
        fps=int(fps),
        beat_times=beat_times.tolist(),
        beat_numbers=beat_numbers.tolist(),
        beat_confidences=beat_confidences.astype(np.float64).tolist(),
        downbeat_times=beat_times[downbeat_mask].tolist(),
        downbeat_confidences=beat_confidences[downbeat_mask].astype(np.float64).tolist(),
    )
