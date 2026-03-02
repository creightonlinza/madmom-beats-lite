from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
import threading
from typing import Callable

import numpy as np

from ._vendor.madmom_lite.features.downbeats import (
    DBNDownBeatTrackingProcessor,
    RNNDownBeatProcessor,
)
from ._vendor.madmom_lite._mem import MemoryLogger

FPS = 100
TARGET_SAMPLE_RATE = 44100
ProgressCallback = Callable[[int], None]


@dataclass
class _ProgressState:
    last_percent: int = -1


@dataclass
class _ProcessorBundle:
    rnn: RNNDownBeatProcessor
    dbn: DBNDownBeatTrackingProcessor
    lock: threading.Lock


def _progress_callback(progress: bool | Callable[[int], None]) -> ProgressCallback:
    if progress is False:
        return lambda _percent: None
    if progress is True:
        return lambda percent: print(f"progress: {percent}")
    if callable(progress):
        return progress
    raise ValueError("`progress` must be False, True, or a callable accepting int percent.")


def _emit_progress(callback: ProgressCallback, state: _ProgressState, percent: int) -> None:
    percent = int(np.clip(percent, 0, 100))
    if percent <= state.last_percent:
        return
    state.last_percent = percent
    callback(percent)


def _validate_pcm(samples: np.ndarray, sample_rate: int) -> None:
    if not isinstance(samples, np.ndarray):
        raise ValueError("`samples` must be a numpy.ndarray of mono float32/float64 PCM samples.")
    if samples.ndim != 1:
        raise ValueError("`samples` must be mono (1D array).")
    if samples.dtype not in (np.float32, np.float64):
        raise ValueError("`samples` must have dtype float32 or float64.")
    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError("`sample_rate` must be exactly 44100.")


def _resolve_num_threads() -> int:
    raw = os.environ.get("MADMOM_BEATS_LITE_NUM_THREADS")
    if raw is None:
        return 1
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("MADMOM_BEATS_LITE_NUM_THREADS must be an integer.") from exc
    if value < 1:
        raise ValueError("MADMOM_BEATS_LITE_NUM_THREADS must be >= 1.")
    return value


@lru_cache(maxsize=4)
def _processor_bundle(num_threads: int, debug: bool) -> _ProcessorBundle:
    # Reuse processor instances to avoid rebuilding multiprocessing pools on
    # every analyze call (a major overhead in repeated invocations).
    rnn = RNNDownBeatProcessor(num_threads=num_threads, debug=debug)
    dbn = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=FPS, num_threads=num_threads)
    return _ProcessorBundle(rnn=rnn, dbn=dbn, lock=threading.Lock())


def _compute_features(
    samples: np.ndarray, emit: Callable[[int], None], num_threads: int, debug: bool
) -> tuple[np.ndarray, np.ndarray]:
    mem = MemoryLogger(enabled=debug)
    emit(5)
    processors = _processor_bundle(num_threads, debug)
    emit(20)
    with processors.lock:
        activations = processors.rnn(samples)
        mem.log("rnn_activations", activations=activations)
        emit(75)
        beats = processors.dbn(activations)
        mem.log("dbn_beats", beats=beats)
    emit(90)
    return activations, beats


def _extract_confidences(
    activations: np.ndarray,
    beat_times: np.ndarray,
    beat_numbers: np.ndarray,
    fps: int,
) -> np.ndarray:
    if beat_times.size == 0:
        return np.empty(0, dtype=float)
    frame_idx = np.rint(beat_times * fps).astype(int)
    frame_idx = np.clip(frame_idx, 0, len(activations) - 1)
    beat_channel = activations[frame_idx, 0]
    downbeat_channel = activations[frame_idx, 1]
    confidences = np.where(beat_numbers == 1, downbeat_channel, beat_channel)
    return np.clip(confidences.astype(float), 0.0, 1.0)


def analyze_pcm(
    samples: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    progress: bool | Callable[[int], None] = False,
    debug: bool = False,
) -> dict:
    """Analyze mono PCM at 44.1kHz and return beat/downbeat contract data."""
    if not isinstance(debug, bool):
        raise ValueError("`debug` must be a bool.")

    callback = _progress_callback(progress)
    state = _ProgressState()
    emit = lambda percent: _emit_progress(callback, state, percent)

    emit(0)
    _validate_pcm(samples, sample_rate)
    num_threads = _resolve_num_threads()

    activations, beats = _compute_features(samples, emit, num_threads, debug)

    beat_times = beats[:, 0].astype(float) if len(beats) else np.empty(0, dtype=float)
    beat_numbers = beats[:, 1].astype(int) if len(beats) else np.empty(0, dtype=int)
    beat_confidences = _extract_confidences(activations, beat_times, beat_numbers, FPS)
    MemoryLogger(enabled=debug).log(
        "contract_arrays",
        beat_times=beat_times,
        beat_numbers=beat_numbers,
        beat_confidences=beat_confidences,
    )

    if not (len(beat_times) == len(beat_numbers) == len(beat_confidences)):
        raise RuntimeError("Output invariant violation: beat arrays must have identical lengths.")
    if len(beat_times) > 1 and not np.all(np.diff(beat_times) > 0):
        raise RuntimeError("Output invariant violation: beat_times must be strictly increasing.")
    if len(beat_confidences) and not np.all((beat_confidences >= 0) & (beat_confidences <= 1)):
        raise RuntimeError("Output invariant violation: beat_confidences must be in [0, 1].")

    emit(100)
    return {
        "fps": FPS,
        "beat_times": beat_times.tolist(),
        "beat_numbers": beat_numbers.tolist(),
        "beat_confidences": beat_confidences.tolist(),
    }
