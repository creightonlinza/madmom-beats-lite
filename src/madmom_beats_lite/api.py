from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ._shared import derive_confidences, ensure_madmom_importable
from .progress import ProgressCallback, ProgressReporter
from .types import BeatResult


@dataclass(frozen=True)
class ExtractionConfig:
    fps: int = 100
    beats_per_bar: Sequence[int] = (3, 4)


def _validate_audio(audio: np.ndarray, sample_rate: int) -> None:
    if not isinstance(audio, np.ndarray):
        raise TypeError("audio must be a numpy.ndarray")
    if audio.ndim not in (1, 2):
        raise ValueError("audio must be a 1D mono or 2D multi-channel array")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")


def _run_rnn_downbeat_with_progress(signal: np.ndarray, reporter: ProgressReporter) -> np.ndarray:
    reporter.emit(3, "prepare", "loading madmom preprocessing and inference components")
    from madmom.audio.signal import FramedSignalProcessor, SignalProcessor
    from madmom.audio.spectrogram import (
        FilteredSpectrogramProcessor,
        LogarithmicSpectrogramProcessor,
        SpectrogramDifferenceProcessor,
    )
    from madmom.audio.stft import ShortTimeFourierTransformProcessor
    from madmom.ml.nn import NeuralNetwork, average_predictions
    from madmom.models import DOWNBEATS_BLSTM

    reporter.emit(4, "preprocess", "resampling and downmixing input signal")
    proc_signal = SignalProcessor(num_channels=1, sample_rate=44100)(signal)

    frame_sizes = [1024, 2048, 4096]
    num_bands = [3, 6, 12]
    branch_features: list[np.ndarray] = []
    branch_pcts = [5, 10, 15]

    for idx, (frame_size, bands, start_pct) in enumerate(zip(frame_sizes, num_bands, branch_pcts), start=1):
        reporter.emit(start_pct, "preprocess", f"branch {idx}/3 framing (frame_size={frame_size})")
        frames = FramedSignalProcessor(frame_size=frame_size, fps=100)(proc_signal)

        reporter.emit(start_pct + 1, "preprocess", f"branch {idx}/3 short-time Fourier transform")
        stft = ShortTimeFourierTransformProcessor()(frames)

        reporter.emit(start_pct + 2, "preprocess", f"branch {idx}/3 filtered spectrogram (bands={bands})")
        filt = FilteredSpectrogramProcessor(
            num_bands=bands,
            fmin=30,
            fmax=17000,
            norm_filters=True,
        )(stft)

        reporter.emit(start_pct + 3, "preprocess", f"branch {idx}/3 logarithmic spectrogram")
        spec = LogarithmicSpectrogramProcessor(mul=1, add=1)(filt)

        reporter.emit(start_pct + 4, "preprocess", f"branch {idx}/3 spectrogram difference")
        diff = SpectrogramDifferenceProcessor(
            diff_ratio=0.5,
            positive_diffs=True,
            stack_diffs=np.hstack,
        )(spec)
        branch_features.append(diff)

    reporter.emit(20, "preprocess", "stacking multiresolution features")
    features = np.hstack(branch_features)

    reporter.emit(21, "inference", "loading downbeat neural network ensemble")
    networks = [NeuralNetwork.load(model_path) for model_path in DOWNBEATS_BLSTM]
    num_networks = len(networks)
    reporter.emit(24, "inference", f"loaded ensemble models ({num_networks} total)")

    predictions = []
    for idx, network in enumerate(networks, start=1):
        predictions.append(network(features))
        # Reserve 25..84 for model-completion milestones.
        model_pct = 24 + (60 * idx // num_networks)
        reporter.emit(model_pct, "inference", f"completed ensemble model {idx}/{num_networks}")

    nn_out = average_predictions(predictions)
    reporter.emit(85, "inference", "averaged ensemble model activations")
    reporter.emit(86, "inference", "removing non-beat activation column")
    return np.delete(nn_out, obj=0, axis=1)


def _run_dbn_tracking_with_progress(
    activations: np.ndarray,
    cfg: "ExtractionConfig",
    reporter: ProgressReporter,
) -> np.ndarray:
    reporter.emit(88, "track", "loading DBN beat/downbeat tracker components")
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor

    reporter.emit(90, "track", "initializing DBN beat/downbeat tracker")
    tracking_processor = DBNDownBeatTrackingProcessor(
        beats_per_bar=list(cfg.beats_per_bar),
        fps=int(cfg.fps),
    )
    reporter.emit(92, "track", "initialized DBN beat/downbeat tracker")
    reporter.emit(94, "track", "decoding beat/downbeat sequence")
    beats = tracking_processor(activations)
    reporter.emit(97, "track", "decoded beat/downbeat sequence")
    return beats


def extract_beats(
    audio: np.ndarray,
    sample_rate: int,
    *,
    config: ExtractionConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> BeatResult:
    """
    Extract beats/downbeats from predecoded audio samples.

    Input contract:
    - `audio` is already decoded by the caller.
    - `sample_rate` is the original sample rate of `audio`.
    - No file decoding occurs in this path.
    """
    _validate_audio(audio, sample_rate)
    ensure_madmom_importable()
    from madmom.audio.signal import Signal

    cfg = config or ExtractionConfig()
    reporter = ProgressReporter(progress_callback)

    reporter.emit(0, "start", "starting beat/downbeat extraction")
    reporter.emit(1, "validate", "validated predecoded input audio")

    # Preserve madmom behavior by attaching the source sample-rate to Signal.
    signal = Signal(audio, sample_rate=int(sample_rate))
    reporter.emit(2, "prepare", "created madmom Signal from decoded audio")

    activations = _run_rnn_downbeat_with_progress(signal, reporter)
    beats = _run_dbn_tracking_with_progress(activations, cfg, reporter)

    if beats.size == 0:
        result = BeatResult(
            fps=int(cfg.fps),
            beat_times=[],
            beat_numbers=[],
            beat_confidences=[],
            downbeat_times=[],
            downbeat_confidences=[],
        )
        reporter.emit(100, "done", "extraction complete")
        return result

    beat_times = beats[:, 0].astype(np.float64)
    beat_numbers = beats[:, 1].astype(np.int64)
    reporter.emit(99, "postprocess", "deriving beat confidences")
    beat_confidences = derive_confidences(activations, beat_times, beat_numbers, int(cfg.fps))

    downbeat_mask = beat_numbers == 1
    result = BeatResult(
        fps=int(cfg.fps),
        beat_times=beat_times.tolist(),
        beat_numbers=beat_numbers.tolist(),
        beat_confidences=beat_confidences.astype(np.float64).tolist(),
        downbeat_times=beat_times[downbeat_mask].tolist(),
        downbeat_confidences=beat_confidences[downbeat_mask].astype(np.float64).tolist(),
    )
    reporter.emit(100, "done", "extraction complete")
    return result
