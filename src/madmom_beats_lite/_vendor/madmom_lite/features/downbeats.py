# encoding: utf-8
"""Minimal downbeat tracking components used by madmom-beats-lite."""

from __future__ import absolute_import, division, print_function

import numpy as np

from .beats import threshold_activations
from .beats_hmm import BarStateSpace, BarTransitionModel, RNNDownBeatTrackingObservationModel
from .._mem import MemoryLogger
from ..audio.filters import LogarithmicFilterbank
from ..audio.signal import FramedSignalProcessor, Signal, SignalProcessor
from ..audio.spectrogram import _diff_frames
from ..audio.stft import ShortTimeFourierTransformProcessor, fft_frequencies
from ..ml.hmm import HiddenMarkovModel
from ..processors import Processor, SequentialProcessor

_FRAME_SIZES = (1024, 2048, 4096)
_NUM_BANDS = (3, 6, 12)
_FPS = 100
_CHUNK_FRAMES = 4096


class _DownBeatFeatureProcessor(Processor):
    """Memory-conscious pre-processor for downbeat RNN features."""

    def __init__(self):
        self.signal = SignalProcessor(num_channels=1, sample_rate=44100, dtype=np.float32)
        self.frame_processors = [
            FramedSignalProcessor(frame_size=frame_size, fps=_FPS) for frame_size in _FRAME_SIZES
        ]
        self.stft_processors = [ShortTimeFourierTransformProcessor() for _ in _FRAME_SIZES]
        self.filterbanks = [None] * len(_FRAME_SIZES)
        self.diff_frames = [None] * len(_FRAME_SIZES)
        self.feature_bands = [None] * len(_FRAME_SIZES)
        self.mem = MemoryLogger()

    def _ensure_filterbank(self, stage, sample_rate, stft=None):
        if self.filterbanks[stage] is None:
            if stft is not None:
                bin_frequencies = stft.bin_frequencies
            else:
                bin_frequencies = fft_frequencies(_FRAME_SIZES[stage] >> 1, sample_rate)
            fb = LogarithmicFilterbank(
                bin_frequencies,
                num_bands=_NUM_BANDS[stage],
                fmin=30,
                fmax=17000,
                norm_filters=True,
            )
            self.filterbanks[stage] = np.asarray(fb, dtype=np.float32, order="C")
            self.feature_bands[stage] = int(self.filterbanks[stage].shape[1])
        if self.diff_frames[stage] is None:
            if stft is not None:
                hop_size = stft.frames.hop_size
                frame_size = stft.frames.frame_size
                window = stft.window
            else:
                hop_size = float(sample_rate) / _FPS
                frame_size = _FRAME_SIZES[stage]
                window = np.hanning
            self.diff_frames[stage] = _diff_frames(
                0.5,
                hop_size=hop_size,
                frame_size=frame_size,
                window=window,
            )
        return self.filterbanks[stage], self.diff_frames[stage], self.feature_bands[stage]

    def process(self, data, **kwargs):
        signal = self.signal(data, **kwargs)
        if signal.dtype != np.float32:
            signal = Signal(np.asarray(signal, dtype=np.float32), sample_rate=signal.sample_rate, num_channels=1)
        self.mem.log("signal", signal=signal)

        num_frames = len(self.frame_processors[0](signal))
        for stage in range(len(_FRAME_SIZES)):
            if self.filterbanks[stage] is None or self.diff_frames[stage] is None:
                probe_frames = self.frame_processors[stage](signal)
                if len(probe_frames):
                    probe_stft = self.stft_processors[stage](probe_frames[0:1])
                else:
                    probe_stft = None
                self._ensure_filterbank(stage, signal.sample_rate, probe_stft)
                del probe_frames, probe_stft

        total_bands = int(sum(self.feature_bands)) * 2
        features = np.empty((num_frames, total_bands), dtype=np.float32)
        self.mem.log("features_prealloc", features=features)
        if num_frames == 0:
            self.mem.log("features_done", features=features)
            return features

        offset = 0
        for stage, frame_size in enumerate(_FRAME_SIZES):
            frames = self.frame_processors[stage](signal)
            filterbank, _, stage_bands = self._ensure_filterbank(stage, signal.sample_rate)
            stage_log = np.empty((num_frames, stage_bands), dtype=np.float32)

            for start in range(0, num_frames, _CHUNK_FRAMES):
                stop = min(start + _CHUNK_FRAMES, num_frames)
                chunk_frames = frames[start:stop]
                stft = self.stft_processors[stage](chunk_frames)
                _, _, _ = self._ensure_filterbank(stage, signal.sample_rate, stft)

                chunk_spec = np.empty(stft.shape, dtype=np.float32)
                np.abs(stft, out=chunk_spec)

                chunk_log = np.empty((stop - start, stage_bands), dtype=np.float32)
                np.dot(chunk_spec, filterbank, out=chunk_log)
                np.add(chunk_log, 1.0, out=chunk_log)
                np.log10(chunk_log, out=chunk_log)
                stage_log[start:stop] = chunk_log

                del chunk_frames, stft, chunk_spec, chunk_log

            diff_frames = int(self.diff_frames[stage])
            stage_diff = np.empty_like(stage_log)
            if diff_frames >= num_frames:
                stage_diff.fill(0.0)
            else:
                stage_diff[:diff_frames] = 0.0
                np.subtract(stage_log[diff_frames:], stage_log[:-diff_frames], out=stage_diff[diff_frames:])
                np.maximum(stage_diff, 0.0, out=stage_diff)

            features[:, offset : offset + stage_bands] = stage_log
            features[:, offset + stage_bands : offset + 2 * stage_bands] = stage_diff
            self.mem.log(
                f"frame_size_{frame_size}",
                stage_log=stage_log,
                stage_diff=stage_diff,
                filterbank=self.filterbanks[stage],
            )
            offset += 2 * stage_bands
            del frames, stage_log, stage_diff

        self.mem.log("features_done", features=features)
        return features


class RNNDownBeatProcessor(SequentialProcessor):
    """Processor returning beat+downbeat activations at 100 FPS."""

    def __init__(self, **kwargs):
        # pylint: disable=unused-argument
        from functools import partial

        from ..ml.nn import NeuralNetworkEnsemble
        from ..models import DOWNBEATS_BLSTM

        pre_processor = _DownBeatFeatureProcessor()
        nn = NeuralNetworkEnsemble.load(DOWNBEATS_BLSTM, **kwargs)
        act = partial(np.delete, obj=0, axis=1)
        super(RNNDownBeatProcessor, self).__init__((pre_processor, nn, act))


def _process_dbn(process_tuple):
    """Extract the best Viterbi path for one HMM/activation pair."""

    return process_tuple[0].viterbi(process_tuple[1])


class DBNDownBeatTrackingProcessor(Processor):
    """DBN/HMM decoder for beat times and in-bar beat numbers."""

    MIN_BPM = 55.0
    MAX_BPM = 215.0
    NUM_TEMPI = 60
    TRANSITION_LAMBDA = 100
    OBSERVATION_LAMBDA = 16
    THRESHOLD = 0.05
    CORRECT = True

    def __init__(
        self,
        beats_per_bar,
        min_bpm=MIN_BPM,
        max_bpm=MAX_BPM,
        num_tempi=NUM_TEMPI,
        transition_lambda=TRANSITION_LAMBDA,
        observation_lambda=OBSERVATION_LAMBDA,
        threshold=THRESHOLD,
        correct=CORRECT,
        fps=None,
        **kwargs,
    ):
        beats_per_bar = np.array(beats_per_bar, ndmin=1)
        min_bpm = np.array(min_bpm, ndmin=1)
        max_bpm = np.array(max_bpm, ndmin=1)
        num_tempi = np.array(num_tempi, ndmin=1)
        transition_lambda = np.array(transition_lambda, ndmin=1)

        if len(min_bpm) != len(beats_per_bar):
            min_bpm = np.repeat(min_bpm, len(beats_per_bar))
        if len(max_bpm) != len(beats_per_bar):
            max_bpm = np.repeat(max_bpm, len(beats_per_bar))
        if len(num_tempi) != len(beats_per_bar):
            num_tempi = np.repeat(num_tempi, len(beats_per_bar))
        if len(transition_lambda) != len(beats_per_bar):
            transition_lambda = np.repeat(transition_lambda, len(beats_per_bar))
        if not (
            len(min_bpm)
            == len(max_bpm)
            == len(num_tempi)
            == len(beats_per_bar)
            == len(transition_lambda)
        ):
            raise ValueError(
                "`min_bpm`, `max_bpm`, `num_tempi`, `num_beats` and `transition_lambda` must all have the same length."
            )

        num_threads = min(len(beats_per_bar), kwargs.get("num_threads", 1))
        self.map = map
        if num_threads != 1:
            import multiprocessing as mp

            self.map = mp.Pool(num_threads).map

        min_interval = 60.0 * fps / max_bpm
        max_interval = 60.0 * fps / min_bpm

        self.hmms = []
        for b, beats in enumerate(beats_per_bar):
            st = BarStateSpace(beats, min_interval[b], max_interval[b], num_tempi[b])
            tm = BarTransitionModel(st, transition_lambda[b])
            om = RNNDownBeatTrackingObservationModel(st, observation_lambda)
            self.hmms.append(HiddenMarkovModel(tm, om))

        self.beats_per_bar = beats_per_bar
        self.threshold = threshold
        self.correct = correct
        self.fps = fps

    def process(self, activations, **kwargs):
        # pylint: disable=arguments-differ
        import itertools as it

        first = 0
        if self.threshold:
            activations, first = threshold_activations(activations, self.threshold)

        if not activations.any():
            return np.empty((0, 2))

        results = list(self.map(_process_dbn, zip(self.hmms, it.repeat(activations))))
        best = np.argmax(list(r[1] for r in results))

        path, _ = results[best]
        st = self.hmms[best].transition_model.state_space
        om = self.hmms[best].observation_model

        positions = st.state_positions[path]
        beat_numbers = positions.astype(int) + 1

        if self.correct:
            beats = np.empty(0, dtype=int)
            beat_range = om.pointers[path] >= 1
            if not beat_range.any():
                return np.empty((0, 2))

            idx = np.nonzero(np.diff(beat_range.astype(int)))[0] + 1
            if beat_range[0]:
                idx = np.r_[0, idx]
            if beat_range[-1]:
                idx = np.r_[idx, beat_range.size]

            if idx.any():
                for left, right in idx.reshape((-1, 2)):
                    peak = np.argmax(activations[left:right]) // 2 + left
                    beats = np.hstack((beats, peak))
        else:
            beats = np.nonzero(np.diff(beat_numbers))[0] + 1

        return np.vstack(((beats + first) / float(self.fps), beat_numbers[beats])).T
