# encoding: utf-8
"""Minimal downbeat tracking components used by madmom-beats-lite."""

from __future__ import absolute_import, division, print_function

import numpy as np

from .beats import threshold_activations
from .beats_hmm import BarStateSpace, BarTransitionModel, RNNDownBeatTrackingObservationModel
from ..ml.hmm import HiddenMarkovModel
from ..processors import ParallelProcessor, Processor, SequentialProcessor


class RNNDownBeatProcessor(SequentialProcessor):
    """Processor returning beat+downbeat activations at 100 FPS."""

    def __init__(self, **kwargs):
        # pylint: disable=unused-argument
        from functools import partial

        from ..audio.signal import FramedSignalProcessor, SignalProcessor
        from ..audio.spectrogram import (
            FilteredSpectrogramProcessor,
            LogarithmicSpectrogramProcessor,
            SpectrogramDifferenceProcessor,
        )
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..ml.nn import NeuralNetworkEnsemble
        from ..models import DOWNBEATS_BLSTM

        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        multi = ParallelProcessor([])
        frame_sizes = [1024, 2048, 4096]
        num_bands = [3, 6, 12]
        for frame_size, num_band in zip(frame_sizes, num_bands):
            frames = FramedSignalProcessor(frame_size=frame_size, fps=100)
            stft = ShortTimeFourierTransformProcessor()
            filt = FilteredSpectrogramProcessor(
                num_bands=num_band, fmin=30, fmax=17000, norm_filters=True
            )
            spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
            diff = SpectrogramDifferenceProcessor(
                diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack
            )
            multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))

        pre_processor = SequentialProcessor((sig, multi, np.hstack))
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
