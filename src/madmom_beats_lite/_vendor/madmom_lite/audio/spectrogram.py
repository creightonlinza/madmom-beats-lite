# encoding: utf-8
"""Minimal spectrogram processing needed by downbeat pipeline."""

from __future__ import absolute_import, division, print_function

import inspect

import numpy as np
from scipy.ndimage import maximum_filter

from .filters import (
    A4,
    FMAX,
    FMIN,
    NORM_FILTERS,
    NUM_BANDS,
    UNIQUE_FILTERS,
    Filterbank,
    LogarithmicFilterbank,
)
from ..processors import BufferProcessor, Processor


class Spectrogram(np.ndarray):
    """Magnitude spectrogram of an STFT."""

    def __init__(self, stft, **kwargs):
        pass

    def __new__(cls, stft, **kwargs):
        from .stft import ShortTimeFourierTransform

        if isinstance(stft, Spectrogram):
            data = stft
        elif isinstance(stft, ShortTimeFourierTransform):
            data = np.abs(stft)
        else:
            stft = ShortTimeFourierTransform(stft, **kwargs)
            data = np.abs(stft)

        obj = np.asarray(data).view(cls)
        obj.stft = stft
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.stft = getattr(obj, "stft", None)

    @property
    def num_frames(self):
        return len(self)

    @property
    def num_bins(self):
        return int(self.shape[1])

    @property
    def bin_frequencies(self):
        return self.stft.bin_frequencies


FILTERBANK = LogarithmicFilterbank


class FilteredSpectrogram(Spectrogram):
    """Filterbank-projected spectrogram."""

    def __init__(
        self,
        spectrogram,
        filterbank=FILTERBANK,
        num_bands=NUM_BANDS,
        fmin=FMIN,
        fmax=FMAX,
        fref=A4,
        norm_filters=NORM_FILTERS,
        unique_filters=UNIQUE_FILTERS,
        **kwargs,
    ):
        pass

    def __new__(
        cls,
        spectrogram,
        filterbank=FILTERBANK,
        num_bands=NUM_BANDS,
        fmin=FMIN,
        fmax=FMAX,
        fref=A4,
        norm_filters=NORM_FILTERS,
        unique_filters=UNIQUE_FILTERS,
        **kwargs,
    ):
        if not isinstance(spectrogram, Spectrogram):
            spectrogram = Spectrogram(spectrogram, **kwargs)

        if inspect.isclass(filterbank) and issubclass(filterbank, Filterbank):
            filterbank = filterbank(
                spectrogram.bin_frequencies,
                num_bands=num_bands,
                fmin=fmin,
                fmax=fmax,
                fref=fref,
                norm_filters=norm_filters,
                unique_filters=unique_filters,
            )
        if not isinstance(filterbank, Filterbank):
            raise TypeError("not a Filterbank type or instance: %s" % filterbank)

        data = np.dot(spectrogram, filterbank)
        obj = np.asarray(data).view(cls)
        obj.filterbank = filterbank
        obj.stft = spectrogram.stft
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.stft = getattr(obj, "stft", None)
        self.filterbank = getattr(obj, "filterbank", None)

    @property
    def bin_frequencies(self):
        return self.filterbank.center_frequencies


class FilteredSpectrogramProcessor(Processor):
    """Processor wrapper for FilteredSpectrogram."""

    def __init__(
        self,
        filterbank=FILTERBANK,
        num_bands=NUM_BANDS,
        fmin=FMIN,
        fmax=FMAX,
        fref=A4,
        norm_filters=NORM_FILTERS,
        unique_filters=UNIQUE_FILTERS,
        **kwargs,
    ):
        self.filterbank = filterbank
        self.num_bands = num_bands
        self.fmin = fmin
        self.fmax = fmax
        self.fref = fref
        self.norm_filters = norm_filters
        self.unique_filters = unique_filters

    def process(self, data, **kwargs):
        args = dict(
            filterbank=self.filterbank,
            num_bands=self.num_bands,
            fmin=self.fmin,
            fmax=self.fmax,
            fref=self.fref,
            norm_filters=self.norm_filters,
            unique_filters=self.unique_filters,
        )
        args.update(kwargs)
        data = FilteredSpectrogram(data, **args)
        self.filterbank = data.filterbank
        return data


LOG = np.log10
MUL = 1.0
ADD = 1.0


class LogarithmicSpectrogram(Spectrogram):
    """Log-scaled spectrogram."""

    def __init__(self, spectrogram, log=LOG, mul=MUL, add=ADD, **kwargs):
        pass

    def __new__(cls, spectrogram, log=LOG, mul=MUL, add=ADD, **kwargs):
        if not isinstance(spectrogram, Spectrogram):
            spectrogram = Spectrogram(spectrogram, **kwargs)
            data = spectrogram
        else:
            data = spectrogram.copy()

        if mul is not None:
            data *= mul
        if add is not None:
            data += add
        if log is not None:
            log(data, data)

        obj = np.asarray(data).view(cls)
        obj.mul = mul
        obj.add = add
        obj.stft = spectrogram.stft
        obj.spectrogram = spectrogram
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.stft = getattr(obj, "stft", None)
        self.spectrogram = getattr(obj, "spectrogram", None)
        self.mul = getattr(obj, "mul", MUL)
        self.add = getattr(obj, "add", ADD)

    @property
    def filterbank(self):
        return self.spectrogram.filterbank

    @property
    def bin_frequencies(self):
        return self.spectrogram.bin_frequencies


class LogarithmicSpectrogramProcessor(Processor):
    """Processor wrapper for LogarithmicSpectrogram."""

    def __init__(self, log=LOG, mul=MUL, add=ADD, **kwargs):
        self.log = log
        self.mul = mul
        self.add = add

    def process(self, data, **kwargs):
        args = dict(log=self.log, mul=self.mul, add=self.add)
        args.update(kwargs)
        return LogarithmicSpectrogram(data, **args)


DIFF_RATIO = 0.5
DIFF_FRAMES = None
DIFF_MAX_BINS = None
POSITIVE_DIFFS = False


def _diff_frames(diff_ratio, hop_size, frame_size, window=np.hanning):
    """Compute diff-frames count for a target overlap ratio."""

    if hasattr(window, "__call__"):
        window = window(frame_size)
    sample = np.argmax(window > float(diff_ratio) * max(window))
    diff_samples = len(window) / 2 - sample
    return int(max(1, round(diff_samples / hop_size)))


class SpectrogramDifference(Spectrogram):
    """First-order temporal spectrogram difference."""

    def __init__(
        self,
        spectrogram,
        diff_ratio=DIFF_RATIO,
        diff_frames=DIFF_FRAMES,
        diff_max_bins=DIFF_MAX_BINS,
        positive_diffs=POSITIVE_DIFFS,
        keep_dims=True,
        **kwargs,
    ):
        pass

    def __new__(
        cls,
        spectrogram,
        diff_ratio=DIFF_RATIO,
        diff_frames=DIFF_FRAMES,
        diff_max_bins=DIFF_MAX_BINS,
        positive_diffs=POSITIVE_DIFFS,
        keep_dims=True,
        **kwargs,
    ):
        if not isinstance(spectrogram, Spectrogram):
            spectrogram = Spectrogram(spectrogram, **kwargs)

        if diff_frames is None:
            diff_frames = _diff_frames(
                diff_ratio,
                hop_size=spectrogram.stft.frames.hop_size,
                frame_size=spectrogram.stft.frames.frame_size,
                window=spectrogram.stft.window,
            )

        if diff_max_bins is not None and diff_max_bins > 1:
            size = (1, int(diff_max_bins))
            diff_spec = maximum_filter(spectrogram, size=size)
        else:
            diff_spec = spectrogram

        if keep_dims:
            diff = np.zeros_like(spectrogram)
            diff[diff_frames:] = spectrogram[diff_frames:] - diff_spec[:-diff_frames]
        else:
            diff = spectrogram[diff_frames:] - diff_spec[:-diff_frames]

        if positive_diffs:
            np.maximum(diff, 0, out=diff)

        obj = np.asarray(diff).view(cls)
        obj.spectrogram = spectrogram
        obj.diff_ratio = diff_ratio
        obj.diff_frames = diff_frames
        obj.diff_max_bins = diff_max_bins
        obj.positive_diffs = positive_diffs
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.diff_ratio = getattr(obj, "diff_ratio", 0.5)
        self.diff_frames = getattr(obj, "diff_frames", None)
        self.diff_max_bins = getattr(obj, "diff_max_bins", None)
        self.positive_diffs = getattr(obj, "positive_diffs", False)

    @property
    def bin_frequencies(self):
        return self.spectrogram.bin_frequencies


class SpectrogramDifferenceProcessor(Processor):
    """Stateful processor for spectrogram differences."""

    def __init__(
        self,
        diff_ratio=DIFF_RATIO,
        diff_frames=DIFF_FRAMES,
        diff_max_bins=DIFF_MAX_BINS,
        positive_diffs=POSITIVE_DIFFS,
        stack_diffs=None,
        **kwargs,
    ):
        self.diff_ratio = diff_ratio
        self.diff_frames = diff_frames
        self.diff_max_bins = diff_max_bins
        self.positive_diffs = positive_diffs
        self.stack_diffs = stack_diffs
        self._buffer = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_buffer", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._buffer = None

    def process(self, data, reset=True, **kwargs):
        args = dict(
            diff_ratio=self.diff_ratio,
            diff_frames=self.diff_frames,
            diff_max_bins=self.diff_max_bins,
            positive_diffs=self.positive_diffs,
        )
        args.update(kwargs)

        if self.diff_frames is None:
            self.diff_frames = _diff_frames(
                args["diff_ratio"],
                frame_size=data.stft.frames.frame_size,
                hop_size=data.stft.frames.hop_size,
                window=data.stft.window,
            )

        if self._buffer is None or reset:
            init = np.empty((self.diff_frames, data.shape[1]))
            init[:] = np.inf
            data = np.insert(data, 0, init, axis=0)
            self._buffer = BufferProcessor(init=data)
        else:
            data = self._buffer(data)

        diff = SpectrogramDifference(data, keep_dims=False, **args)
        diff[np.isinf(diff)] = 0

        if self.stack_diffs is None:
            return diff
        return self.stack_diffs((diff.spectrogram[self.diff_frames :], diff))

    def reset(self):
        self._buffer = None
