# encoding: utf-8
"""Minimal Short-Time Fourier Transform utilities for downbeat pipeline."""

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.fftpack as fftpack

try:
    from pyfftw.builders import rfft as rfft_builder
except ImportError:

    def rfft_builder(*args, **kwargs):
        return None


from ..processors import Processor
from .signal import FramedSignal

STFT_DTYPE = np.complex64


def fft_frequencies(num_fft_bins, sample_rate):
    """Return frequencies of FFT bins."""

    return np.fft.fftfreq(num_fft_bins * 2, 1.0 / sample_rate)[:num_fft_bins]


def stft(frames, window, fft_size=None, circular_shift=False, include_nyquist=False, fftw=None):
    """Compute complex STFT of framed signal."""

    if frames.ndim != 2:
        raise ValueError("frames must be a 2D array or iterable, got %s with shape %s." % (type(frames), frames.shape))

    num_frames, frame_size = frames.shape

    if fft_size is None:
        fft_size = frame_size

    num_fft_bins = fft_size >> 1
    if include_nyquist:
        num_fft_bins += 1

    if circular_shift:
        fft_shift = frame_size >> 1

    data = np.empty((num_frames, num_fft_bins), STFT_DTYPE)

    for f, frame in enumerate(frames):
        if circular_shift:
            if window is not None:
                signal = np.multiply(frame, window)
            else:
                signal = frame
            fft_signal = np.zeros(fft_size)
            fft_signal[:fft_shift] = signal[fft_shift:]
            fft_signal[-fft_shift:] = signal[:fft_shift]
        else:
            if window is not None:
                fft_signal = np.multiply(frame, window)
            else:
                fft_signal = frame

        if fftw:
            data[f] = fftw(fft_signal)[:num_fft_bins]
        else:
            data[f] = fftpack.fft(fft_signal, fft_size, axis=0)[:num_fft_bins]

    return data


class _PropertyMixin(object):
    @property
    def num_frames(self):
        return len(self)

    @property
    def num_bins(self):
        return int(self.shape[1])


class ShortTimeFourierTransform(_PropertyMixin, np.ndarray):
    """Complex STFT ndarray wrapper."""

    def __init__(
        self,
        frames,
        window=np.hanning,
        fft_size=None,
        circular_shift=False,
        include_nyquist=False,
        fft_window=None,
        fftw=None,
        **kwargs,
    ):
        pass

    def __new__(
        cls,
        frames,
        window=np.hanning,
        fft_size=None,
        circular_shift=False,
        include_nyquist=False,
        fft_window=None,
        fftw=None,
        **kwargs,
    ):
        if isinstance(frames, ShortTimeFourierTransform):
            frames = frames.frames

        if not isinstance(frames, FramedSignal):
            frames = FramedSignal(frames, **kwargs)

        frame_size = frames.shape[1]

        if fft_window is None:
            if hasattr(window, "__call__"):
                window = window(frame_size)
            try:
                max_range = float(np.iinfo(frames.signal.dtype).max)
                try:
                    fft_window = window / max_range
                except TypeError:
                    fft_window = np.ones(frame_size) / max_range
            except ValueError:
                fft_window = window

        try:
            fftw = rfft_builder(fft_window, fft_size, axis=0)
        except AttributeError:
            pass

        data = stft(
            frames,
            fft_window,
            fft_size=fft_size,
            circular_shift=circular_shift,
            include_nyquist=include_nyquist,
            fftw=fftw,
        )

        obj = np.asarray(data).view(cls)
        obj.frames = frames
        obj.window = window
        obj.fft_window = fft_window
        obj.fftw = fftw
        obj.fft_size = fft_size if fft_size else frame_size
        obj.circular_shift = circular_shift
        obj.include_nyquist = include_nyquist
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.frames = getattr(obj, "frames", None)
        self.window = getattr(obj, "window", np.hanning)
        self.fft_window = getattr(obj, "fft_window", None)
        self.fftw = getattr(obj, "fftw", None)
        self.fft_size = getattr(obj, "fft_size", None)
        self.circular_shift = getattr(obj, "circular_shift", False)
        self.include_nyquist = getattr(obj, "include_nyquist", False)

    @property
    def bin_frequencies(self):
        return fft_frequencies(self.num_bins, self.frames.signal.sample_rate)


STFT = ShortTimeFourierTransform


class ShortTimeFourierTransformProcessor(Processor):
    """Processor wrapper for STFT extraction."""

    def __init__(self, window=np.hanning, fft_size=None, circular_shift=False, include_nyquist=False, **kwargs):
        self.window = window
        self.fft_size = fft_size
        self.circular_shift = circular_shift
        self.include_nyquist = include_nyquist
        self.fft_window = None
        self.fftw = None

    def process(self, data, **kwargs):
        data = ShortTimeFourierTransform(
            data,
            window=self.window,
            fft_size=self.fft_size,
            circular_shift=self.circular_shift,
            include_nyquist=self.include_nyquist,
            fft_window=self.fft_window,
            fftw=self.fftw,
            **kwargs,
        )
        self.fft_window = data.fft_window
        self.fftw = data.fftw
        return data


STFTProcessor = ShortTimeFourierTransformProcessor
