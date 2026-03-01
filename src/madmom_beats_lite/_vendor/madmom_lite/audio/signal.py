# encoding: utf-8
"""Minimal signal/frame utilities required by downbeat pipeline."""

from __future__ import absolute_import, division, print_function

import numpy as np

from ..processors import Processor
from ..utils import integer_types


def adjust_gain(signal, gain):
    gain = np.power(np.sqrt(10.0), 0.1 * gain)
    if gain > 1 and np.issubdtype(signal.dtype, np.integer):
        raise ValueError("positive gain adjustments are only supported for float dtypes.")
    return np.asanyarray(signal * gain, dtype=signal.dtype)


def normalize(signal):
    scaling = float(np.max(np.abs(signal)))
    if np.issubdtype(signal.dtype, np.integer):
        if signal.dtype in (np.int16, np.int32):
            scaling /= np.iinfo(signal.dtype).max
        else:
            raise ValueError("only float and np.int16/32 dtypes supported, not %s." % signal.dtype)
    return np.asanyarray(signal / scaling, dtype=signal.dtype)


def remix(signal, num_channels, channel=None):
    if num_channels == signal.ndim or num_channels is None:
        return signal
    if num_channels == 1 and signal.ndim > 1:
        if channel is None:
            return np.mean(signal, axis=-1).astype(signal.dtype)
        return signal[:, channel]
    if num_channels > 1 and signal.ndim == 1:
        return np.tile(signal[:, np.newaxis], num_channels)
    raise NotImplementedError(
        "Requested %d channels, but got %d channels and channel conversion is not implemented."
        % (num_channels, signal.shape[1])
    )


def resample(signal, sample_rate, **kwargs):
    from ..io.audio import load_ffmpeg_file

    if not isinstance(signal, Signal):
        raise ValueError("only Signals can resampled, not %s" % type(signal))
    if signal.sample_rate == sample_rate:
        return signal

    dtype = kwargs.get("dtype", signal.dtype)
    num_channels = kwargs.get("num_channels", signal.num_channels)
    signal, sample_rate = load_ffmpeg_file(
        signal,
        sample_rate=sample_rate,
        num_channels=num_channels,
        dtype=dtype,
    )
    return Signal(signal, sample_rate=sample_rate)


SAMPLE_RATE = None
NUM_CHANNELS = None
CHANNEL = None
START = None
STOP = None
NORM = False
GAIN = 0.0
DTYPE = None


class Signal(np.ndarray):
    """Signal ndarray wrapper with sample-rate metadata."""

    def __init__(
        self,
        data,
        sample_rate=SAMPLE_RATE,
        num_channels=NUM_CHANNELS,
        channel=CHANNEL,
        start=START,
        stop=STOP,
        norm=NORM,
        gain=GAIN,
        dtype=DTYPE,
        **kwargs,
    ):
        pass

    def __new__(
        cls,
        data,
        sample_rate=SAMPLE_RATE,
        num_channels=NUM_CHANNELS,
        channel=CHANNEL,
        start=START,
        stop=STOP,
        norm=NORM,
        gain=GAIN,
        dtype=DTYPE,
        **kwargs,
    ):
        if not isinstance(data, np.ndarray):
            from ..io.audio import load_audio_file

            data, sample_rate = load_audio_file(
                data,
                sample_rate=sample_rate,
                num_channels=num_channels,
                start=start,
                stop=stop,
                dtype=dtype,
            )

        if not isinstance(data, Signal):
            data = np.asarray(data).view(cls)
            data.sample_rate = sample_rate

        if num_channels:
            data = remix(data, num_channels, channel)
        if norm:
            data = normalize(data)
        if gain is not None and gain != 0:
            data = adjust_gain(data, gain)
        if sample_rate != data.sample_rate:
            data = resample(data, sample_rate)

        if start is not None:
            data.start = start
            data.stop = start + float(len(data)) / sample_rate
        return data

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.sample_rate = getattr(obj, "sample_rate", None)
        self.start = getattr(obj, "start", None)
        self.stop = getattr(obj, "stop", None)

    def __reduce__(self):
        state = super(Signal, self).__reduce__()
        new_state = state[2] + (self.__dict__,)
        return state[0], state[1], new_state

    def __setstate__(self, state):
        self.__dict__.update(state[-1])
        super(Signal, self).__setstate__(state[:-1])

    @property
    def num_channels(self):
        if self.ndim == 1:
            return 1
        return np.shape(self)[1]


class SignalProcessor(Processor):
    """Processor wrapper creating `Signal` from input data."""

    def __init__(
        self,
        sample_rate=SAMPLE_RATE,
        num_channels=NUM_CHANNELS,
        start=START,
        stop=STOP,
        norm=NORM,
        gain=GAIN,
        dtype=DTYPE,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.start = start
        self.stop = stop
        self.norm = norm
        self.gain = gain
        self.dtype = dtype

    def process(self, data, **kwargs):
        args = dict(
            sample_rate=self.sample_rate,
            num_channels=self.num_channels,
            start=self.start,
            stop=self.stop,
            norm=self.norm,
            gain=self.gain,
            dtype=self.dtype,
        )
        args.update(kwargs)
        return Signal(data, **args)


def signal_frame(signal, index, frame_size, hop_size, origin=0, pad=0):
    """Return one frame of a signal with pad/repeat boundary behavior."""

    frame_size = int(frame_size)
    num_samples = len(signal)
    ref_sample = int(index * hop_size)
    start = ref_sample - frame_size // 2 - int(origin)
    stop = start + frame_size

    if start >= 0 and stop <= num_samples:
        return signal[start:stop]

    frame = np.repeat(signal[:1], frame_size, axis=0)

    left, right = 0, 0
    if start < 0:
        left = min(stop, 0) - start
        frame[:left] = np.repeat(signal[:1], left, axis=0)
        if pad != "repeat":
            frame[:left] = pad
        start = 0
    if stop > num_samples:
        right = stop - max(start, num_samples)
        frame[-right:] = np.repeat(signal[-1:], right, axis=0)
        if pad != "repeat":
            frame[-right:] = pad
        stop = num_samples

    frame[left : frame_size - right] = signal[min(start, num_samples) : max(stop, 0)]
    return frame


FRAME_SIZE = 2048
HOP_SIZE = 441.0
FPS = None
ORIGIN = 0
END_OF_SIGNAL = "normal"
NUM_FRAMES = None


class FramedSignal(object):
    """Iterable/indexable signal frame view."""

    def __init__(
        self,
        signal,
        frame_size=FRAME_SIZE,
        hop_size=HOP_SIZE,
        fps=FPS,
        origin=ORIGIN,
        end=END_OF_SIGNAL,
        num_frames=NUM_FRAMES,
        **kwargs,
    ):
        if not isinstance(signal, Signal):
            signal = Signal(signal, **kwargs)
        self.signal = signal

        if frame_size:
            self.frame_size = int(frame_size)
        if hop_size:
            self.hop_size = float(hop_size)
        if fps:
            self.hop_size = self.signal.sample_rate / float(fps)

        if origin in ("center", "offline"):
            origin = 0
        elif origin in ("left", "past", "online"):
            origin = (frame_size - 1) / 2
        elif origin in ("right", "future", "stream"):
            origin = -(frame_size / 2)
        self.origin = int(origin)

        if num_frames is None:
            if end == "extend":
                num_frames = np.floor(len(self.signal) / float(self.hop_size) + 1)
            elif end == "normal":
                num_frames = np.ceil(len(self.signal) / float(self.hop_size))
            else:
                raise ValueError("end of signal handling '%s' unknown" % end)
        self.num_frames = int(num_frames)

    def __getitem__(self, index):
        if isinstance(index, integer_types):
            if index < 0:
                index += self.num_frames
            if index < self.num_frames:
                return signal_frame(
                    self.signal,
                    index,
                    frame_size=self.frame_size,
                    hop_size=self.hop_size,
                    origin=self.origin,
                )
            raise IndexError("end of signal reached")

        if isinstance(index, slice):
            start, stop, step = index.indices(self.num_frames)
            if step != 1:
                raise ValueError("only slices with a step size of 1 supported")
            num_frames = stop - start
            origin = self.origin - self.hop_size * start
            return FramedSignal(
                self.signal,
                frame_size=self.frame_size,
                hop_size=self.hop_size,
                origin=origin,
                num_frames=num_frames,
            )

        raise TypeError("frame indices must be slices or integers")

    def __len__(self):
        return self.num_frames

    @property
    def shape(self):
        shape = (self.num_frames, self.frame_size)
        if self.signal.num_channels != 1:
            shape += (self.signal.num_channels,)
        return shape

    @property
    def ndim(self):
        return len(self.shape)


class FramedSignalProcessor(Processor):
    """Processor wrapper creating `FramedSignal` from input data."""

    def __init__(
        self,
        frame_size=FRAME_SIZE,
        hop_size=HOP_SIZE,
        fps=FPS,
        origin=ORIGIN,
        end=END_OF_SIGNAL,
        num_frames=NUM_FRAMES,
        **kwargs,
    ):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.fps = fps
        self.origin = origin
        self.end = end
        self.num_frames = num_frames

    def process(self, data, **kwargs):
        args = dict(
            frame_size=self.frame_size,
            hop_size=self.hop_size,
            fps=self.fps,
            origin=self.origin,
            end=self.end,
            num_frames=self.num_frames,
        )
        args.update(kwargs)
        if self.origin == "stream":
            data = data[-self.frame_size :]
        return FramedSignal(data, **args)
