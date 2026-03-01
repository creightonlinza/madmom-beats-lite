"""Minimal audio subpackage used by downbeat pipeline."""

from .signal import FramedSignal, FramedSignalProcessor, Signal, SignalProcessor
from .spectrogram import (
    FilteredSpectrogram,
    FilteredSpectrogramProcessor,
    LogarithmicSpectrogram,
    LogarithmicSpectrogramProcessor,
    SpectrogramDifference,
    SpectrogramDifferenceProcessor,
)
from .stft import ShortTimeFourierTransform, ShortTimeFourierTransformProcessor

__all__ = [
    "Signal",
    "SignalProcessor",
    "FramedSignal",
    "FramedSignalProcessor",
    "ShortTimeFourierTransform",
    "ShortTimeFourierTransformProcessor",
    "FilteredSpectrogram",
    "FilteredSpectrogramProcessor",
    "LogarithmicSpectrogram",
    "LogarithmicSpectrogramProcessor",
    "SpectrogramDifference",
    "SpectrogramDifferenceProcessor",
]
