# encoding: utf-8
"""Minimal filter/filterbank functionality required by downbeat pipeline."""

from __future__ import absolute_import, division, print_function

import numpy as np

FILTER_DTYPE = np.float32
A4 = 440.0


def log_frequencies(bands_per_octave, fmin, fmax, fref=A4):
    """Return frequencies aligned on a logarithmic frequency scale."""

    left = np.floor(np.log2(float(fmin) / fref) * bands_per_octave)
    right = np.ceil(np.log2(float(fmax) / fref) * bands_per_octave)
    frequencies = fref * 2.0 ** (np.arange(left, right) / float(bands_per_octave))
    frequencies = frequencies[np.searchsorted(frequencies, fmin) :]
    frequencies = frequencies[: np.searchsorted(frequencies, fmax, "right")]
    return frequencies


def frequencies2bins(frequencies, bin_frequencies, unique_bins=False):
    """Map frequencies to closest FFT-bin indices."""

    frequencies = np.asarray(frequencies)
    bin_frequencies = np.asarray(bin_frequencies)

    indices = bin_frequencies.searchsorted(frequencies)
    indices = np.clip(indices, 1, len(bin_frequencies) - 1)
    left = bin_frequencies[indices - 1]
    right = bin_frequencies[indices]
    indices -= frequencies - left < right - frequencies

    if unique_bins:
        indices = np.unique(indices)
    return indices


def bins2frequencies(bins, bin_frequencies):
    """Map bin indices to frequencies."""

    return np.asarray(bin_frequencies, dtype=float)[np.asarray(bins)]


class Filter(np.ndarray):
    """Generic 1D filter with start/stop metadata."""

    def __init__(self, data, start=0, norm=False):
        pass

    def __new__(cls, data, start=0, norm=False):
        if isinstance(data, np.ndarray):
            obj = np.asarray(data, dtype=FILTER_DTYPE).view(cls)
        else:
            raise TypeError("wrong input data for Filter, must be np.ndarray")
        if obj.ndim != 1:
            raise NotImplementedError("please add multi-dimension support")
        if norm:
            obj /= np.sum(obj)
        obj.start = int(start)
        obj.stop = int(start + len(data))
        return obj

    @classmethod
    def band_bins(cls, bins, **kwargs):
        raise NotImplementedError("needs to be implemented by sub-classes")

    @classmethod
    def filters(cls, bins, norm, **kwargs):
        filters = []
        for filter_args in cls.band_bins(bins, **kwargs):
            filters.append(cls(*filter_args, norm=norm))
        return filters


class TriangularFilter(Filter):
    """Triangular filter."""

    def __init__(self, start, center, stop, norm=False):
        pass

    def __new__(cls, start, center, stop, norm=False):
        if not start <= center < stop:
            raise ValueError("`center` must be between `start` and `stop`")

        center = int(center)
        start = int(start)
        stop = int(stop)

        center -= start
        stop -= start

        data = np.zeros(stop)
        data[:center] = np.linspace(0, 1, center, endpoint=False)
        data[center:] = np.linspace(1, 0, stop - center, endpoint=False)

        obj = Filter.__new__(cls, data, start, norm)
        obj.center = start + center
        return obj

    @classmethod
    def band_bins(cls, bins, overlap=True):
        if len(bins) < 3:
            raise ValueError("not enough bins to create a TriangularFilter")

        index = 0
        while index + 3 <= len(bins):
            start, center, stop = bins[index : index + 3]
            if not overlap:
                start = int(np.floor((center + start) / 2.0))
                stop = int(np.ceil((center + stop) / 2.0))
            if stop - start < 2:
                center = start
                stop = start + 1
            yield start, center, stop
            index += 1


FMIN = 30.0
FMAX = 17000.0
NUM_BANDS = 12
NORM_FILTERS = True
UNIQUE_FILTERS = True


class Filterbank(np.ndarray):
    """Generic filterbank matrix with bin frequency metadata."""

    def __init__(self, data, bin_frequencies):
        pass

    def __new__(cls, data, bin_frequencies):
        if isinstance(data, np.ndarray) and data.ndim == 2:
            obj = np.asarray(data, dtype=FILTER_DTYPE).view(cls)
        else:
            raise TypeError("wrong input data for Filterbank, must be a 2D np.ndarray")

        if len(bin_frequencies) != obj.shape[0]:
            raise ValueError("`bin_frequencies` must have the same length as the first dimension of `data`.")
        obj.bin_frequencies = np.asarray(bin_frequencies, dtype=float)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.bin_frequencies = getattr(obj, "bin_frequencies", None)

    @classmethod
    def _put_filter(cls, filt, band):
        if not isinstance(filt, Filter):
            raise ValueError("unable to determine start position of Filter")

        start = filt.start
        stop = start + len(filt)

        if start < 0:
            filt = filt[-start:]
            start = 0
        if stop > len(band):
            filt = filt[: -(stop - len(band))]
            stop = len(band)

        filter_position = band[start:stop]
        np.maximum(filt, filter_position, out=filter_position)

    @classmethod
    def from_filters(cls, filters, bin_frequencies):
        fb = np.zeros((len(bin_frequencies), len(filters)))
        for band_id, band_filter in enumerate(filters):
            band = fb[:, band_id]
            if isinstance(band_filter, list):
                for filt in band_filter:
                    cls._put_filter(filt, band)
            else:
                cls._put_filter(band_filter, band)
        return Filterbank.__new__(cls, fb, bin_frequencies)

    @property
    def num_bins(self):
        return self.shape[0]

    @property
    def num_bands(self):
        return self.shape[1]

    @property
    def center_frequencies(self):
        freqs = []
        for band in range(self.num_bands):
            bins = np.nonzero(self[:, band])[0]
            min_bin = np.min(bins)
            max_bin = np.max(bins)
            if self[min_bin, band] == self[max_bin, band]:
                center = int(min_bin + (max_bin - min_bin) / 2.0)
            else:
                center = min_bin + np.argmax(self[min_bin:max_bin, band])
            freqs.append(center)
        return bins2frequencies(freqs, self.bin_frequencies)


class LogarithmicFilterbank(Filterbank):
    """Log-spaced triangular filterbank."""

    NUM_BANDS_PER_OCTAVE = 12

    def __init__(
        self,
        bin_frequencies,
        num_bands=NUM_BANDS_PER_OCTAVE,
        fmin=FMIN,
        fmax=FMAX,
        fref=A4,
        norm_filters=NORM_FILTERS,
        unique_filters=UNIQUE_FILTERS,
        bands_per_octave=True,
    ):
        pass

    def __new__(
        cls,
        bin_frequencies,
        num_bands=NUM_BANDS_PER_OCTAVE,
        fmin=FMIN,
        fmax=FMAX,
        fref=A4,
        norm_filters=NORM_FILTERS,
        unique_filters=UNIQUE_FILTERS,
        bands_per_octave=True,
    ):
        if bands_per_octave:
            num_bands_per_octave = num_bands
            frequencies = log_frequencies(num_bands, fmin, fmax, fref)
            bins = frequencies2bins(frequencies, bin_frequencies, unique_bins=unique_filters)
        else:
            raise NotImplementedError(
                "please implement `num_bands` with `bands_per_octave` set to 'False' for LogarithmicFilterbank"
            )

        filters = TriangularFilter.filters(bins, norm=norm_filters, overlap=True)
        obj = cls.from_filters(filters, bin_frequencies)
        obj.fref = fref
        obj.num_bands_per_octave = num_bands_per_octave
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.num_bands_per_octave = getattr(obj, "num_bands_per_octave", self.NUM_BANDS_PER_OCTAVE)
        self.fref = getattr(obj, "fref", A4)


LogFilterbank = LogarithmicFilterbank
