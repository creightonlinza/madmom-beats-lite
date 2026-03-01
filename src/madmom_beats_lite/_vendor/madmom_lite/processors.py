# encoding: utf-8
"""Minimal processor primitives required by madmom-beats-lite."""

from __future__ import absolute_import, division, print_function

import itertools as it
import multiprocessing as mp
import pickle
import warnings
from collections.abc import MutableSequence
from importlib import import_module

import numpy as np

from .utils import integer_types

try:
    from numpy.exceptions import VisibleDeprecationWarning as _NPVisibleDeprecationWarning
except Exception:  # pragma: no cover
    _NPVisibleDeprecationWarning = Warning


class _MadmomModuleMapUnpickler(pickle.Unpickler):
    """Remap upstream `madmom.*` modules to vendored namespace while loading."""

    _UPSTREAM_ROOT = "madmom"
    _VENDORED_ROOT = "madmom_beats_lite._vendor.madmom_lite"

    def find_class(self, module, name):
        if module == self._UPSTREAM_ROOT or module.startswith(f"{self._UPSTREAM_ROOT}."):
            module = module.replace(self._UPSTREAM_ROOT, self._VENDORED_ROOT, 1)
        imported = import_module(module)
        return getattr(imported, name)


class Processor(object):
    """Abstract base class for processing data."""

    @classmethod
    def load(cls, infile):
        from .io import open_file

        with open_file(infile, "rb") as f:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
                        category=_NPVisibleDeprecationWarning,
                    )
                    obj = _MadmomModuleMapUnpickler(f, encoding="latin1").load()
            except TypeError:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
                        category=_NPVisibleDeprecationWarning,
                    )
                    obj = _MadmomModuleMapUnpickler(f).load()
        return obj

    def dump(self, outfile):
        from .io import open_file

        with open_file(outfile, "wb") as f:
            pickle.dump(self, f, protocol=2)

    def process(self, data, **kwargs):
        raise NotImplementedError("Must be implemented by subclass.")

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)


def _process(process_tuple):
    """Top-level helper (pickle-safe) for sequential/parallel execution."""

    if process_tuple[0] is None:
        return process_tuple[1]
    if isinstance(process_tuple[0], Processor):
        return process_tuple[0](*process_tuple[1:-1], **process_tuple[-1])
    return process_tuple[0](*process_tuple[1:-1])


class SequentialProcessor(MutableSequence, Processor):
    """Sequential chain of processors/functions."""

    def __init__(self, processors):
        self.processors = []
        for processor in processors:
            if isinstance(processor, (list, tuple)):
                processor = SequentialProcessor(processor)
            self.processors.append(processor)

    def __getitem__(self, index):
        return self.processors[index]

    def __setitem__(self, index, processor):
        self.processors[index] = processor

    def __delitem__(self, index):
        del self.processors[index]

    def __len__(self):
        return len(self.processors)

    def insert(self, index, processor):
        self.processors.insert(index, processor)

    def append(self, other):
        self.processors.append(other)

    def extend(self, other):
        self.processors.extend(other)

    def process(self, data, **kwargs):
        for processor in self.processors:
            data = _process((processor, data, kwargs))
        return data


class ParallelProcessor(SequentialProcessor):
    """Parallel fan-out processor over a shared input."""

    def __init__(self, processors, num_threads=None):
        super(ParallelProcessor, self).__init__(processors)
        if num_threads is None:
            num_threads = 1
        self.map = map
        if min(len(processors), max(1, num_threads)) > 1:
            self.map = mp.Pool(num_threads).map

    def process(self, data, **kwargs):
        if len(self.processors) == 1:
            return [_process((self.processors[0], data, kwargs))]
        return list(self.map(_process, zip(self.processors, it.repeat(data), it.repeat(kwargs))))


class BufferProcessor(Processor):
    """Rolling frame-context buffer."""

    def __init__(self, buffer_size=None, init=None, init_value=0):
        if buffer_size is None and init is not None:
            buffer_size = init.shape
        elif isinstance(buffer_size, integer_types):
            buffer_size = (buffer_size,)
        if buffer_size is not None and init is None:
            init = np.ones(buffer_size) * init_value

        self.buffer_size = buffer_size
        self.init = init
        self.data = init

    @property
    def buffer_length(self):
        return self.buffer_size[0]

    def reset(self, init=None):
        self.data = init if init is not None else self.init

    def process(self, data, **kwargs):
        ndmin = len(self.buffer_size)
        if data.ndim < ndmin:
            data = np.array(data, ndmin=ndmin)

        data_length = len(data)
        if data_length >= self.buffer_length:
            self.data = data[-self.buffer_length :]
        else:
            self.data = np.roll(self.data, -data_length, axis=0)
            self.data[-data_length:] = data
        return self.data

    buffer = process

    def __getitem__(self, index):
        return self.data[index]
