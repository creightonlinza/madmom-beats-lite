"""Minimal I/O helpers used by vendored processors."""

from __future__ import annotations

import contextlib
import io as _io

from ..utils import string_types


@contextlib.contextmanager
def open_file(filename, mode="r"):
    if isinstance(filename, string_types):
        f = fid = _io.open(filename, mode)
    else:
        f = filename
        fid = None
    try:
        yield f
    finally:
        if fid:
            fid.close()
