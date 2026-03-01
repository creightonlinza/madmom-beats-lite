# encoding: utf-8
"""Minimal activation functions required by bundled downbeat models."""

from __future__ import absolute_import, division, print_function

import numpy as np


def tanh(x, out=None):
    return np.tanh(x, out)


try:
    from scipy.special import expit as _sigmoid
except ImportError:
    def _sigmoid(x, out=None):
        if out is None:
            out = np.asarray(0.5 * x)
        else:
            if out is not x:
                out[:] = x
            out *= 0.5
        np.tanh(out, out=out)
        out += 1
        out *= 0.5
        return out


def sigmoid(x, out=None):
    return _sigmoid(x, out)


def softmax(x, out=None):
    tmp = np.amax(x, axis=1, keepdims=True)
    if out is None:
        out = np.exp(x - tmp)
    else:
        np.exp(x - tmp, out=out)
    np.sum(out, axis=1, keepdims=True, out=tmp)
    out /= tmp
    return out
