"""Minimal beat helpers required by downbeat decoder."""

from __future__ import annotations

import numpy as np


def threshold_activations(activations: np.ndarray, threshold: float) -> tuple[np.ndarray, int]:
    """Threshold activations to the main active segment (upstream logic)."""
    first = last = 0
    idx = np.nonzero(activations >= threshold)[0]
    if idx.any():
        first = max(first, np.min(idx))
        last = min(len(activations), np.max(idx) + 1)
    return activations[first:last], first
