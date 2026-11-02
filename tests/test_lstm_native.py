from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("madmom.ml.nn._lstm")

from madmom.ml.nn import NeuralNetwork
from madmom.ml.nn.layers import BidirectionalLayer, LSTMLayer
from madmom.models import DOWNBEATS_BLSTM


def _lstm_layers() -> list[tuple[LSTMLayer, int]]:
    layers = []
    for layer in NeuralNetwork.load(DOWNBEATS_BLSTM[0]).layers:
        if isinstance(layer, BidirectionalLayer):
            num_inputs = layer.fwd_layer.input_gate.weights.shape[0]
            layers += [(layer.fwd_layer, num_inputs), (layer.bwd_layer, num_inputs)]
    return layers


def _run_both(layer: LSTMLayer, data: np.ndarray) -> None:
    assert layer._supports_native(data)
    layer.reset()
    python = layer._activate_python(data)
    python_prev, python_state = layer._prev.copy(), layer._state.copy()
    layer.reset()
    native = layer.activate(data)
    assert native.dtype == python.dtype
    assert native.tobytes() == python.tobytes()
    assert layer._prev.tobytes() == python_prev.tobytes()
    assert layer._state.tobytes() == python_state.tobytes()


@pytest.mark.parametrize("num_frames", [0, 1, 2, 7, 500])
def test_native_lstm_is_bit_identical_to_python(num_frames: int) -> None:
    rng = np.random.default_rng(num_frames)
    for layer, num_inputs in _lstm_layers():
        data = rng.standard_normal((num_frames, num_inputs), dtype=np.float32)
        _run_both(layer, data)
        _run_both(layer, data[::-1])


def test_native_lstm_continues_from_layer_state() -> None:
    layer, num_inputs = _lstm_layers()[0]
    data = np.random.default_rng(0).standard_normal((20, num_inputs), dtype=np.float32)
    whole = layer.activate(data)
    first = layer.activate(data[:8])
    second = layer.activate(data[8:], reset=False)
    assert np.concatenate((first, second)).tobytes() == whole.tobytes()
    assert not layer.init.any() and not layer.cell_init.any()


def test_python_implementation_handles_other_dtypes() -> None:
    layer, num_inputs = _lstm_layers()[0]
    data = np.random.default_rng(1).standard_normal((5, num_inputs), dtype=np.float64)
    assert not layer._supports_native(data)
    assert layer.activate(data).shape == (5, layer.cell.bias.size)
