# encoding: utf-8
"""Minimal neural network layers required by bundled downbeat models."""

from __future__ import absolute_import, division, print_function

import numpy as np

from .activations import sigmoid, tanh

NN_DTYPE = np.float32


class Layer(object):
    """Generic callable network layer."""

    def __call__(self, *args, **kwargs):
        return self.activate(*args, **kwargs)

    def activate(self, data):
        raise NotImplementedError("must be implemented by subclass.")

    def reset(self):
        return None


class FeedForwardLayer(Layer):
    """Feed-forward network layer."""

    def __init__(self, weights, bias, activation_fn=None):
        self.weights = np.asarray(weights, dtype=NN_DTYPE)
        self.bias = np.asarray(bias, dtype=NN_DTYPE).reshape(-1)
        self.activation_fn = activation_fn

    def activate(self, data, **kwargs):
        out = np.dot(data, self.weights) + self.bias
        if self.activation_fn is not None:
            self.activation_fn(out, out=out)
        return out


class RecurrentLayer(FeedForwardLayer):
    """Recurrent network layer."""

    def __init__(self, weights, bias, recurrent_weights, activation_fn=tanh, init=None):
        super(RecurrentLayer, self).__init__(weights, bias, activation_fn)
        self.recurrent_weights = np.asarray(recurrent_weights, dtype=NN_DTYPE)
        if init is None:
            init = np.zeros(self.bias.size, dtype=NN_DTYPE)
        self.init = init
        self._prev = self.init

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_prev", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "init"):
            self.init = np.zeros(self.bias.size, dtype=NN_DTYPE)
        self._prev = self.init

    def reset(self, init=None):
        self._prev = init if init is not None else self.init

    def activate(self, data, reset=True):
        if reset:
            self.reset()
        out = np.dot(data, self.weights) + self.bias
        for i in range(len(data)):
            out[i] += np.dot(self._prev, self.recurrent_weights)
            if self.activation_fn is not None:
                out[i] = self.activation_fn(out[i])
            self._prev = out[i]
        return out


class BidirectionalLayer(Layer):
    """Bidirectional wrapper around two recurrent layers."""

    def __init__(self, fwd_layer, bwd_layer):
        self.fwd_layer = fwd_layer
        self.bwd_layer = bwd_layer

    def activate(self, data, **kwargs):
        fwd = self.fwd_layer(data, **kwargs)
        bwd = self.bwd_layer(data[::-1], **kwargs)
        return np.hstack((fwd, bwd[::-1]))


class Gate(RecurrentLayer):
    """LSTM gate."""

    def __init__(self, weights, bias, recurrent_weights, peephole_weights=None, activation_fn=sigmoid):
        super(Gate, self).__init__(weights, bias, recurrent_weights, activation_fn=activation_fn)
        if peephole_weights is not None:
            peephole_weights = np.asarray(peephole_weights, dtype=NN_DTYPE).reshape(-1)
        self.peephole_weights = peephole_weights

    def activate(self, data, prev, state=None):
        out = np.dot(data, self.weights) + self.bias
        if self.peephole_weights is not None:
            out += state * self.peephole_weights
        out += np.dot(prev, self.recurrent_weights)
        return self.activation_fn(out)


class Cell(Gate):
    """LSTM cell (gate without peephole connections)."""

    def __init__(self, weights, bias, recurrent_weights, activation_fn=tanh):
        super(Cell, self).__init__(weights, bias, recurrent_weights, activation_fn=activation_fn)


class LSTMLayer(RecurrentLayer):
    """Recurrent layer with Long Short-Term Memory units."""

    def __init__(self, input_gate, forget_gate, cell, output_gate, activation_fn=tanh, init=None, cell_init=None):
        self.input_gate = input_gate
        self.forget_gate = forget_gate
        self.cell = cell
        self.output_gate = output_gate
        self.activation_fn = activation_fn

        if init is None:
            init = np.zeros(self.cell.bias.size, dtype=NN_DTYPE)
        self.init = init
        self._prev = self.init

        if cell_init is None:
            cell_init = np.zeros(self.cell.bias.size, dtype=NN_DTYPE)
        self.cell_init = cell_init
        self._state = self.cell_init

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_prev", None)
        state.pop("_state", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "init"):
            self.init = np.zeros(self.cell.bias.size, dtype=NN_DTYPE)
        if not hasattr(self, "cell_init"):
            self.cell_init = np.zeros(self.cell.bias.size, dtype=NN_DTYPE)
        self._prev = self.init
        self._state = self.cell_init

    def reset(self, init=None, cell_init=None):
        self._prev = init if init is not None else self.init
        self._state = cell_init if cell_init is not None else self.cell_init

    def activate(self, data, reset=True):
        if reset:
            self.reset()

        size = len(data)
        out = np.zeros((size, self.cell.bias.size), dtype=NN_DTYPE)
        for i in range(size):
            data_ = data[i]
            ig = self.input_gate.activate(data_, self._prev, self._state)
            fg = self.forget_gate.activate(data_, self._prev, self._state)
            cell = self.cell.activate(data_, self._prev)
            self._state = cell * ig + self._state * fg
            og = self.output_gate.activate(data_, self._prev, self._state)
            out[i] = self.activation_fn(self._state) * og
            self._prev = out[i]
        return out
