# encoding: utf-8
"""Minimal neural network runtime for bundled downbeat models."""

from __future__ import absolute_import, division, print_function

import numpy as np

from ...processors import ParallelProcessor, Processor, SequentialProcessor


def average_predictions(predictions):
    """Average predictions across ensemble members."""

    if len(predictions) == 1:
        return predictions[0]

    def avg(pred):
        return sum(pred) / len(pred)

    if isinstance(predictions[0], tuple):
        avg_pred = []
        for pred in list(zip(*predictions)):
            avg_pred.append(avg(pred))
        return tuple(avg_pred)
    return avg(predictions)


class NeuralNetwork(Processor):
    """Sequential network over vendored layer objects."""

    def __init__(self, layers):
        self.layers = layers

    def process(self, data, reset=True, **kwargs):
        if isinstance(data, np.ndarray) and data.ndim < 2:
            data = np.array(data, subok=True, copy=False, ndmin=2)

        for layer in self.layers:
            data = layer(data, reset=reset)

        try:
            return data.squeeze()
        except AttributeError:
            return tuple([d.squeeze() for d in data])

    def reset(self):
        for layer in self.layers:
            layer.reset()


class NeuralNetworkEnsemble(SequentialProcessor):
    """Parallel ensemble of multiple neural networks."""

    def __init__(self, networks, ensemble_fn=average_predictions, num_threads=None, **kwargs):
        networks_processor = ParallelProcessor(networks, num_threads=num_threads)
        super(NeuralNetworkEnsemble, self).__init__((networks_processor, ensemble_fn))

    @classmethod
    def load(cls, nn_files, **kwargs):
        networks = [NeuralNetwork.load(f) for f in nn_files]
        return cls(networks, **kwargs)
