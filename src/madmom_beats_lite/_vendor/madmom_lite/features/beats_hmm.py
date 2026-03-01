# encoding: utf-8
"""Minimal HMM state/transition/observation models for downbeat tracking."""

from __future__ import absolute_import, division, print_function

import numpy as np

from ..ml.hmm import ObservationModel, TransitionModel


class BeatStateSpace(object):
    """Beat state space for discretized tempo intervals."""

    def __init__(self, min_interval, max_interval, num_intervals=None):
        intervals = np.arange(np.round(min_interval), np.round(max_interval) + 1)
        if num_intervals is not None and num_intervals < len(intervals):
            num_log_intervals = num_intervals
            intervals = []
            while len(intervals) < num_intervals:
                intervals = np.logspace(
                    np.log2(min_interval),
                    np.log2(max_interval),
                    num_log_intervals,
                    base=2,
                )
                intervals = np.unique(np.round(intervals))
                num_log_intervals += 1

        self.intervals = np.ascontiguousarray(intervals, dtype=int)
        self.num_states = int(np.sum(intervals))
        self.num_intervals = len(intervals)

        first_states = np.cumsum(np.r_[0, self.intervals[:-1]])
        self.first_states = first_states.astype(int)
        self.last_states = np.cumsum(self.intervals) - 1

        self.state_positions = np.empty(self.num_states)
        self.state_intervals = np.empty(self.num_states, dtype=int)

        idx = 0
        for interval in self.intervals:
            self.state_positions[idx : idx + interval] = np.linspace(
                0, 1, interval, endpoint=False
            )
            self.state_intervals[idx : idx + interval] = interval
            idx += interval


class BarStateSpace(object):
    """Bar state space by stacking beat state spaces."""

    def __init__(self, num_beats, min_interval, max_interval, num_intervals=None):
        self.num_beats = int(num_beats)
        self.state_positions = np.empty(0)
        self.state_intervals = np.empty(0, dtype=int)
        self.num_states = 0
        self.first_states = []
        self.last_states = []

        bss = BeatStateSpace(min_interval, max_interval, num_intervals)
        for b in range(self.num_beats):
            self.state_positions = np.hstack((self.state_positions, bss.state_positions + b))
            self.state_intervals = np.hstack((self.state_intervals, bss.state_intervals))
            self.first_states.append(bss.first_states + self.num_states)
            self.last_states.append(bss.last_states + self.num_states)
            self.num_states += bss.num_states


def exponential_transition(
    from_intervals,
    to_intervals,
    transition_lambda,
    threshold=np.spacing(1),
    norm=True,
):
    """Exponential tempo transition distribution."""

    if transition_lambda is None:
        return np.diag(np.diag(np.ones((len(from_intervals), len(to_intervals)))))

    ratio = to_intervals.astype(float) / from_intervals.astype(float)[:, np.newaxis]
    prob = np.exp(-transition_lambda * abs(ratio - 1.0))
    prob[prob <= threshold] = 0
    if norm:
        prob /= np.sum(prob, axis=1)[:, np.newaxis]
    return prob


class BarTransitionModel(TransitionModel):
    """Tempo transition model for bar-level state spaces."""

    def __init__(self, state_space, transition_lambda):
        if not isinstance(transition_lambda, list):
            transition_lambda = [transition_lambda] * state_space.num_beats
        if state_space.num_beats != len(transition_lambda):
            raise ValueError("length of `transition_lambda` must be equal to `num_beats` of `state_space`.")

        self.state_space = state_space
        self.transition_lambda = transition_lambda

        states = np.arange(state_space.num_states, dtype=np.uint32)
        states = np.setdiff1d(states, state_space.first_states)
        prev_states = states - 1
        probabilities = np.ones_like(states, dtype=float)

        for beat in range(state_space.num_beats):
            to_states = state_space.first_states[beat]
            from_states = state_space.last_states[beat - 1]
            from_int = state_space.state_intervals[from_states]
            to_int = state_space.state_intervals[to_states]
            prob = exponential_transition(from_int, to_int, transition_lambda[beat])
            from_prob, to_prob = np.nonzero(prob)
            states = np.hstack((states, to_states[to_prob]))
            prev_states = np.hstack((prev_states, from_states[from_prob]))
            probabilities = np.hstack((probabilities, prob[prob != 0]))

        transitions = self.make_sparse(states, prev_states, probabilities)
        super(BarTransitionModel, self).__init__(*transitions)


class RNNDownBeatTrackingObservationModel(ObservationModel):
    """Observation model for joint beat/downbeat activations."""

    def __init__(self, state_space, observation_lambda):
        self.observation_lambda = observation_lambda
        pointers = np.zeros(state_space.num_states, dtype=np.uint32)
        border = 1.0 / observation_lambda
        pointers[state_space.state_positions % 1 < border] = 1
        pointers[state_space.state_positions < border] = 2
        super(RNNDownBeatTrackingObservationModel, self).__init__(pointers)

    def log_densities(self, observations):
        log_densities = np.empty((len(observations), 3), dtype=float)
        log_densities[:, 0] = np.log(
            (1.0 - np.sum(observations, axis=1)) / (self.observation_lambda - 1)
        )
        log_densities[:, 1] = np.log(observations[:, 0])
        log_densities[:, 2] = np.log(observations[:, 1])
        return log_densities
