from __future__ import annotations

import warnings

import numpy as np
import pytest

from madmom.ml.hmm import DiscreteObservationModel, HiddenMarkovModel, TransitionModel


def _reference_viterbi(hmm: HiddenMarkovModel, observations: np.ndarray) -> tuple[np.ndarray, float]:
    """Textbook Viterbi over all states, with one back-pointer per state and frame."""
    tm, om = hmm.transition_model, hmm.observation_model
    densities = om.log_densities(observations)
    previous = np.log(hmm.initial_distribution)
    pointers = np.zeros((len(observations), tm.num_states), dtype=np.uint32)
    current = previous
    for frame in range(len(observations)):
        current = np.full(tm.num_states, -np.inf)
        for state in range(tm.num_states):
            density = densities[frame, om.pointers[state]]
            for pointer in range(tm.pointers[state], tm.pointers[state + 1]):
                prev_state = tm.states[pointer]
                probability = previous[prev_state] + tm.log_probabilities[pointer] + density
                if probability > current[state]:
                    current[state] = probability
                    pointers[frame, state] = prev_state
        previous = current
    state = int(np.argmax(current))
    log_probability = float(current[state])
    if np.isinf(log_probability):
        return np.empty(0, dtype=np.uint32), log_probability
    path = np.empty(len(observations), dtype=np.uint32)
    for frame in range(len(observations) - 1, -1, -1):
        path[frame] = state
        state = pointers[frame, state]
    return path, log_probability


class _FixedDensities(DiscreteObservationModel):
    """Observation model returning given log densities, which may hold NaN or inf."""

    def __init__(self, om_pointers: np.ndarray) -> None:
        self.pointers = om_pointers.astype(np.uint32)

    def log_densities(self, observations: np.ndarray) -> np.ndarray:
        return observations


def _random_hmm(rng: np.random.Generator, num_states: int, num_densities: int) -> HiddenMarkovModel:
    # a chain of states as in the bar state spaces, with some extra transitions and tied values
    states, prev_states, probabilities = [], [], []
    for state in range(num_states):
        states.append(state)
        prev_states.append((state - 1) % num_states)
        probabilities.append(1.0)
    for _ in range(num_states // 3):
        state, prev_state = rng.integers(num_states, size=2)
        states.append(int(state))
        prev_states.append(int(prev_state))
        probabilities.append(1.0)
    states_a = np.array(states)
    prev_a = np.array(prev_states)
    probabilities_a = np.array(probabilities) / np.bincount(prev_a, minlength=num_states)[prev_a]
    tm = TransitionModel(*TransitionModel.make_sparse(states_a, prev_a, probabilities_a))
    om = _FixedDensities(np.repeat(rng.integers(num_densities, size=num_states // 4 + 1), 4)[:num_states])
    return HiddenMarkovModel(tm, om)


@pytest.mark.parametrize("seed", range(30))
def test_viterbi_matches_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    num_densities = 3
    hmm = _random_hmm(rng, int(rng.integers(2, 40)), num_densities)
    # coarse values produce ties between paths
    observations = np.log(rng.integers(1, 4, size=(int(rng.integers(1, 30)), num_densities)) / 4.0)
    if seed % 3 == 1:
        observations[rng.random(observations.shape) < 0.1] = -np.inf
    if seed % 3 == 2:
        observations[rng.random(observations.shape) < 0.05] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        path, log_probability = hmm.viterbi(observations)
    expected_path, expected_log_probability = _reference_viterbi(hmm, observations)
    assert path.tobytes() == expected_path.tobytes()
    assert np.float64(log_probability).tobytes() == np.float64(expected_log_probability).tobytes()
