from __future__ import annotations

import numpy as np
import pytest

from madmom_beats_lite import analyze_pcm
import madmom_beats_lite.analyze as analyze_mod


def test_analyze_pcm_validates_sample_rate() -> None:
    samples = np.zeros(44100, dtype=np.float32)
    with pytest.raises(ValueError, match="sample_rate"):
        analyze_pcm(samples, sample_rate=48000)


def test_analyze_pcm_validates_mono() -> None:
    samples = np.zeros((100, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="mono"):
        analyze_pcm(samples, sample_rate=44100)


def test_analyze_pcm_validates_dtype() -> None:
    samples = np.zeros(100, dtype=np.int16)
    with pytest.raises(ValueError, match="float32 or float64"):
        analyze_pcm(samples, sample_rate=44100)


def test_analyze_pcm_contract_and_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    activations = np.array(
        [
            [0.1, 0.7],
            [0.8, 0.1],
            [0.9, 0.2],
            [0.4, 0.8],
        ],
        dtype=np.float32,
    )
    beats = np.array(
        [
            [0.00, 1],
            [0.01, 2],
            [0.03, 1],
        ],
        dtype=float,
    )

    def _fake_compute(_samples, emit, _num_threads):
        emit(12)
        emit(88)
        return activations, beats

    monkeypatch.setattr(analyze_mod, "_compute_features", _fake_compute)

    seen: list[int] = []
    result = analyze_pcm(np.zeros(64, dtype=np.float32), sample_rate=44100, progress=seen.append)

    assert result["fps"] == 100
    assert result["beat_times"] == [0.0, 0.01, 0.03]
    assert result["beat_numbers"] == [1, 2, 1]

    # downbeats use activation channel 1, other beats use channel 0
    assert np.allclose(result["beat_confidences"], [0.7, 0.8, 0.8])

    assert seen[0] == 0
    assert seen[-1] == 100
    assert seen == sorted(set(seen))


def test_progress_true_prints(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_compute(_samples, emit, _num_threads):
        emit(50)
        return np.zeros((1, 2), dtype=np.float32), np.empty((0, 2))

    monkeypatch.setattr(analyze_mod, "_compute_features", _fake_compute)

    analyze_pcm(np.zeros(64, dtype=np.float32), sample_rate=44100, progress=True)
    out = capsys.readouterr().out.strip().splitlines()
    assert out[0] == "progress: 0"
    assert out[-1] == "progress: 100"


def test_resolve_num_threads_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MADMOM_BEATS_LITE_NUM_THREADS", raising=False)
    monkeypatch.setattr(analyze_mod.os, "cpu_count", lambda: 64)
    assert analyze_mod._resolve_num_threads() == 1

    monkeypatch.setattr(analyze_mod.os, "cpu_count", lambda: 4)
    assert analyze_mod._resolve_num_threads() == 1

    monkeypatch.setenv("MADMOM_BEATS_LITE_NUM_THREADS", "3")
    assert analyze_mod._resolve_num_threads() == 3
