from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

import madmom
from madmom_beats_lite import ExtractionConfig, extract_beats
from madmom_beats_lite.parity import compare_payloads

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_SCRIPT = ROOT / "scripts" / "upstream_extract.py"
SAMPLE_WAV = ROOT / "vendor" / "madmom" / "tests" / "data" / "audio" / "sample.wav"
PREDECODED_DIR = ROOT / "benchmarks" / "predecoded"
# CI sets this so that a broken upstream build fails instead of skipping the comparison.
REQUIRE_UPSTREAM = os.environ.get("REQUIRE_UPSTREAM_PARITY") == "1"
LOCAL_ASSETS = (
    sorted(PREDECODED_DIR.glob("*.npz")) if os.environ.get("UPSTREAM_PARITY_ASSETS") == "1" else []
)


@pytest.fixture(scope="module", autouse=True)
def upstream_available() -> None:
    check = subprocess.run([sys.executable, str(UPSTREAM_SCRIPT)], capture_output=True, text=True)
    if check.returncode != 0:
        message = f"upstream madmom is not usable, run scripts/setup_upstream.sh:\n{check.stderr}"
        if REQUIRE_UPSTREAM:
            pytest.fail(message)
        pytest.skip(message)


def _assert_matches_upstream(
    audio: np.ndarray,
    sample_rate: int,
    tmp_path: Path,
    beats_per_bar: tuple[int, ...] = (3, 4),
) -> None:
    npz_path = tmp_path / "input.npz"
    output_path = tmp_path / "upstream.json"
    np.savez(npz_path, audio=audio, sample_rate=np.int64(sample_rate))
    subprocess.run(
        [
            sys.executable,
            str(UPSTREAM_SCRIPT),
            "--input-npz",
            str(npz_path),
            "--output",
            str(output_path),
            "--beats-per-bar",
            ",".join(str(b) for b in beats_per_bar),
        ],
        check=True,
    )
    upstream = json.loads(output_path.read_text(encoding="utf-8"))
    candidate = extract_beats(audio, sample_rate, config=ExtractionConfig(beats_per_bar=beats_per_bar)).to_dict()
    errors = compare_payloads(upstream, candidate, float_tolerance=0.0)
    assert not errors, "\n".join(errors)
    assert candidate == upstream


def _sample() -> tuple[np.ndarray, int]:
    sample_rate, audio = wavfile.read(SAMPLE_WAV)
    return audio, int(sample_rate)


def test_candidate_uses_the_bundled_runtime() -> None:
    assert ROOT / "vendor" not in Path(madmom.__file__).resolve().parents


def test_mono_integer_audio(tmp_path: Path) -> None:
    audio, sample_rate = _sample()
    _assert_matches_upstream(audio, sample_rate, tmp_path)


def test_stereo_float_audio(tmp_path: Path) -> None:
    audio, sample_rate = _sample()
    left = audio.astype(np.float32) / 32768
    _assert_matches_upstream(np.stack((left, left * 0.5), axis=1), sample_rate, tmp_path)


def test_single_bar_length(tmp_path: Path) -> None:
    audio, sample_rate = _sample()
    _assert_matches_upstream(audio, sample_rate, tmp_path, beats_per_bar=(4,))


def test_silence_has_no_beats(tmp_path: Path) -> None:
    _assert_matches_upstream(np.zeros(88200, dtype=np.float32), 44100, tmp_path)


@pytest.mark.skipif(shutil.which("ffmpeg") is None and not REQUIRE_UPSTREAM, reason="ffmpeg is missing")
def test_resampled_audio(tmp_path: Path) -> None:
    audio, _ = _sample()
    _assert_matches_upstream(audio, 48000, tmp_path)


@pytest.mark.parametrize("npz_path", LOCAL_ASSETS, ids=[path.stem for path in LOCAL_ASSETS])
def test_local_track(npz_path: Path, tmp_path: Path) -> None:
    with np.load(npz_path, allow_pickle=False) as payload:
        audio = payload["audio"]
        sample_rate = int(np.asarray(payload["sample_rate"]).reshape(()).item())
    _assert_matches_upstream(audio, sample_rate, tmp_path)
