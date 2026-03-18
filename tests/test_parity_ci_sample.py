from __future__ import annotations

from pathlib import Path

import pytest
from scipy.io import wavfile

from madmom_beats_lite.api import extract_beats
from madmom_beats_lite.parity import compare_payloads
from madmom_beats_lite.reference import run_reference_extraction

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_WAV = ROOT / "vendor" / "madmom" / "tests" / "data" / "audio" / "sample.wav"


def test_ci_reference_parity_on_vendor_sample() -> None:
    if not SAMPLE_WAV.exists():
        pytest.skip("vendor sample audio is missing")

    sample_rate, audio = wavfile.read(SAMPLE_WAV)

    golden = run_reference_extraction(audio, int(sample_rate)).to_dict()
    candidate = extract_beats(audio, int(sample_rate)).to_dict()

    errors = compare_payloads(golden, candidate, float_tolerance=0.0)
    assert not errors, "\n".join(errors)
