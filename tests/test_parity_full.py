from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from madmom_beats_lite.api import extract_beats
from madmom_beats_lite.parity import compare_payloads

ROOT = Path(__file__).resolve().parents[1]
FULL_IDS = sorted(p.stem.replace(".reference", "") for p in (ROOT / "tests" / "golden").glob("*_full.reference.json"))


@pytest.mark.parametrize(
    "asset_id",
    FULL_IDS,
)
def test_full_track_parity_against_golden(asset_id: str) -> None:
    npz_path = ROOT / "benchmarks" / "predecoded" / f"{asset_id}.npz"
    golden_path = ROOT / "tests" / "golden" / f"{asset_id}.reference.json"

    if not npz_path.exists() or not golden_path.exists():
        pytest.skip(f"Missing parity assets for {asset_id}; run scripts/generate_golden.sh full")

    with np.load(npz_path, allow_pickle=False) as payload:
        audio = payload["audio"]
        sample_rate = int(np.asarray(payload["sample_rate"]).reshape(()).item())

    candidate = extract_beats(audio, sample_rate).to_dict()
    golden = json.loads(golden_path.read_text(encoding="utf-8"))

    errors = compare_payloads(golden, candidate, float_tolerance=0.0)
    assert not errors, "\n".join(errors)
