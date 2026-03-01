from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from madmom_beats_lite import analyze_pcm
from tools.regen.common import FIXTURES_DIR, GOLDENS_DIR, read_wav_mono_float

MAX_BEAT_TIME_ABS = 1e-6
MAX_CONF_ABS = 1e-7


def compare_result(ours: dict, golden: dict, fixture_name: str) -> list[str]:
    errors: list[str] = []
    if ours["fps"] != golden["fps"]:
        errors.append(f"{fixture_name}: fps mismatch {ours['fps']} != {golden['fps']}")

    our_times = np.asarray(ours["beat_times"], dtype=float)
    gold_times = np.asarray(golden["beat_times"], dtype=float)
    our_nums = np.asarray(ours["beat_numbers"], dtype=int)
    gold_nums = np.asarray(golden["beat_numbers"], dtype=int)
    our_conf = np.asarray(ours["beat_confidences"], dtype=float)
    gold_conf = np.asarray(golden["beat_confidences"], dtype=float)

    if len(our_times) != len(gold_times):
        errors.append(f"{fixture_name}: beat length mismatch {len(our_times)} != {len(gold_times)}")
        return errors

    if len(our_nums) != len(gold_nums):
        errors.append(f"{fixture_name}: beat number length mismatch")
    if len(our_conf) != len(gold_conf):
        errors.append(f"{fixture_name}: beat confidence length mismatch")

    if len(our_times):
        max_time = float(np.max(np.abs(our_times - gold_times)))
        if max_time > MAX_BEAT_TIME_ABS:
            errors.append(f"{fixture_name}: beat_times max abs diff {max_time:.6f} > {MAX_BEAT_TIME_ABS}")

        if not np.array_equal(our_nums, gold_nums):
            errors.append(f"{fixture_name}: beat_numbers mismatch")

        max_conf = float(np.max(np.abs(our_conf - gold_conf)))
        if max_conf > MAX_CONF_ABS:
            errors.append(f"{fixture_name}: beat_confidences max abs diff {max_conf:.8f} > {MAX_CONF_ABS}")

    return errors


def compare_fixture(golden_path: Path, progress: bool = False) -> list[str]:
    payload = json.loads(golden_path.read_text(encoding="utf-8"))
    fixture_name = payload["fixture"]
    fixture_path = FIXTURES_DIR / fixture_name
    if not fixture_path.exists():
        return [f"missing fixture for golden: {fixture_name}"]

    samples, sample_rate = read_wav_mono_float(fixture_path)
    ours = analyze_pcm(samples, sample_rate=sample_rate, progress=progress)
    return compare_result(ours, payload["result"], fixture_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare madmom-beats-lite output against parity goldens.")
    parser.add_argument("--progress", action="store_true", help="Enable progress output while analyzing fixtures.")
    args = parser.parse_args()

    golden_files = sorted(GOLDENS_DIR.glob("*.json"))
    if not golden_files:
        raise RuntimeError("No goldens found. Run tools/regen/generate_goldens.py first.")

    errors: list[str] = []
    for golden_path in golden_files:
        errors.extend(compare_fixture(golden_path, progress=args.progress))

    if errors:
        for err in errors:
            print(f"FAIL: {err}")
        raise SystemExit(1)

    print(f"PASS: compared {len(golden_files)} golden files")


if __name__ == "__main__":
    main()
