#!/usr/bin/env python3
"""Run the reference extraction on unmodified upstream madmom from vendor/madmom."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parents[1] / "vendor" / "madmom"

# Must precede any madmom import, as the bundled runtime has the same package name.
sys.path.insert(0, str(VENDOR_DIR))

import madmom  # noqa: E402
import madmom.ml.hmm  # noqa: E402,F401

from madmom_beats_lite._shared import load_audio_from_npz  # noqa: E402
from madmom_beats_lite.reference import run_reference_extraction  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--beats-per-bar", type=str, default="3,4")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if VENDOR_DIR not in Path(madmom.__file__).resolve().parents:
        raise RuntimeError(f"expected upstream madmom, imported {madmom.__file__}")
    if args.input_npz is None:
        return 0
    beats_per_bar = tuple(int(x) for x in args.beats_per_bar.split(","))
    audio, sample_rate = load_audio_from_npz(args.input_npz)
    result = run_reference_extraction(audio, sample_rate, beats_per_bar=beats_per_bar)
    args.output.write_text(json.dumps(result.to_dict()), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
