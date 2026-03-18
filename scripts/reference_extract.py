#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from madmom_beats_lite._shared import load_audio_from_npz
from madmom_beats_lite.reference import run_reference_extraction


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run direct madmom reference extraction")
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beats-per-bar", type=str, default="3,4")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    beats_per_bar = tuple(int(x.strip()) for x in args.beats_per_bar.split(",") if x.strip())
    audio, sample_rate = load_audio_from_npz(args.input_npz)
    result = run_reference_extraction(audio, sample_rate, beats_per_bar=beats_per_bar).to_dict()

    if args.pretty:
        text = json.dumps(result, indent=2, sort_keys=False) + "\n"
    else:
        text = json.dumps(result, separators=(",", ":")) + "\n"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
