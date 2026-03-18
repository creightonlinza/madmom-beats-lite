from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ._shared import load_audio_from_npz
from .api import ExtractionConfig, extract_beats
from .types import ProgressEvent

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="madmom-beats-lite",
        description="Beat/downbeat extraction for predecoded audio arrays.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-npz", type=Path, help="NPZ file with keys {audio|samples|waveform,sample_rate}")
    input_group.add_argument("--input-npy", type=Path, help="NPY file containing decoded audio samples")

    parser.add_argument("--sample-rate", type=int, help="Required with --input-npy")
    parser.add_argument("--output", type=Path, help="Write final JSON result to this path instead of stdout")
    parser.add_argument("--beats-per-bar", type=str, default="3,4", help="DBN beats-per-bar list (default: 3,4)")
    parser.add_argument("--no-progress", action="store_true", help="Disable structured progress JSON on stderr")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print final JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.input_npy and args.sample_rate is None:
        parser.error("--sample-rate is required with --input-npy")

    if args.input_npz:
        audio, sample_rate = load_audio_from_npz(args.input_npz)
    else:
        audio = np.load(args.input_npy, allow_pickle=False)
        sample_rate = int(args.sample_rate)

    beats_per_bar = tuple(int(x.strip()) for x in args.beats_per_bar.split(",") if x.strip())
    config = ExtractionConfig(fps=100, beats_per_bar=beats_per_bar)

    def progress_cb(event: ProgressEvent) -> None:
        if not args.no_progress:
            sys.stderr.write(json.dumps(event.to_dict(), separators=(",", ":")) + "\n")
            sys.stderr.flush()

    result = extract_beats(audio, sample_rate, config=config, progress_callback=progress_cb)
    payload = result.to_dict()

    if args.pretty:
        text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    else:
        text = json.dumps(payload, separators=(",", ":")) + "\n"

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
