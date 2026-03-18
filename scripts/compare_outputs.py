#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from madmom_beats_lite.parity import compare_payloads


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict parity comparison for madmom-beats-lite payloads")
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--float-tolerance", type=float, default=0.0)
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))

    errors = compare_payloads(golden, candidate, float_tolerance=args.float_tolerance)
    if errors:
        for msg in errors:
            print(msg, file=sys.stderr)
        return 1

    print("PARITY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
