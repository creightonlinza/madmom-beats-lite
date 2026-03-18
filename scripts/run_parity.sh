#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRE_DIR="$ROOT_DIR/benchmarks/predecoded"
GOLDEN_DIR="$ROOT_DIR/tests/golden"
OUT_DIR="$ROOT_DIR/benchmarks/results"

mkdir -p "$OUT_DIR"
shopt -s nullglob

ran=0
for golden in "$GOLDEN_DIR"/*.reference.json; do
  filename="$(basename "$golden")"
  id="${filename%.reference.json}"
  npz="$PRE_DIR/$id.npz"
  candidate="$OUT_DIR/$id.lite.json"

  if [[ -f "$npz" && -f "$golden" ]]; then
    python3 -m madmom_beats_lite.cli --input-npz "$npz" --output "$candidate" --no-progress
    python3 "$ROOT_DIR/scripts/compare_outputs.py" --golden "$golden" --candidate "$candidate" --float-tolerance 0.0
    ran=1
  fi
done

if [[ "$ran" -eq 0 ]]; then
  echo "No parity inputs found. Expected files under $PRE_DIR and $GOLDEN_DIR." >&2
  exit 1
fi

echo "Parity checks passed."
