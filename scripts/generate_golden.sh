#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-all}"

SHORT_DIR="$ROOT_DIR/benchmarks/short"
FULL_DIR="$ROOT_DIR/benchmarks/full"
PRE_DIR="$ROOT_DIR/benchmarks/predecoded"
GOLDEN_DIR="$ROOT_DIR/tests/golden"

mkdir -p "$PRE_DIR" "$GOLDEN_DIR"

build_golden_from_source() {
  local input="$1"
  local label="$2"
  local stem id
  stem="$(basename "${input%.*}")"
  id="${stem}_${label}"
  if [[ -f "$input" ]]; then
    python3 "$ROOT_DIR/scripts/predecode_to_npz.py" --input "$input" --output "$PRE_DIR/$id.npz"
    python3 "$ROOT_DIR/scripts/reference_extract.py" --input-npz "$PRE_DIR/$id.npz" --output "$GOLDEN_DIR/$id.reference.json"
  fi
}

if [[ "$MODE" == "short" || "$MODE" == "all" ]]; then
  for input in "$SHORT_DIR"/*; do
    [[ -f "$input" ]] || continue
    build_golden_from_source "$input" "short"
  done
fi

if [[ "$MODE" == "full" || "$MODE" == "all" ]]; then
  for input in "$FULL_DIR"/*; do
    [[ -f "$input" ]] || continue
    build_golden_from_source "$input" "full"
  done
fi

echo "Golden generation complete: $GOLDEN_DIR"
