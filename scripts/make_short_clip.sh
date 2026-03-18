#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/benchmarks/short"

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <input-audio> [<input-audio> ...]" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

for input in "$@"; do
  if [[ ! -f "$input" ]]; then
    echo "Missing source: $input" >&2
    exit 1
  fi

  filename="$(basename "$input")"
  stem="${filename%.*}"
  ext="${filename##*.}"
  output="$OUT_DIR/${stem}_30s.${ext}"

  ffmpeg -v error -y -i "$input" -ss 0 -t 30 -c copy "$output"
  echo "Wrote: $output"
done
