#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
AUDIO_PATH="${1:-$ROOT/tools/regen/_cache/fixtures/gangnam.wav}"
RUNS="${RUNS:-3}"
THREADS="${THREADS:-1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "python not found/executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$AUDIO_PATH" ]]; then
  echo "audio file not found: $AUDIO_PATH" >&2
  exit 1
fi

RUN_SCRIPT="$(mktemp -t mbl_time_l_run_XXXXXX.py)"
RESULTS_CSV="$(mktemp -t mbl_time_l_results_XXXXXX.csv)"
cleanup() {
  rm -f "$RUN_SCRIPT" "$RESULTS_CSV"
}
trap cleanup EXIT

cat >"$RUN_SCRIPT" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

repo = Path(sys.argv[1])
audio = Path(sys.argv[2])
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "src"))

from madmom_beats_lite import analyze_pcm
from tools.regen.common import read_wav_mono_float

def main() -> None:
    samples, sample_rate = read_wav_mono_float(audio)
    _ = analyze_pcm(samples, sample_rate=sample_rate, progress=False)


if __name__ == "__main__":
    main()
PY

echo "run,real_s,user_s,sys_s,max_rss_bytes,peak_memory_bytes" >"$RESULTS_CSV"
for i in $(seq 1 "$RUNS"); do
  out="$(
    { /usr/bin/time -l env MADMOM_BEATS_LITE_NUM_THREADS="$THREADS" "$PYTHON_BIN" "$RUN_SCRIPT" "$ROOT" "$AUDIO_PATH" >/dev/null; } 2>&1
  )"
  real_s="$(echo "$out" | awk '/ real / {print $1; exit}')"
  user_s="$(echo "$out" | awk '/ real / {print $3; exit}')"
  sys_s="$(echo "$out" | awk '/ real / {print $5; exit}')"
  max_rss="$(echo "$out" | awk '/maximum resident set size/ {print $1; exit}')"
  peak_mem="$(echo "$out" | awk '/peak memory footprint/ {print $1; exit}')"
  if [[ -z "$real_s" || -z "$max_rss" ]]; then
    echo "failed to parse /usr/bin/time -l output on run $i" >&2
    echo "$out" >&2
    exit 1
  fi
  echo "$i,$real_s,$user_s,$sys_s,$max_rss,$peak_mem" >>"$RESULTS_CSV"
  echo "run $i: real=${real_s}s user=${user_s}s sys=${sys_s}s max_rss=${max_rss} peak=${peak_mem}"
done

python - "$RESULTS_CSV" <<'PY'
from __future__ import annotations

import csv
import statistics
import sys

path = sys.argv[1]
rows = list(csv.DictReader(open(path, "r", encoding="utf-8")))
if not rows:
    raise SystemExit("no benchmark rows")

def vals(key: str):
    return [float(r[key]) for r in rows]

def stat_line(key: str, unit: str) -> str:
    data = vals(key)
    return (
        f"{key}: min={min(data):.3f}{unit} "
        f"median={statistics.median(data):.3f}{unit} "
        f"max={max(data):.3f}{unit}"
    )

print("")
print(f"runs: {len(rows)}")
print(stat_line("real_s", "s"))
print(stat_line("user_s", "s"))
print(stat_line("sys_s", "s"))
print(stat_line("max_rss_bytes", "B"))
peak_vals = vals("peak_memory_bytes")
if all(v > 0 for v in peak_vals):
    print(stat_line("peak_memory_bytes", "B"))
PY
