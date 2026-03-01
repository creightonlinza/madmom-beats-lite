from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _summary(samples: list[float]) -> dict:
    return {
        "runs": len(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "mean_s": statistics.mean(samples),
        "median_s": statistics.median(samples),
        "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


_LITE_ONCE_SCRIPT = """
from pathlib import Path
import sys

repo = Path(sys.argv[1])
audio = Path(sys.argv[2])
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "src"))

from madmom_beats_lite import analyze_pcm
from tools.regen.common import read_wav_mono_float

samples, sample_rate = read_wav_mono_float(audio)
_ = analyze_pcm(samples, sample_rate=sample_rate, progress=False)
"""


def _time_upstream_cli(exe: Path, audio_path: Path, runs: int, warmup: int, threads: int) -> list[float]:
    timings: list[float] = []

    def one() -> float:
        with tempfile.NamedTemporaryFile(prefix="upstream_bench_", suffix=".txt", delete=False) as f:
            out_path = Path(f.name)
        cmd = [str(exe), "single", str(audio_path), "-j", str(threads), "-o", str(out_path)]
        start = time.perf_counter()
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            out_path.unlink(missing_ok=True)
        return time.perf_counter() - start

    for _ in range(warmup):
        one()
    for _ in range(runs):
        timings.append(one())
    return timings


def _time_lite(python_exe: Path, audio_path: Path, runs: int, warmup: int, threads: int) -> list[float]:
    timings: list[float] = []
    env = os.environ.copy()
    env["MADMOM_BEATS_LITE_NUM_THREADS"] = str(threads)

    def one() -> float:
        cmd = [str(python_exe), "-c", _LITE_ONCE_SCRIPT, str(PROJECT_ROOT), str(audio_path)]
        start = time.perf_counter()
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        return time.perf_counter() - start

    for _ in range(warmup):
        one()
    for _ in range(runs):
        timings.append(one())
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark madmom-beats-lite vs upstream DBNDownBeatTracker CLI.")
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path("tools/regen/_cache/fixtures/parity_long_input.wav"),
        help="Long-track WAV path (mono 44.1k preferred).",
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--threads",
        type=int,
        default=10,
        help="Worker threads for both upstream CLI (-j) and lite (MADMOM_BEATS_LITE_NUM_THREADS).",
    )
    parser.add_argument(
        "--upstream-cli",
        type=Path,
        default=Path(".venv/bin/DBNDownBeatTracker"),
        help="Path to upstream CLI executable.",
    )
    parser.add_argument(
        "--lite-python",
        type=Path,
        default=Path(".venv/bin/python"),
        help="Python executable used to benchmark madmom-beats-lite in isolated subprocesses.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary.")
    args = parser.parse_args()

    if not args.audio.exists():
        raise SystemExit(f"Audio file not found: {args.audio}")
    if not args.upstream_cli.exists():
        raise SystemExit(f"Upstream CLI not found: {args.upstream_cli}")
    if not args.lite_python.exists():
        raise SystemExit(f"Lite Python executable not found: {args.lite_python}")
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if args.threads < 1:
        raise SystemExit("--threads must be >= 1")

    upstream_timings = _time_upstream_cli(args.upstream_cli, args.audio, args.runs, args.warmup, args.threads)
    lite_timings = _time_lite(args.lite_python, args.audio, args.runs, args.warmup, args.threads)

    upstream = _summary(upstream_timings)
    lite = _summary(lite_timings)
    ratio = {
        "median_speedup_x": upstream["median_s"] / lite["median_s"],
        "mean_speedup_x": upstream["mean_s"] / lite["mean_s"],
    }

    payload = {
        "audio": str(args.audio),
        "warmup": args.warmup,
        "runs": args.runs,
        "threads": args.threads,
        "upstream_cli": upstream,
        "madmom_beats_lite": lite,
        "ratio": ratio,
        "raw": {
            "upstream_cli_s": upstream_timings,
            "madmom_beats_lite_s": lite_timings,
        },
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"audio: {payload['audio']}")
        print(f"warmup: {args.warmup}, runs: {args.runs}, threads: {args.threads}")
        print("upstream_cli:")
        print(f"  median={upstream['median_s']:.3f}s mean={upstream['mean_s']:.3f}s min={upstream['min_s']:.3f}s max={upstream['max_s']:.3f}s")
        print("madmom_beats_lite (subprocess read_wav + analyze_pcm):")
        print(f"  median={lite['median_s']:.3f}s mean={lite['mean_s']:.3f}s min={lite['min_s']:.3f}s max={lite['max_s']:.3f}s")
        print(f"speedup: median={ratio['median_speedup_x']:.3f}x mean={ratio['mean_speedup_x']:.3f}x")


if __name__ == "__main__":
    main()
