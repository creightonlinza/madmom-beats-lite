#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path


def _default_audio_path(root: Path) -> Path:
    fixtures = root / "tools" / "regen" / "_cache" / "fixtures"
    preferred = fixtures / "parity_long_input.wav"
    if preferred.exists():
        return preferred
    wavs = sorted(fixtures.glob("*.wav"))
    if wavs:
        return wavs[0]
    return preferred


def _run_time_l(cmd: list[str]) -> dict:
    timed_cmd = ["/usr/bin/time", "-l", *cmd]
    proc = subprocess.run(timed_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False)
    out = proc.stderr

    real_pat = re.compile(r"\s*([0-9.]+) real\s+([0-9.]+) user\s+([0-9.]+) sys")
    rss_pat = re.compile(r"\s*(\d+)\s+maximum resident set size")
    peak_pat = re.compile(r"\s*(\d+)\s+peak memory footprint")

    real_m = real_pat.search(out)
    rss_m = rss_pat.search(out)
    peak_m = peak_pat.search(out)

    if real_m is None:
        raise RuntimeError(f"Failed to parse /usr/bin/time -l output:\n{out}")
    if proc.returncode != 0 and "Traceback (most recent call last):" in out:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\n{out}")

    return {
        "time_exit_code": proc.returncode,
        "real_s": float(real_m.group(1)),
        "user_s": float(real_m.group(2)),
        "sys_s": float(real_m.group(3)),
        "max_rss_bytes": int(rss_m.group(1)) if rss_m else None,
        "peak_memory_bytes": int(peak_m.group(1)) if peak_m else None,
        "raw_time_output": out,
    }


def _run_upstream(upstream_cli: Path, audio: Path, output_file: Path, threads: int) -> dict:
    cmd = [str(upstream_cli), "single", str(audio), "-j", str(threads), "-o", str(output_file)]
    return _run_time_l(cmd)


def _run_lite(venv_python: Path, repo_root: Path, audio: Path, threads: int) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix="_lite_cpu_mem.py", delete=False) as f:
        script = Path(f.name)
        f.write(
            "from __future__ import annotations\n"
            "import sys\n"
            f"sys.path.insert(0, {repr(str(repo_root))})\n"
            f"sys.path.insert(0, {repr(str(repo_root / 'src'))})\n"
            "from pathlib import Path\n"
            "from madmom_beats_lite import analyze_pcm\n"
            "from tools.regen.common import read_wav_mono_float\n"
            "def main():\n"
            f"    samples, sr = read_wav_mono_float(Path({repr(str(audio))}))\n"
            "    _ = analyze_pcm(samples, sample_rate=sr, progress=False)\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )

    try:
        cmd = [str(venv_python), str(script)]
        env = dict(os.environ)
        env["MADMOM_BEATS_LITE_NUM_THREADS"] = str(threads)
        timed_cmd = ["/usr/bin/time", "-l", *cmd]
        proc = subprocess.run(timed_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False, env=env)
        out = proc.stderr

        real_pat = re.compile(r"\s*([0-9.]+) real\s+([0-9.]+) user\s+([0-9.]+) sys")
        rss_pat = re.compile(r"\s*(\d+)\s+maximum resident set size")
        peak_pat = re.compile(r"\s*(\d+)\s+peak memory footprint")
        real_m = real_pat.search(out)
        rss_m = rss_pat.search(out)
        peak_m = peak_pat.search(out)
        if real_m is None:
            raise RuntimeError(f"Failed to parse /usr/bin/time -l output:\n{out}")
        if proc.returncode != 0 and "Traceback (most recent call last):" in out:
            raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\n{out}")
        return {
            "time_exit_code": proc.returncode,
            "real_s": float(real_m.group(1)),
            "user_s": float(real_m.group(2)),
            "sys_s": float(real_m.group(3)),
            "max_rss_bytes": int(rss_m.group(1)) if rss_m else None,
            "peak_memory_bytes": int(peak_m.group(1)) if peak_m else None,
            "raw_time_output": out,
        }
    finally:
        script.unlink(missing_ok=True)


def _format_bytes(n: int | None) -> str:
    if n is None:
        return "n/a"
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} PB"


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot CPU+RAM comparison: upstream madmom CLI vs madmom-beats-lite")
    parser.add_argument("--audio", type=Path, default=None, help="Path to long WAV file to benchmark")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--venv-python", type=Path, default=Path(".venv/bin/python"))
    parser.add_argument("--upstream-cli", type=Path, default=Path(".venv/bin/DBNDownBeatTracker"))
    parser.add_argument(
        "--threads",
        type=int,
        default=10,
        help="Thread count applied to both upstream CLI (-j) and lite (MADMOM_BEATS_LITE_NUM_THREADS).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("tools/regen/_cache/benchmarks"))
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    audio = args.audio.resolve() if args.audio else _default_audio_path(repo_root).resolve()
    venv_python = (repo_root / args.venv_python) if not args.venv_python.is_absolute() else args.venv_python
    upstream_cli = (repo_root / args.upstream_cli) if not args.upstream_cli.is_absolute() else args.upstream_cli
    out_dir = (repo_root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not audio.exists():
        raise SystemExit(f"Audio file not found: {audio}")
    if not venv_python.exists():
        raise SystemExit(f"Python not found: {venv_python}")
    if not upstream_cli.exists():
        raise SystemExit(f"Upstream CLI not found: {upstream_cli}")
    if args.threads < 1:
        raise SystemExit("--threads must be >= 1")

    upstream_out = out_dir / "upstream_once.txt"
    upstream = _run_upstream(upstream_cli, audio, upstream_out, threads=args.threads)
    lite = _run_lite(venv_python, repo_root, audio, threads=args.threads)

    payload = {
        "audio": str(audio),
        "threads": args.threads,
        "upstream": upstream,
        "lite": lite,
        "comparison": {
            "real_speedup_x": upstream["real_s"] / lite["real_s"],
            "max_rss_ratio_x": (
                upstream["max_rss_bytes"] / lite["max_rss_bytes"]
                if upstream["max_rss_bytes"] is not None and lite["max_rss_bytes"] is not None
                else None
            ),
        },
    }

    result_path = out_dir / "cpu_mem_once.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"audio: {audio}")
    print(f"threads: {args.threads}")
    print("upstream (DBNDownBeatTracker)")
    print(f"  time_exit_code={upstream['time_exit_code']}")
    print(f"  real={upstream['real_s']:.2f}s user={upstream['user_s']:.2f}s sys={upstream['sys_s']:.2f}s")
    print(
        f"  max_rss={upstream['max_rss_bytes']} ({_format_bytes(upstream['max_rss_bytes'])}) "
        f"peak={upstream['peak_memory_bytes']} ({_format_bytes(upstream['peak_memory_bytes'])})"
    )
    print("madmom-beats-lite")
    print(f"  time_exit_code={lite['time_exit_code']}")
    print(f"  real={lite['real_s']:.2f}s user={lite['user_s']:.2f}s sys={lite['sys_s']:.2f}s")
    print(
        f"  max_rss={lite['max_rss_bytes']} ({_format_bytes(lite['max_rss_bytes'])}) "
        f"peak={lite['peak_memory_bytes']} ({_format_bytes(lite['peak_memory_bytes'])})"
    )
    max_rss_ratio = payload["comparison"]["max_rss_ratio_x"]
    max_rss_ratio_str = f"{max_rss_ratio:.3f}x" if max_rss_ratio is not None else "n/a"
    print(f"comparison: real_speedup={payload['comparison']['real_speedup_x']:.3f}x max_rss_ratio={max_rss_ratio_str}")
    print(f"saved: {result_path}")


if __name__ == "__main__":
    main()
