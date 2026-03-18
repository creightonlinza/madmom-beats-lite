#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def _probe_audio(path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    output = subprocess.check_output(cmd, text=True)
    payload = json.loads(output)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"No audio stream found in {path}")
    stream = streams[0]
    sample_rate = int(stream["sample_rate"])
    channels = int(stream["channels"])
    return sample_rate, channels


def predecode_to_npz(input_path: Path, output_path: Path) -> None:
    sample_rate, channels = _probe_audio(input_path)

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-",
    ]
    raw = subprocess.check_output(cmd)
    audio = np.frombuffer(raw, dtype=np.float32)
    if channels > 1:
        audio = audio.reshape((-1, channels))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, audio=audio, sample_rate=np.int64(sample_rate))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predecode media with ffmpeg into NPZ {audio,sample_rate}")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    predecode_to_npz(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
