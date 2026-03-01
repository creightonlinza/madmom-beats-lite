from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = PROJECT_ROOT / "tools" / "regen" / "_cache"
FIXTURES_DIR = CACHE_ROOT / "fixtures"
GOLDENS_DIR = CACHE_ROOT / "goldens"
MANIFEST_PATH = PROJECT_ROOT / "tools" / "regen" / "fixtures_manifest.json"


def ensure_cache_dirs() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    GOLDENS_DIR.mkdir(parents=True, exist_ok=True)


def load_manifest() -> list[dict]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_wav_mono_float(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        num_channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        num_frames = wav.getnframes()
        frames = wav.readframes(num_frames)

    if sample_width == 1:
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if num_channels > 1:
        data = data.reshape(-1, num_channels).mean(axis=1)

    return data.astype(np.float32, copy=False), sample_rate


def write_wav_mono_float(path: Path, samples: np.ndarray, sample_rate: int = 44100) -> None:
    samples = np.asarray(samples, dtype=np.float32)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def generate_waltz_click_track(path: Path, sample_rate: int = 44100, bars: int = 8, bpm: float = 180.0) -> None:
    beats_per_bar = 3
    beat_duration = 60.0 / bpm
    total_beats = bars * beats_per_bar
    total_samples = int((total_beats * beat_duration + 0.5) * sample_rate)
    out = np.zeros(total_samples, dtype=np.float32)

    click_len = int(0.02 * sample_rate)
    t = np.arange(click_len, dtype=np.float32) / sample_rate
    env = np.exp(-t * 80.0)
    strong = np.sin(2 * np.pi * 1600.0 * t) * env
    weak = np.sin(2 * np.pi * 1100.0 * t) * env

    for beat in range(total_beats):
        start = int(round(beat * beat_duration * sample_rate))
        stop = min(total_samples, start + click_len)
        if stop <= start:
            continue
        pulse = strong if beat % beats_per_bar == 0 else weak
        out[start:stop] += pulse[: stop - start]

    write_wav_mono_float(path, out, sample_rate=sample_rate)
