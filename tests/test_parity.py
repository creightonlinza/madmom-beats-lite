from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from madmom_beats_lite import analyze_pcm
from tools.regen.common import FIXTURES_DIR, GOLDENS_DIR, ensure_cache_dirs, read_wav_mono_float
from tools.regen.compare_parity import MAX_BEAT_TIME_ABS, MAX_CONF_ABS, compare_result
from tools.regen.generate_goldens import generate_one, import_upstream_madmom


@pytest.mark.parity
def test_parity_against_cached_goldens() -> None:
    if not FIXTURES_DIR.exists() or not GOLDENS_DIR.exists():
        pytest.skip("parity cache missing (run tools/regen scripts first)")

    golden_files = sorted(GOLDENS_DIR.glob("*.json"))
    if not golden_files:
        pytest.skip("no cached parity goldens found")

    errors: list[str] = []
    for golden_path in golden_files:
        payload = json.loads(golden_path.read_text(encoding="utf-8"))
        fixture_name = payload["fixture"]
        fixture_path = FIXTURES_DIR / fixture_name
        if not fixture_path.exists():
            errors.append(f"missing fixture {fixture_name}")
            continue

        samples, sample_rate = read_wav_mono_float(fixture_path)
        ours = analyze_pcm(samples, sample_rate=sample_rate)
        errors.extend(compare_result(ours, payload["result"], fixture_name))

    if errors:
        pytest.fail("\n".join(errors))


@pytest.mark.parity_long
def test_parity_long_local_file() -> None:
    source_env = os.environ.get("PARITY_LONG_SOURCE")
    if not source_env:
        pytest.skip("set PARITY_LONG_SOURCE to a local audio file path for parity_long")
    source = Path(source_env).expanduser()
    if not source.exists():
        pytest.skip(f"missing PARITY_LONG_SOURCE file: {source}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not found")

    ensure_cache_dirs()
    fixture = FIXTURES_DIR / "parity_long_input.wav"
    if not fixture.exists():
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            "44100",
            str(fixture),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    project_root = Path(__file__).resolve().parents[1]
    rnn_cls, dbn_cls = import_upstream_madmom(project_root)
    golden_path = generate_one(fixture, rnn_cls, dbn_cls)

    payload = json.loads(golden_path.read_text(encoding="utf-8"))
    samples, sample_rate = read_wav_mono_float(fixture)

    progress = os.environ.get("PROGRESS") == "1"
    ours = analyze_pcm(samples, sample_rate=sample_rate, progress=progress)

    errors = compare_result(ours, payload["result"], fixture.name)
    if errors:
        pytest.fail("\n".join(errors))

    # explicit tolerances from project contract
    assert MAX_BEAT_TIME_ABS == 0.01
    assert MAX_CONF_ABS == 1e-5
