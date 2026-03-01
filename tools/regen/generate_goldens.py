from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.regen.common import FIXTURES_DIR, GOLDENS_DIR, CACHE_ROOT, ensure_cache_dirs, read_wav_mono_float


def import_upstream_madmom(project_root: Path):
    source_root = project_root / "source" / "madmom"
    if not source_root.exists():
        raise RuntimeError("Missing upstream madmom submodule at source/madmom")

    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    try:
        import pyximport
    except ImportError as exc:
        raise RuntimeError("Cython/pyximport is required to import upstream madmom.") from exc

    pyximport.install(
        build_dir=str(CACHE_ROOT / ".pyxbld"),
        setup_args={"include_dirs": np.get_include()},
        language_level=3,
    )

    # Upstream imports `distribution("madmom")` in `madmom.__init__` which
    # fails when loading from source checkout only. Provide a lightweight shim.
    import importlib.metadata as _ilm

    _real_distribution = _ilm.distribution

    class _MadmomDist:
        version = "source-checkout"

    def _shim_distribution(name: str):
        if name == "madmom":
            return _MadmomDist()
        return _real_distribution(name)

    _ilm.distribution = _shim_distribution

    from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

    return RNNDownBeatProcessor, DBNDownBeatTrackingProcessor


def compute_contract(samples: np.ndarray, sample_rate: int, rnn_cls, dbn_cls) -> dict:
    if sample_rate != 44100:
        raise ValueError(f"Fixture sample rate must be 44100 Hz, got {sample_rate}")

    rnn = rnn_cls()
    activations = rnn(samples)
    dbn = dbn_cls(beats_per_bar=[3, 4], fps=100)
    beats = dbn(activations)

    beat_times = beats[:, 0].astype(float) if len(beats) else np.empty(0, dtype=float)
    beat_numbers = beats[:, 1].astype(int) if len(beats) else np.empty(0, dtype=int)

    if len(beat_times):
        frame_idx = np.clip(np.rint(beat_times * 100).astype(int), 0, len(activations) - 1)
        beat_ch = activations[frame_idx, 0]
        downbeat_ch = activations[frame_idx, 1]
        confidences = np.where(beat_numbers == 1, downbeat_ch, beat_ch)
        confidences = np.clip(confidences.astype(float), 0.0, 1.0)
    else:
        confidences = np.empty(0, dtype=float)

    return {
        "fps": 100,
        "beat_times": beat_times.tolist(),
        "beat_numbers": beat_numbers.tolist(),
        "beat_confidences": confidences.tolist(),
    }


def generate_one(fixture_path: Path, rnn_cls, dbn_cls) -> Path:
    samples, sample_rate = read_wav_mono_float(fixture_path)
    result = compute_contract(samples, sample_rate, rnn_cls, dbn_cls)
    payload = {
        "fixture": fixture_path.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    out_path = GOLDENS_DIR / f"{fixture_path.stem}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate upstream madmom parity goldens.")
    parser.add_argument(
        "--fixture",
        action="append",
        default=None,
        help="Specific fixture filename in tools/regen/_cache/fixtures (can be repeated).",
    )
    args = parser.parse_args()

    ensure_cache_dirs()
    project_root = Path(__file__).resolve().parents[2]
    rnn_cls, dbn_cls = import_upstream_madmom(project_root)

    if args.fixture:
        fixtures = [FIXTURES_DIR / name for name in args.fixture]
    else:
        fixtures = sorted(FIXTURES_DIR.glob("*.wav"))

    if not fixtures:
        raise RuntimeError("No fixtures found. Run tools/regen/fetch_fixtures.py first.")

    for fixture in fixtures:
        if not fixture.exists():
            raise RuntimeError(f"Fixture not found: {fixture}")
        out_path = generate_one(fixture, rnn_cls, dbn_cls)
        print(f"generated: {out_path}")


if __name__ == "__main__":
    main()
