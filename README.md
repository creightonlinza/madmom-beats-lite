# madmom-beats-lite

A fresh, minimal package that vendors only the required madmom code paths and model assets to analyze mono PCM and return:

- `beat_times` (seconds)
- `beat_numbers` (1-based in-bar index; 3/4 or 4/4 selected by DBN)
- `beat_confidences` (`[0, 1]`)

Downbeats are represented implicitly via `beat_numbers == 1`.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```python
import numpy as np
from madmom_beats_lite import analyze_pcm

# mono float32/float64 PCM at exactly 44100 Hz
samples = np.zeros(44100, dtype=np.float32)
result = analyze_pcm(samples, sample_rate=44100, progress=True)
print(result)
```

Set `debug=True` to emit coarse memory telemetry lines prefixed with `mbl_mem` (stage, major array sizes, current/max RSS).

Output contract:

```json
{
  "fps": 100,
  "beat_times": [0.2059, 0.645, 1.0222],
  "beat_numbers": [1, 2, 3],
  "beat_confidences": [0.83, 0.79, 0.81]
}
```

## Tests

Run unit tests:

```bash
pytest
```

Run short parity tests (skips automatically if parity cache is missing):

```bash
pytest -m parity
```

Run long parity test (requires `ffmpeg` and a local source file via `PARITY_LONG_SOURCE`):

```bash
PARITY_LONG_SOURCE=your_long_track.m4a PROGRESS=1 pytest -m parity_long
```

## Regen tools

See `tools/regen/README.md`.

## Performance (single-thread baseline)

For deployment parity, benchmark defaults are single-thread (`threads=1`).

Latest long-track benchmark (`gangnam.wav`, 3-run medians, `/usr/bin/time -l`):

- Upstream (`DBNDownBeatTracker -j 1`): `real=14.88s`, `max_rss=3,465,396,224`
- Lite (`analyze_pcm`, `MADMOM_BEATS_LITE_NUM_THREADS=1`): `real=14.94s`, `max_rss=2,642,706,432`
- Runtime vs upstream: `-0.40%` (slightly slower, within noise)
- Peak RSS vs upstream: `+23.74%` improvement (lower memory)

Reproduce lite-only timing/RSS stats (3 runs, min/median/max):

```bash
tools/bench/run_time_l.sh tools/regen/_cache/fixtures/gangnam.wav
```

Run one upstream-vs-lite CPU/memory comparison (defaults to `threads=1`):

```bash
.venv/bin/python tools/benchmark_cpu_mem.py --audio tools/regen/_cache/fixtures/gangnam.wav
```

## Build wheel artifacts

Build a local wheel (and verify bundled model assets are present in the wheel):

```bash
python -m pip install --upgrade build
python tools/build_wheels.py --out-dir wheelhouse --clean
```

Build wheel + sdist:

```bash
python tools/build_wheels.py --out-dir wheelhouse --sdist --clean
```

If your local environment is already provisioned and cannot reach package indexes, use:

```bash
.venv/bin/python tools/build_wheels.py --out-dir wheelhouse --sdist --clean --no-isolation
```

## Release workflow

GitHub Actions workflow `.github/workflows/release-wheels.yml` builds and publishes artifacts when a semver tag like `v1.0.0` is pushed.

- Wheel matrix: Linux/macOS/Windows for Python 3.11, 3.12, and 3.13
- One source distribution (`.tar.gz`)
- All artifacts are attached to the corresponding GitHub Release for direct download

## Licensing notes

- Vendored code derives from upstream madmom. See `src/madmom_beats_lite/_vendor/madmom_lite/UPSTREAM_MADMOM_LICENSE`.
- Bundled downbeat model files are under CC BY-NC-SA. See `src/madmom_beats_lite/assets/MODELS_LICENSE` and `ATTRIBUTION.md`.
