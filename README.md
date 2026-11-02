# madmom-beats-lite

Minimal, parity-safe beat/downbeat extraction library with bundled madmom runtime components.

## Purpose

`madmom-beats-lite` exists to preserve the current beat/downbeat behavior while exposing:

- a standalone install artifact (no separate `madmom` package install),
- a focused Python API for predecoded audio,
- a practical CLI,
- structured progress events (`0..100`, monotonic),
- strict parity workflow scripts and tests,
- a wheel artifact build path for downstream consumption.

Behavior parity is the priority: performance work must keep the output bit-identical.

## Why This Is a Win

- Lite API surface: one focused extraction contract for beat/downbeat output + progress events.
- Standalone artifact: no separate `madmom` package install required for consumers; wheel includes the required trimmed `madmom` runtime + `downbeats_blstm` model files.
- Strict parity guardrails: output is compared against upstream `madmom` with zero tolerance.
- Practical observability: monotonic structured progress events for CLI and Python API users.
- Meaningful footprint reduction after trimming:
  - vendored runtime (`src/madmom`): about `41M` -> `6.7M`
  - vendored models (`src/madmom/models`): about `34M` -> `3.2M`
  - local wheel build comparison (cp311/macOS arm64):
    - regular vendored `madmom`: `24,542,003` bytes (~`24.5M`)
    - `madmom-beats-lite`: `3,465,609` bytes (~`3.5M`)
    - reduction: ~`7.08x` smaller (~`85.88%` smaller)

## Lite Scope

The bundled `madmom` runtime is intentionally trimmed to the offline beat/downbeat path used here:

- kept: signal/STFT/spectrogram preprocessing, downbeat tracking, HMM decoder, required NN layers, and `downbeats_blstm` model files,
- removed: unrelated feature domains (onsets/chords/key/notes/tempo/pattern tracking assets), evaluation modules, and unused model families.

## Pinned Upstream

Official upstream `madmom` is pinned as a git submodule:

- path: `vendor/madmom`
- url: `https://github.com/CPJKU/madmom.git`
- commit: `27f032e8947204902c675e5e341a3faf5dc86dae`

No floating branch is used.

## Architecture

Two paths are intentionally maintained:

1. Reference parity path (`run_reference_extraction`), directly mirroring upstream:
   - `Signal(audio, sample_rate=<source_sr>)`
   - `RNNDownBeatProcessor()` at `fps=100`
   - `DBNDownBeatTrackingProcessor(beats_per_bar=[3,4], fps=100)`
2. Production API path (`extract_beats`), equivalent processing stages instrumented for fine-grained progress reporting.
3. Shared output normalization to the structured result contract.

The core API accepts predecoded `numpy.ndarray` audio only. It does not decode files.

## Input Contract

Core library path (`extract_beats`) expects:

- `audio`: predecoded `numpy.ndarray` (`1D mono` or `2D multi-channel`)
- `sample_rate`: original sample rate of that array

No decode layer is introduced in the core path.

## Output Contract

Minimum guaranteed fields:

```json
{
  "fps": 100,
  "beat_times": [0.51, 1.03, 1.54],
  "beat_numbers": [1, 2, 3],
  "beat_confidences": [0.93, 0.88, 0.91],
  "downbeat_times": [0.51],
  "downbeat_confidences": [0.93]
}
```

`downbeat_*` fields are preserved in a consistent structured form.

## Progress Contract

Progress events are structured JSON with monotonic integer percentages (`0..100`):

```json
{"percent":12,"stage":"preprocess","message":"building features"}
```

- CLI: progress on `stderr`, final result on `stdout` (or `--output` file)
- Python API: provide `progress_callback(event)`
- Main runtime dependencies for beat/downbeat extraction: `numpy`, `scipy`; `ffmpeg` for input that is not 44.1 kHz

## Local Setup

```bash
./scripts/setup_submodule.sh
./scripts/setup_env.sh
source .venv/bin/activate
```

## Python API Usage

```python
import numpy as np
from madmom_beats_lite import extract_beats

# audio: predecoded ndarray, sample_rate: int
result = extract_beats(audio, sample_rate)
payload = result.to_dict()
```

## CLI Usage

```bash
python3 -m madmom_beats_lite.cli \
  --input-npz path/to/audio.npz \
  --output result.json
```

NPZ input must contain `audio` and `sample_rate`.

## Parity Workflow

Parity is checked against unmodified upstream `madmom` from `vendor/madmom`, run in the same environment. Output of numerical code depends on the platform's math libraries, so results stored on one setup do not stay comparable on another.

```bash
./scripts/setup_upstream.sh                # part of setup_env.sh
pytest tests/test_parity_upstream.py       # sample audio from the submodule
```

Comparison is strict (`float_tolerance=0.0`).

To also compare your own tracks, predecode them to `benchmarks/predecoded/`:

```bash
python3 scripts/predecode_to_npz.py --input /path/to/track.m4a --output benchmarks/predecoded/track.npz
UPSTREAM_PARITY_ASSETS=1 pytest tests/test_parity_upstream.py
```

`./scripts/make_short_clip.sh /path/to/track.m4a` writes a 30s clip to `benchmarks/short/` to predecode instead.

## Tests

Run everything:

```bash
pytest
```

Included:

- output contract tests
- progress contract tests
- native LSTM and Viterbi regression tests
- parity with the reference path (`test_parity_ci_sample.py`)
- parity with upstream `madmom` (`test_parity_upstream.py`, skipped until `setup_upstream.sh` has run)

## Build Wheel

```bash
python3 -m build --no-isolation
```

## GitHub Actions

Workflow at `.github/workflows/build.yml`:

1. checkout with submodules,
2. install pinned dependencies,
3. install `madmom-beats-lite` (bundled madmom runtime) with `--no-deps`,
4. run tests/parity checks suitable for CI,
5. build wheels via `cibuildwheel` for:
   - Linux `x86_64`: `cp310`, `cp311`
   - macOS `universal2`: `cp310`, `cp311`
   - Windows `AMD64`: `cp310`, `cp311`
6. build sdist (`.tar.gz`),
7. on GitHub Release publish, upload all wheels + sdist as release assets.
