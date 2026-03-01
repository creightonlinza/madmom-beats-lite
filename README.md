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

## Licensing notes

- Vendored code derives from upstream madmom. See `src/madmom_beats_lite/_vendor/madmom_lite/UPSTREAM_MADMOM_LICENSE`.
- Bundled downbeat model files are under CC BY-NC-SA. See `src/madmom_beats_lite/assets/MODELS_LICENSE` and `ATTRIBUTION.md`.
