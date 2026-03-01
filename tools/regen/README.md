# Regen tooling

This directory contains reproducibility scripts for fixtures and parity goldens.

All generated outputs go to `tools/regen/_cache/` (gitignored):

- `tools/regen/_cache/fixtures/`
- `tools/regen/_cache/goldens/`
- `tools/regen/_cache/.pyxbld/` (temporary build artifacts for upstream Cython import)

## Scripts

- `fetch_fixtures.py`: fetches/downstreams fixture audio into cache.
- `generate_goldens.py`: runs upstream madmom on cached fixtures to produce golden JSON.
- `compare_parity.py`: compares `madmom_beats_lite.analyze_pcm` against cached goldens.

## Typical flow

```bash
python tools/regen/fetch_fixtures.py
python tools/regen/generate_goldens.py
python tools/regen/compare_parity.py
```
