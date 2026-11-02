#!/usr/bin/env bash
set -euo pipefail

# Build the extensions of unmodified upstream madmom in place for the parity tests,
# and its package metadata, which madmom reads its version from on import.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/vendor/madmom"
python setup.py --quiet egg_info build_ext --inplace
