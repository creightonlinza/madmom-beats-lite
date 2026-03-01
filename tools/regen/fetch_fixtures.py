from __future__ import annotations

import shutil
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.regen.common import (
    FIXTURES_DIR,
    ensure_cache_dirs,
    generate_waltz_click_track,
    load_manifest,
    read_wav_mono_float,
)


def _download(url: str, dest: Path, timeout: int = 30) -> None:
    with requests.get(url, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        with dest.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 14):
                if chunk:
                    f.write(chunk)


def fetch_fixture(entry: dict) -> tuple[Path, str]:
    filename = entry["filename"]
    destination = FIXTURES_DIR / filename
    if destination.exists():
        try:
            _, sample_rate = read_wav_mono_float(destination)
            if sample_rate == 44100:
                return destination, "cached"
        except Exception:
            pass
        destination.unlink()

    url = entry.get("url")
    if url:
        try:
            _download(url, destination)
            _, sample_rate = read_wav_mono_float(destination)
            if sample_rate != 44100:
                raise RuntimeError(f"fixture sample_rate must be 44100, got {sample_rate}")
            return destination, "downloaded"
        except Exception:
            if destination.exists():
                destination.unlink()

    fallback = entry.get("fallback_local")
    if fallback:
        fallback_path = PROJECT_ROOT / fallback
        if fallback_path.exists():
            shutil.copy2(fallback_path, destination)
            _, sample_rate = read_wav_mono_float(destination)
            if sample_rate != 44100:
                raise RuntimeError(
                    f"fallback fixture sample_rate must be 44100, got {sample_rate}: {fallback_path}"
                )
            return destination, "copied-local"

    if entry.get("generate_if_missing") == "waltz_click_track":
        generate_waltz_click_track(destination)
        return destination, "generated"

    raise RuntimeError(f"Unable to fetch fixture: {entry['name']}")


def main() -> None:
    ensure_cache_dirs()
    manifest = load_manifest()
    for entry in manifest:
        path, status = fetch_fixture(entry)
        print(f"{entry['name']}: {status} -> {path}")


if __name__ == "__main__":
    main()
