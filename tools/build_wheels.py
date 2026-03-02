#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REQUIRED_WHEEL_FILES = (
    "madmom_beats_lite/assets/MODELS_LICENSE",
)

REQUIRED_WHEEL_GLOBS = (
    "madmom_beats_lite/assets/downbeats/2016/downbeats_blstm_*.pkl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build wheel/sdist artifacts and verify packaged model assets."
    )
    parser.add_argument(
        "--out-dir",
        default="wheelhouse",
        help="Output directory for built artifacts (default: wheelhouse).",
    )
    parser.add_argument(
        "--sdist",
        action="store_true",
        help="Build an sdist in addition to wheel artifacts.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output directory before building.",
    )
    parser.add_argument(
        "--skip-verify-assets",
        action="store_true",
        help="Skip wheel asset verification after build.",
    )
    parser.add_argument(
        "--no-isolation",
        action="store_true",
        help="Build without isolated PEP 517 environments.",
    )
    return parser.parse_args()


def ensure_build_module() -> None:
    if importlib.util.find_spec("build") is None:
        raise SystemExit(
            "Missing dependency: python -m build is required. "
            "Install it with: python -m pip install build"
        )


def run_build(
    repo_root: Path, out_dir: Path, include_sdist: bool, no_isolation: bool
) -> None:
    cmd = [sys.executable, "-m", "build", "--wheel", "--outdir", str(out_dir)]
    if include_sdist:
        cmd.insert(3, "--sdist")
    if no_isolation:
        cmd.append("--no-isolation")
    subprocess.run(cmd, cwd=repo_root, check=True)


def verify_wheel_assets(wheel_path: Path) -> None:
    with zipfile.ZipFile(wheel_path) as whl:
        names = set(whl.namelist())
        missing = [name for name in REQUIRED_WHEEL_FILES if name not in names]
        if missing:
            raise SystemExit(
                f"{wheel_path.name} is missing required asset files: {missing}"
            )

        for pattern in REQUIRED_WHEEL_GLOBS:
            if not any(fnmatch.fnmatch(name, pattern) for name in names):
                raise SystemExit(
                    f"{wheel_path.name} is missing asset files matching {pattern!r}"
                )


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir = out_dir.resolve()

    ensure_build_module()

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_build(
        repo_root=repo_root,
        out_dir=out_dir,
        include_sdist=args.sdist,
        no_isolation=args.no_isolation,
    )

    wheels = sorted(out_dir.glob("*.whl"))
    if not wheels:
        raise SystemExit(f"No wheel artifacts were produced in {out_dir}")

    if not args.skip_verify_assets:
        for wheel_path in wheels:
            verify_wheel_assets(wheel_path)

    print(f"Built {len(wheels)} wheel(s) in {out_dir}")
    if args.sdist:
        sdists = sorted(out_dir.glob("*.tar.gz"))
        print(f"Built {len(sdists)} sdist artifact(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
