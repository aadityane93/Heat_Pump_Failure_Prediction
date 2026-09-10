"""Download the public HEAPO data and build local dashboard artifacts."""

from __future__ import annotations

import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable


DEFAULT_DATA_URL = (
    "https://zenodo.org/records/15056919/files/heapo_data.zip?download=1"
)
HEAPO_LOADER_URL = "https://raw.githubusercontent.com/tbrumue/heapo/main/src/heapo.py"


def _download(
    url: str,
    target: Path,
    progress: Callable[[str], None] | None = None,
    label: str = "HEAPO data",
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0

            def report(block_count: int, block_size: int, _total: int) -> None:
                nonlocal downloaded
                downloaded = min(block_count * block_size, total or block_count * block_size)
                if total:
                    message = f"Downloading {label}: {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({downloaded / total:.0%})"
                else:
                    message = f"Downloading {label}: {downloaded / 1e6:.1f} MB"
                if progress:
                    progress(message)
                else:
                    print(f"\r{message}", end="", flush=True)

            block_size = 1024 * 1024
            block_count = 0
            while True:
                block = response.read(block_size)
                if not block:
                    break
                output.write(block)
                block_count += 1
                report(block_count, block_size, total)
            if not progress:
                print()
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _extract(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe path in HEAPO archive: {member.filename}")
        archive.extractall(destination)


def ensure_heapo_loader(base_dir: Path) -> None:
    """Download the upstream loader module, which is not packaged for pip."""
    loader_dir = base_dir / ".heapo_loader"
    loader_path = loader_dir / "heapo.py"
    if not loader_path.exists():
        print("HEAPO Python loader not found; downloading it from GitHub")
        _download(HEAPO_LOADER_URL, loader_path, label="HEAPO Python loader")
    if str(loader_dir) not in sys.path:
        sys.path.insert(0, str(loader_dir))


def _run_with_args(function, arguments: list[str]) -> None:
    previous_argv = sys.argv
    try:
        sys.argv = arguments
        function()
    finally:
        sys.argv = previous_argv


def ensure_raw_data(
    data_root: Path, progress: Callable[[str], None] | None = None
) -> Path:
    """Download and extract HEAPO data when the requested folder is missing."""
    ensure_heapo_loader(data_root.parent)
    zip_path = data_root.parent / "heapo_data.zip"
    source_marker = data_root / "reports" / "protocols.csv"

    if not source_marker.exists():
        message = f"HEAPO data not found under {data_root}; downloading it from Zenodo"
        if progress:
            progress(message)
        else:
            print(message)
        _download(os.getenv("HEAPO_DATA_URL", DEFAULT_DATA_URL), zip_path, progress)
        if progress:
            progress("Download complete. Extracting HEAPO data...")
        else:
            print("Download complete. Extracting HEAPO data...")
        _extract(zip_path, data_root.parent)
        if not source_marker.exists():
            raise FileNotFoundError(
                "The HEAPO archive downloaded successfully, but reports/protocols.csv "
                f"was not found under {data_root}."
            )
    return data_root


def ensure_artifacts(
    artifacts_dir: Path, progress: Callable[[str], None] | None = None
) -> str:
    """Return a status message after ensuring data and models are available."""
    has_dataset = any(
        (artifacts_dir / filename).exists()
        for filename in ("dataset.parquet", "dataset.csv.gz")
    )
    if has_dataset and (artifacts_dir / "models.joblib").exists():
        return "Using the bundled dashboard artifacts."

    data_root = Path(os.getenv("HEAPO_DATA_DIR", "heapo_data"))
    data_root = data_root if data_root.is_absolute() else artifacts_dir.parent / data_root
    ensure_raw_data(data_root, progress)

    from prepare_data import main as prepare_main
    from train_models import main as train_main

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if progress:
        progress("Building dashboard artifacts and fitting models...")
    else:
        print("Building dashboard artifacts and fitting models...")
    _run_with_args(
        prepare_main,
        ["prepare_data.py", "--data-path", str(data_root), "--out", str(artifacts_dir)],
    )
    _run_with_args(
        train_main,
        ["train_models.py", "--artifacts", str(artifacts_dir)],
    )
    return "HEAPO data downloaded and dashboard artifacts built."