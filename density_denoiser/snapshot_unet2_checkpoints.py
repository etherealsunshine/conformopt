from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, suffix=".pt", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def snapshot(model_directory: Path) -> Path | None:
    progress_path = model_directory / "progress.json"
    checkpoint_path = model_directory / "denoiser_last.pt"
    if not progress_path.exists() or not checkpoint_path.exists():
        return None
    progress = json.loads(progress_path.read_text())
    epoch = int(progress.get("epoch", progress.get("last_epoch", -1)))
    if epoch < 0:
        return None
    destination = model_directory / f"epoch_{epoch:03d}.pt"
    if destination.exists():
        return None
    _atomic_copy(checkpoint_path, destination)
    print(json.dumps({
        "epoch": epoch,
        "snapshot": str(destination),
        "bytes": destination.stat().st_size,
    }), flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot completed U-Net 2.0 epochs")
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--training-pid-file", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=15.0)
    args = parser.parse_args()
    pid = int(args.training_pid_file.read_text().strip())
    while _process_alive(pid):
        snapshot(args.model_directory)
        time.sleep(args.interval)
    snapshot(args.model_directory)


if __name__ == "__main__":
    main()
