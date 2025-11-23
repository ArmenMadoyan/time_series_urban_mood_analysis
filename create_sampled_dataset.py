"""
Utility script to create a sampled copy of the sentiment dataset.

This keeps the original `dataverse_files` directory untouched by copying its
structure to a new destination directory and optionally sampling a percentage of
rows from every CSV file. Non-CSV assets (if any) are copied verbatim.
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Iterable

import pandas as pd


def _iter_files(source_root: str) -> Iterable[tuple[str, str]]:
    """Yield (dirpath, filename) pairs for every file under source_root."""
    for dirpath, _, filenames in os.walk(source_root):
        for filename in filenames:
            yield dirpath, filename


def _make_destination_path(source_root: str, dest_root: str, dirpath: str, filename: str) -> str:
    """Compute the destination path mirroring the source tree."""
    relative_dir = os.path.relpath(dirpath, source_root)
    dest_dir = os.path.join(dest_root, relative_dir)
    os.makedirs(dest_dir, exist_ok=True)
    return os.path.join(dest_dir, filename)


def _copy_or_sample_csv(source_path: str, dest_path: str, sample_fraction: float, random_state: int) -> None:
    """Copy a CSV file, optionally sampling a percentage of rows."""
    df = pd.read_csv(source_path)
    if 0 < sample_fraction < 1:
        df = df.sample(frac=sample_fraction, random_state=random_state)
    df.to_csv(dest_path, index=False)


def create_sampled_dataset(
    source_root: str,
    dest_root: str,
    sample_fraction: float,
    random_state: int = 42,
) -> None:
    """
    Copy source_root into dest_root while sampling CSV rows.

    Args:
        source_root: Directory that contains the full dataset.
        dest_root: Directory where the sampled copy should live.
        sample_fraction: Fraction of rows to keep from each CSV (0-1].
        random_state: RNG seed to keep sampling deterministic across runs.
    """
    if not os.path.isdir(source_root):
        raise ValueError(f"Source directory does not exist: {source_root}")
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in the (0, 1] interval.")

    if os.path.exists(dest_root):
        shutil.rmtree(dest_root)

    for dirpath, filename in _iter_files(source_root):
        source_path = os.path.join(dirpath, filename)
        dest_path = _make_destination_path(source_root, dest_root, dirpath, filename)

        if filename.lower().endswith(".csv"):
            _copy_or_sample_csv(source_path, dest_path, sample_fraction, random_state)
        else:
            shutil.copy2(source_path, dest_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a sampled copy of the sentiment dataset.")
    parser.add_argument(
        "--source",
        default="dataverse_files",
        help="Path to the original dataset directory.",
    )
    parser.add_argument(
        "--destination",
        default="dataverse_files_sampled",
        help="Where to place the sampled dataset.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.2,
        help="Fraction of rows to keep from each CSV (0-1].",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_sampled_dataset(args.source, args.destination, args.fraction, args.random_state)
    print(
        f"Sampled dataset created at {args.destination} "
        f"using fraction={args.fraction} and random_state={args.random_state}."
    )


if __name__ == "__main__":
    main()
