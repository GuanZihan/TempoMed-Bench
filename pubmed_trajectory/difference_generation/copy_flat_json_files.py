#!/usr/bin/env python3
"""Copy nested JSON files into a single flat output directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy all JSON files found under the source directory into a new "
            "directory without recreating the source subdirectories."
        )
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Directory to scan recursively for JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where copied JSON files will be placed in a single layer.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing file in the output directory if names collide.",
    )
    return parser.parse_args()


def copy_flat_json_files(source_dir: Path, output_dir: Path, overwrite: bool) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    for json_path in sorted(source_dir.rglob("*.json")):
        if not json_path.is_file():
            continue

        destination = output_dir / json_path.name
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing file: {destination}. "
                "Use --overwrite if that is intended."
            )

        shutil.copy2(json_path, destination)
        copied_count += 1

    return copied_count


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_dir}")

    copied_count = copy_flat_json_files(source_dir, output_dir, args.overwrite)
    print(f"Copied {copied_count} JSON files to {output_dir}")


if __name__ == "__main__":
    main()
