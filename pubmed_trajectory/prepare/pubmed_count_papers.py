#!/usr/bin/env python3
"""Count XML papers in PMC OA tar archives."""

import argparse
import tarfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMM_DIR = PROJECT_ROOT / "pmc_oa_comm_xml_2026"
DEFAULT_NONCOMM_DIR = PROJECT_ROOT / "pmc_oa_noncomm_xml_2026"
DEFAULT_OTHER_DIR = PROJECT_ROOT / "pmc_oa_other_xml_2026"


def iter_archives(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*.tar.gz"))


def count_xml_members(archive_path: Path) -> int:
    with tarfile.open(archive_path, "r:gz") as tar:
        return sum(1 for member in tar if member.isfile() and member.name.endswith(".xml"))


def count_directory(root: Path) -> Tuple[int, int]:
    archives = list(iter_archives(root))
    total = 0
    for archive_path in tqdm(archives, desc=f"Count {root.name}"):
        total += count_xml_members(archive_path)
    return len(archives), total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count how many XML papers are stored in PMC OA tar archives.")
    parser.add_argument("--comm-dir", default=DEFAULT_COMM_DIR)
    parser.add_argument("--noncomm-dir", default=DEFAULT_NONCOMM_DIR)
    parser.add_argument("--other-dir", default=DEFAULT_OTHER_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets: Dict[str, Path] = {
        "comm": Path(args.comm_dir),
        "noncomm": Path(args.noncomm_dir),
        "other": Path(args.other_dir),
    }

    grand_total = 0
    for label, root in targets.items():
        archive_count, paper_count = count_directory(root)
        grand_total += paper_count
        print(f"{label}: archives={archive_count}, papers={paper_count}, path={root}")
    print(f"overall_papers={grand_total}")


if __name__ == "__main__":
    main()
