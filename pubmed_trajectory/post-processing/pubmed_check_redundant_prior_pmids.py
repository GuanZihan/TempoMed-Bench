import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "comm_guideline_trajectory_2026_relaxed_augmented"


def normalize_pmid(value: Any) -> str | None:
    if value in (None, "", 0, "0"):
        return None
    text = str(value).strip()
    return text or None


def iter_json_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*/*.json"))


def find_duplicate_prior_pmids(payload: Dict[str, Any]) -> Dict[str, List[int]]:
    seen: Dict[str, List[int]] = {}
    for idx, prior in enumerate(payload.get("prior_guidelines") or []):
        if not isinstance(prior, dict):
            continue
        pmid = normalize_pmid(prior.get("PMID"))
        if pmid is None:
            continue
        seen.setdefault(pmid, []).append(idx)
    return {pmid: indices for pmid, indices in seen.items() if len(indices) > 1}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether prior_guidelines contain redundant entries with the same nonzero PMID."
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Root directory containing trajectory JSON files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON report path. Defaults to <input-dir>/redundant_prior_pmids_report.json.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output) if args.output else input_dir / "redundant_prior_pmids_report.json"

    rows = []
    paths = list(iter_json_files(input_dir))
    for path in tqdm(paths, desc="Check duplicate prior PMIDs"):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        duplicates = find_duplicate_prior_pmids(payload)
        if not duplicates:
            continue
        row = {
            "file": str(path),
            "relative_file": str(path.relative_to(input_dir)),
            "current_pmid": payload.get("PMID"),
            "title": payload.get("Title"),
            "duplicate_prior_pmids": duplicates,
        }
        rows.append(row)
        print(row["file"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dir": str(input_dir),
                "scanned_files": len(paths),
                "files_with_duplicate_prior_pmids": len(rows),
                "rows": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Scanned files: {len(paths)}")
    print(f"Files with duplicate prior PMIDs: {len(rows)}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
