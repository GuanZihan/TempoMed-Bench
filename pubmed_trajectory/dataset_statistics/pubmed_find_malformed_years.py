import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMM_DIR = PROJECT_ROOT / "comm_guideline_trajectory_2026_relaxed_augmented_year_calibrated"
DEFAULT_NONCOMM_DIR = PROJECT_ROOT / "noncomm_guideline_trajectory_2026_relaxed_augmented_year_calibrated"
DEFAULT_OTHER_DIR = PROJECT_ROOT / "other_guideline_trajectory_2026_relaxed_augmented_year_calibrated"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "pubmed_trajectory" / "dataset_statistics" / "results" / "malformed_year_trajectories.jsonl"
MIN_VALID_YEAR = 1900
MAX_VALID_YEAR = 2030
SUBSETS = ["comm", "noncomm", "other"]


def iter_json_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*/*.json"))


def classify_year_value(value: Any) -> Optional[Dict[str, Any]]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return {"kind": "bool", "raw": value}
    if isinstance(value, int):
        parsed = value
    else:
        stripped = str(value).strip()
        try:
            parsed = int(stripped)
        except Exception:
            return {"kind": "non_integer", "raw": value}
    if parsed == 0:
        return {"kind": "zero", "raw": value, "parsed": parsed}
    if parsed < MIN_VALID_YEAR:
        return {"kind": "too_small", "raw": value, "parsed": parsed}
    if parsed > MAX_VALID_YEAR:
        return {"kind": "too_large", "raw": value, "parsed": parsed}
    return None


def scan_subset(subset: str, root: Path) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    paths = list(iter_json_files(root))
    for path in tqdm(paths, desc=f"Scan {subset} trajectories"):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        issues: List[Dict[str, Any]] = []
        current_issue = classify_year_value(payload.get("year_of_current_guidance"))
        if current_issue is not None:
            issues.append(
                {
                    "location": "current",
                    "field": "year_of_current_guidance",
                    **current_issue,
                }
            )
        for idx, prior in enumerate(payload.get("prior_guidelines") or []):
            if not isinstance(prior, dict):
                continue
            prior_issue = classify_year_value(prior.get("year"))
            if prior_issue is not None:
                issues.append(
                    {
                        "location": "prior",
                        "prior_index": idx,
                        "field": "prior_guidelines.year",
                        "title": prior.get("title"),
                        **prior_issue,
                    }
                )
        if issues:
            findings.append(
                {
                    "subset": subset,
                    "file_path": str(path),
                    "relative_path": str(path.relative_to(root)),
                    "title": payload.get("Title"),
                    "pmid": payload.get("PMID"),
                    "issue_count": len(issues),
                    "issues": issues,
                }
            )
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Find trajectories with malformed year fields.")
    parser.add_argument("--comm-dir", default=DEFAULT_COMM_DIR)
    parser.add_argument("--noncomm-dir", default=DEFAULT_NONCOMM_DIR)
    parser.add_argument("--other-dir", default=DEFAULT_OTHER_DIR)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    all_findings: List[Dict[str, Any]] = []
    all_findings.extend(scan_subset("comm", Path(args.comm_dir)))
    all_findings.extend(scan_subset("noncomm", Path(args.noncomm_dir)))
    all_findings.extend(scan_subset("other", Path(args.other_dir)))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in all_findings:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Malformed trajectories: {len(all_findings)}")
    print(f"Saved report to: {output_path}")
    for subset in SUBSETS:
        subset_count = sum(1 for row in all_findings if row["subset"] == subset)
        print(f"  {subset}: {subset_count}")


if __name__ == "__main__":
    main()
