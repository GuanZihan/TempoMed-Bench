import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "noncomm_guideline_trajectory_2026_relaxed_augmented_filtered"


@dataclass
class TrajectoryRecord:
    path: Path
    relative_path: Path
    current_pmid: Optional[str]
    year: Optional[int]
    topic: str
    title: str
    prior_pmids: List[str]
    payload: Dict[str, Any]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(normalize_text(v) for v in value if normalize_text(v))
    return str(value).strip()


def normalize_pmid(value: Any) -> Optional[str]:
    if value in (None, "", 0, "0"):
        return None
    text = str(value).strip()
    return text or None


def to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except Exception:
        return None


def iter_json_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*/*.json"))


def load_record(path: Path, input_dir: Path) -> TrajectoryRecord:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    prior_pmids = []
    for prior in payload.get("prior_guidelines") or []:
        if not isinstance(prior, dict):
            continue
        pmid = normalize_pmid(prior.get("PMID"))
        if pmid:
            prior_pmids.append(pmid)
    return TrajectoryRecord(
        path=path,
        relative_path=path.relative_to(input_dir),
        current_pmid=normalize_pmid(payload.get("PMID")),
        year=to_int(payload.get("year_of_current_guidance")),
        topic=normalize_text(payload.get("Topic")),
        title=normalize_text(payload.get("Title")),
        prior_pmids=prior_pmids,
        payload=payload,
    )


def collect_records(input_dir: Path) -> List[TrajectoryRecord]:
    paths = list(iter_json_files(input_dir))
    return [load_record(path, input_dir) for path in tqdm(paths, desc="Load trajectories")]


def build_newer_prior_map(records: List[TrajectoryRecord]) -> Dict[str, List[TrajectoryRecord]]:
    prior_map: Dict[str, List[TrajectoryRecord]] = {}
    for record in records:
        for prior_pmid in record.prior_pmids:
            prior_map.setdefault(prior_pmid, []).append(record)
    return prior_map


def find_drop_candidates(records: List[TrajectoryRecord]) -> List[Dict[str, Any]]:
    prior_map = build_newer_prior_map(records)
    flagged: List[Dict[str, Any]] = []
    for record in records:
        if not record.current_pmid:
            continue
        referencing = prior_map.get(record.current_pmid, [])
        newer_refs = []
        for ref in referencing:
            if ref.path == record.path:
                continue
            if record.year is not None and ref.year is not None and ref.year <= record.year:
                continue
            newer_refs.append(ref)
        if not newer_refs:
            continue
        flagged.append(
            {
                "file": str(record.path),
                "relative_file": str(record.relative_path),
                "current_pmid": record.current_pmid,
                "year": record.year,
                "topic": record.topic,
                "title": record.title,
                "referenced_by": [
                    {
                        "file": str(ref.path),
                        "relative_file": str(ref.relative_path),
                        "year": ref.year,
                        "topic": ref.topic,
                        "title": ref.title,
                    }
                    for ref in sorted(newer_refs, key=lambda item: (item.year or 9999, str(item.relative_path)))
                ],
            }
        )
    flagged.sort(key=lambda row: (row["year"] or 9999, row["relative_file"]))
    return flagged


def write_report(input_dir: Path, flagged: List[Dict[str, Any]]) -> Path:
    report_path = input_dir / "nonterminal_trajectories_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dir": str(input_dir),
                "flagged_count": len(flagged),
                "rows": flagged,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return report_path


def copy_filtered_tree(input_dir: Path, output_dir: Path, flagged: List[Dict[str, Any]]) -> Path:
    flagged_relpaths = {row["relative_file"] for row in flagged}
    output_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    paths = list(iter_json_files(input_dir))
    for path in tqdm(paths, desc="Copy filtered trajectories"):
        rel = str(path.relative_to(input_dir))
        if rel in flagged_relpaths:
            continue
        dst = output_dir / path.relative_to(input_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        kept += 1
    summary_path = output_dir / "filter_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "removed_count": len(flagged),
                "kept_count": kept,
                "removed_relative_files": sorted(flagged_relpaths),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter out trajectory JSON files whose current PMID already appears as a prior PMID in a newer trajectory."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Input trajectory directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for a filtered copy. If omitted, the script only writes a report.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    records = collect_records(input_dir)
    flagged = find_drop_candidates(records)
    report_path = write_report(input_dir, flagged)

    print(f"Scanned files: {len(records)}")
    print(f"Flagged nonterminal trajectories: {len(flagged)}")
    print(f"Report: {report_path}")

    if output_dir is not None:
        summary_path = copy_filtered_tree(input_dir, output_dir, flagged)
        print(f"Filtered copy written to: {output_dir}")
        print(f"Filter summary: {summary_path}")


if __name__ == "__main__":
    main()
