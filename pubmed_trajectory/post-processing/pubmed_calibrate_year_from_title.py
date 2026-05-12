import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET_DIR = PROJECT_ROOT / "comm_guideline_trajectory_2026_relaxed_augmented_filtered"
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
YEAR_RANGE_RE = re.compile(r"\b(19|20)\d{2}\s*[-/–—]\s*(19|20)\d{2}\b")
YEAR_IN_PARENS_RE = re.compile(r"\((19|20)\d{2}\)")
INVALID_TITLE_KEYWORDS = [
    "commentary",
    "synopsis",
    "systematic review",
    "meta-analysis",
    "overview",
    "correction",
    "correction to",
    "author correction",
]


def iter_json_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*/*.json"))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def title_supports_year_calibration(title: Any) -> bool:
    if title is None:
        return False
    text = str(title)
    lowered = text.lower()
    if any(keyword in lowered for keyword in INVALID_TITLE_KEYWORDS):
        return False
    return "edition" in lowered or YEAR_IN_PARENS_RE.search(text) is not None


def extract_year_from_title(title: Any) -> Optional[int]:
    if title is None:
        return None
    text = str(title)
    range_matches = YEAR_RANGE_RE.findall(text)
    if range_matches:
        return None
    match = YEAR_RE.search(text)
    if not match:
        return None
    return int(match.group(0))


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


def within_two_years(existing_year: Optional[int], title_year: Optional[int]) -> bool:
    if title_year is None:
        return False
    if existing_year is None:
        return True
    return abs(title_year - existing_year) <= 2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate guideline years from years appearing in guideline titles."
    )
    parser.add_argument(
        "--target-dir",
        default=DEFAULT_TARGET_DIR,
        help="Root directory containing trajectory JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for a calibrated copy. If omitted, files are modified in place.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without modifying files.",
    )
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    paths = list(iter_json_files(target_dir))

    updated_files = 0
    updated_current = 0
    updated_prior = 0

    for path in tqdm(paths, desc="Calibrate years from title"):
        payload = load_json(path)
        changed = False

        current_title = payload.get("Title")
        current_title_year = extract_year_from_title(current_title) if title_supports_year_calibration(current_title) else None
        current_year = to_int(payload.get("year_of_current_guidance"))
        if within_two_years(current_year, current_title_year) and current_title_year != current_year:
            if args.dry_run:
                print(f"CURRENT\t{path}\t{current_year}\t{current_title_year}\t{payload.get('Title')}")
            else:
                payload["year_of_current_guidance"] = current_title_year
            changed = True
            updated_current += 1

        priors = payload.get("prior_guidelines") or []
        for idx, prior in enumerate(priors):
            if not isinstance(prior, dict):
                continue
            prior_title = prior.get("title")
            prior_title_year = extract_year_from_title(prior_title) if title_supports_year_calibration(prior_title) else None
            prior_year = to_int(prior.get("year"))
            if within_two_years(prior_year, prior_title_year) and prior_title_year != prior_year:
                if args.dry_run:
                    print(f"PRIOR\t{path}\tindex={idx}\t{prior_year}\t{prior_title_year}\t{prior.get('title')}")
                else:
                    prior["year"] = prior_title_year
                changed = True
                updated_prior += 1

        destination = path if output_dir is None else output_dir / path.relative_to(target_dir)
        if not args.dry_run and (changed or output_dir is not None):
            write_json(destination, payload)
        if changed:
            updated_files += 1

    print(f"Scanned files: {len(paths)}")
    print(f"Updated files: {updated_files}")
    print(f"Updated current guideline years: {updated_current}")
    print(f"Updated prior guideline years: {updated_prior}")
    print(f"Mode: {'dry-run' if args.dry_run else ('copy' if output_dir is not None else 'write')}")
    if output_dir is not None:
        print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
