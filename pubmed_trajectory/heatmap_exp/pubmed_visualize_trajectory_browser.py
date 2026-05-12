import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from ui_utils import render_html, render_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "pubmed_trajectory" / "heatmap_exp" / "results"
DEFAULT_INPUTS = [
    (
        "comm",
        PROJECT_ROOT / "comm_guideline_trajectory_2026_relaxed_augmented_year_calibrated",
        "Commercial",
        "#0f766e",
    ),
    (
        "noncomm",
        PROJECT_ROOT / "noncomm_guideline_trajectory_2026_relaxed_augmented_year_calibrated",
        "Non-commercial",
        "#b45309",
    ),
    (
        "other",
        PROJECT_ROOT / "other_guideline_trajectory_2026_relaxed_augmented_year_calibrated",
        "Other",
        "#7c3aed",
    ),
]


def normalize_text(value):
    text = str(value or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def compact_text(value, fallback="Unknown"):
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "null"}:
        return fallback
    return re.sub(r"\s+", " ", text)


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def year_bucket(year):
    if isinstance(year, int) and 1800 <= year <= 2100:
        return f"{year // 10 * 10}s"
    return "Unknown"


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def build_record(category_key, category_label, color, path, data):
    topic = compact_text(data.get("Topic"))
    title = compact_text(data.get("Title"))
    year = to_int(data.get("year_of_current_guidance"))
    pmid = to_int(data.get("PMID"))
    organizations = [
        compact_text(org, "") for org in (data.get("Organization") or []) if compact_text(org, "")
    ]
    priors = []
    for prior in data.get("prior_guidelines") or []:
        priors.append(
            {
                "year": to_int(prior.get("year")),
                "pmid": to_int(prior.get("PMID")),
                "organization": compact_text(prior.get("Organization")),
                "title": compact_text(prior.get("title")),
                "reason": compact_text(prior.get("reason")),
            }
        )
    return {
        "category": category_key,
        "category_label": category_label,
        "color": color,
        "path": str(path),
        "pmc_id": path.stem,
        "pmid": pmid,
        "topic": topic,
        "title": title,
        "year": year,
        "year_bucket": year_bucket(year),
        "organizations": organizations,
        "organization_count": len(organizations),
        "prior_guidelines": priors,
        "prior_count": len(priors),
        "node_count": len(priors) + 1,
    }


def load_records():
    records = []
    for category_key, input_dir, category_label, color in DEFAULT_INPUTS:
        root = Path(input_dir)
        if not root.exists():
            raise FileNotFoundError(f"Input directory not found: {root}")
        for path in sorted(root.rglob("*.json")):
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            records.append(build_record(category_key, category_label, color, path, data))
    return records


def classify_record(record):
    # check if there any issues with the record
    issues = []
    if record["prior_count"] < 1:
        issues.append("no_prior_guidelines")
    if not (isinstance(record["year"], int) and 1800 <= record["year"] <= 2100):
        issues.append("invalid_current_year")
    if normalize_text(record["title"]) in {"", "unknown"}:
        issues.append("missing_or_placeholder_title")
    if normalize_text(record["topic"]) in {"", "unknown"}:
        issues.append("missing_or_placeholder_topic")
    if not record["organizations"]:
        issues.append("missing_current_organization")

    seen_titles = Counter()
    for prior in record["prior_guidelines"]:
        title_key = normalize_text(prior["title"])
        seen_titles[title_key] += 1
        if prior["pmid"] in (None, 0):
            issues.append("missing_prior_pmid")
        if normalize_text(prior["title"]) in {"", "unknown"}:
            issues.append("missing_or_placeholder_prior_title")
        if normalize_text(prior["organization"]) in {"", "unknown"}:
            issues.append("missing_prior_organization")
        if prior["year"] == record["year"]:
            issues.append("same_year_prior")
        if isinstance(prior["year"], int) and isinstance(record["year"], int) and prior["year"] > record["year"]:
            issues.append("future_prior")

    if any(key not in {"", "unknown"} and count > 1 for key, count in seen_titles.items()):
        issues.append("duplicate_prior_title_within_record")

    unique_issues = sorted(set(issues))
    record["issues"] = unique_issues
    # only keep clean trajectory with two and more nodes
    record["is_clean"] = not unique_issues and record["node_count"] >= 2
    return record


def summarize_clean_records(records):
    clean_records = [classify_record(record) for record in records]
    clean_records = [record for record in clean_records if record["is_clean"]]

    by_category = defaultdict(list)
    for record in clean_records:
        by_category[record["category"]].append(record)

    category_summaries = []
    for category_key, grouped_records in sorted(by_category.items()):
        years = [record["year"] for record in grouped_records if isinstance(record["year"], int)]
        depths = [record["prior_count"] for record in grouped_records]
        category_summaries.append(
            {
                "category": category_key,
                "label": grouped_records[0]["category_label"],
                "color": grouped_records[0]["color"],
                "total_records": len(grouped_records),
                "median_prior_depth": median(depths) if depths else 0,
                "max_prior_depth": max(depths) if depths else 0,
                "min_year": min(years) if years else None,
                "max_year": max(years) if years else None,
            }
        )

    year_histogram = Counter(record["year_bucket"] for record in clean_records)
    depth_histogram = Counter(str(record["prior_count"]) for record in clean_records)

    return {
        "records": clean_records,
        "overall": {
            "total_clean_trajectories": len(clean_records),
            "definition": "At least 2 nodes, no incompleteness issues, no chronology issues, and no within-trajectory redundancy.",
        },
        "category_summaries": category_summaries,
        "year_histogram": dict(sorted(year_histogram.items())),
        "prior_depth_histogram": dict(sorted(depth_histogram.items(), key=lambda item: int(item[0]))),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate an interactive HTML browser for clean guideline trajectories only.")
    parser.add_argument(
        "--output-html",
        default=RESULTS_ROOT / "trajectory_browser_2026_relaxed_clean.html",
        help="Path to the output HTML file.",
    )
    parser.add_argument(
        "--output-json",
        default=RESULTS_ROOT / "trajectory_browser_2026_relaxed_clean_summary.json",
        help="Path to the output summary JSON file.",
    )
    parser.add_argument(
        "--output-report",
        default=RESULTS_ROOT / "trajectory_browser_2026_relaxed_clean_report.md",
        help="Path to the output Markdown report.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = summarize_clean_records(load_records())
    html_path = Path(args.output_html)
    json_path = Path(args.output_json)
    report_path = Path(args.output_report)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(render_html(summary), encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(render_report(summary), encoding="utf-8")
    print(f"Wrote HTML to {html_path}")
    print(f"Wrote JSON summary to {json_path}")
    print(f"Wrote Markdown report to {report_path}")
    print(f"Clean trajectories: {summary['overall']['total_clean_trajectories']}")


if __name__ == "__main__":
    main()
