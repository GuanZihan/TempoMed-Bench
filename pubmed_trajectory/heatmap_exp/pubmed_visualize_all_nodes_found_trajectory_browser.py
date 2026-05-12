import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from ui_utils import render_html, render_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "pubmed_trajectory" / "heatmap_exp" / "results"
DEFAULT_CLEAN_SUMMARY_JSON = RESULTS_ROOT / "trajectory_browser_2026_relaxed_clean_summary.json"
DEFAULT_MISSING_SUMMARY_JSON = RESULTS_ROOT / "trajectory_missing_papers_clean_summary.json"
DEFAULT_OUTPUT_HTML = RESULTS_ROOT / "trajectory_browser_all_nodes_found.html"
DEFAULT_OUTPUT_JSON = RESULTS_ROOT / "trajectory_browser_all_nodes_found_summary.json"
DEFAULT_OUTPUT_REPORT = RESULTS_ROOT / "trajectory_browser_all_nodes_found_report.md"


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_all_nodes_found_pmcids(missing_summary_path):
    data = load_json(missing_summary_path)
    return {
        str(row.get("pmc_id"))
        for row in data.get("results", [])
        if row.get("all_nodes_found") and row.get("pmc_id")
    }


def summarize_subset(clean_summary, allowed_pmcids):
    records = [
        record for record in clean_summary.get("records", [])
        if str(record.get("pmc_id")) in allowed_pmcids
    ]

    by_category = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(record)

    category_summaries = []
    for category_key, grouped_records in sorted(by_category.items()):
        years = [record["year"] for record in grouped_records if isinstance(record.get("year"), int)]
        depths = [record.get("prior_count", 0) for record in grouped_records]
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

    year_histogram = Counter(record["year_bucket"] for record in records)
    depth_histogram = Counter(str(record.get("prior_count", 0)) for record in records)

    return {
        "records": records,
        "overall": {
            "total_clean_trajectories": len(records),
            "definition": "Clean trajectories whose every node is found in the local extracted PMC OA database.",
        },
        "category_summaries": category_summaries,
        "year_histogram": dict(sorted(year_histogram.items())),
        "prior_depth_histogram": dict(sorted(depth_histogram.items(), key=lambda item: int(item[0]))),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML browser for clean trajectories whose every node is found in the local database."
    )
    parser.add_argument(
        "--clean-summary-json",
        default=DEFAULT_CLEAN_SUMMARY_JSON,
        help="Path to the clean trajectory browser summary JSON.",
    )
    parser.add_argument(
        "--missing-summary-json",
        default=DEFAULT_MISSING_SUMMARY_JSON,
        help="Path to the missing-papers summary JSON that contains all_nodes_found.",
    )
    parser.add_argument(
        "--output-html",
        default=DEFAULT_OUTPUT_HTML,
        help="Path to the output HTML file.",
    )
    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT_JSON,
        help="Path to the output summary JSON file.",
    )
    parser.add_argument(
        "--output-report",
        default=DEFAULT_OUTPUT_REPORT,
        help="Path to the output Markdown report.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    clean_summary = load_json(args.clean_summary_json)
    allowed_pmcids = load_all_nodes_found_pmcids(args.missing_summary_json)
    summary = summarize_subset(clean_summary, allowed_pmcids)

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
    print(f"Trajectories with all nodes found: {summary['overall']['total_clean_trajectories']}")


if __name__ == "__main__":
    main()
