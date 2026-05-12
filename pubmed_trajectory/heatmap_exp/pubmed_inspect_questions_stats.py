#!/usr/bin/env python3
"""Compute descriptive statistics for evaluated pubmed_inspect_questions trajectories."""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "pubmed_trajectory" / "heatmap_exp" / "results"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize evaluated pubmed_inspect_questions trajectories and visualize question active-year distributions."
    )
    parser.add_argument(
        "--inspect-root",
        default=RESULTS_ROOT / "inspect_questions",
        help="Root folder containing experiment_* outputs from pubmed_inspect_questions.py.",
    )
    parser.add_argument(
        "--output-root",
        default=RESULTS_ROOT / "inspect_questions_stats",
        help="Directory where statistics outputs will be written.",
    )
    parser.add_argument(
        "--model-spec",
        action="append",
        default=None,
        help="Optional model spec(s), e.g. azure:azure-gpt-5 or vllm:Qwen/Qwen2.5-7B-Instruct. If omitted, all detected evaluated model directories are analyzed.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=320,
        help="Figure DPI.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    value = value.strip().replace(":", "_").replace("/", "_")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("_") or "output"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def discover_model_slugs(inspect_root: Path) -> List[str]:
    model_slugs = set()
    for experiment_dir in sorted(inspect_root.glob("experiment_*")):
        if not experiment_dir.is_dir():
            continue
        for child in experiment_dir.iterdir():
            if child.is_dir() and (child / "evaluation_results.json").exists():
                model_slugs.add(child.name)
    return sorted(model_slugs)


def resolve_requested_models(inspect_root: Path, model_specs: Sequence[str] | None) -> List[Tuple[str, str]]:
    if model_specs:
        return [(spec, slugify(spec)) for spec in model_specs]
    return [(slug, slug) for slug in discover_model_slugs(inspect_root)]


def collect_model_rows(inspect_root: Path, model_slug: str) -> Dict[str, Any]:
    trajectory_rows: List[Dict[str, Any]] = []
    interval_rows: List[Dict[str, Any]] = []
    year_truth_counts: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {
            "prior_yes": 0,
            "prior_no": 0,
            "current_yes": 0,
            "current_no": 0,
            "total": 0,
        }
    )
    year_prediction_counts: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {
            "prior_yes": 0,
            "prior_no": 0,
            "prior_invalid": 0,
            "current_yes": 0,
            "current_no": 0,
            "current_invalid": 0,
            "total": 0,
        }
    )
    year_trajectory_sets: Dict[int, set] = defaultdict(set)

    for experiment_dir in sorted(inspect_root.glob("experiment_*")):
        if not experiment_dir.is_dir():
            continue
        eval_path = experiment_dir / model_slug / "evaluation_results.json"
        question_path = experiment_dir / "generated_questions.jsonl"
        if not eval_path.exists() or not question_path.exists():
            continue

        evaluation_rows = read_json(eval_path)
        question_rows = read_jsonl(question_path)
        active_rows = [row for row in question_rows if row.get("available") and row.get("Question")]
        active_eval_rows = [row for row in evaluation_rows if row.get("available") and row.get("Question")]
        if not active_rows:
            continue

        experiment_name = experiment_dir.name
        experiment_index = active_rows[0].get("experiment_index")
        years = sorted({int(row["question_target_year"]) for row in active_rows})
        trajectory_rows.append(
            {
                "experiment_dir": experiment_name,
                "experiment_index": experiment_index,
                "active_row_count": len(active_rows),
                "evaluated_row_count": len(evaluation_rows),
                "min_year": years[0],
                "max_year": years[-1],
                "year_span": years[-1] - years[0] + 1,
            }
        )

        grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
        for row in active_rows:
            key = (
                row.get("experiment_index"),
                row.get("pair_id"),
                row.get("difference_idx"),
                row.get("statement_source"),
            )
            grouped[key].append(row)
            year = int(row["question_target_year"])
            source = row["statement_source"]
            correct_choice = row.get("Answer", {}).get("Correct")
            truth_label = "yes" if correct_choice == "A" else "no" if correct_choice == "B" else "unknown"
            if truth_label in {"yes", "no"}:
                year_truth_counts[year][f"{source}_{truth_label}"] += 1
                year_truth_counts[year]["total"] += 1
            year_trajectory_sets[year].add(experiment_name)

        for row in active_eval_rows:
            year = int(row["question_target_year"])
            source = row["statement_source"]
            prediction = row.get("prediction")
            pred_label = "yes" if prediction == "A" else "no" if prediction == "B" else "invalid"
            year_prediction_counts[year][f"{source}_{pred_label}"] += 1
            year_prediction_counts[year]["total"] += 1

        for rows in grouped.values():
            ordered = sorted(rows, key=lambda row: int(row["question_target_year"]))
            active_years = [int(row["question_target_year"]) for row in ordered]
            interval_rows.append(
                {
                    "experiment_dir": experiment_name,
                    "experiment_index": ordered[0].get("experiment_index"),
                    "pair_id": ordered[0].get("pair_id"),
                    "difference_idx": ordered[0].get("difference_idx"),
                    "statement_source": ordered[0].get("statement_source"),
                    "target_pmcid": ordered[0].get("target_pmcid"),
                    "target_title": ordered[0].get("target_title"),
                    "start_year": active_years[0],
                    "end_year": active_years[-1],
                    "active_year_count": len(active_years),
                    "active_span_years": active_years[-1] - active_years[0] + 1,
                }
            )

    year_rows = []
    for year in sorted(year_truth_counts):
        year_rows.append(
            {
                "year": year,
                "prior_yes_rows": year_truth_counts[year]["prior_yes"],
                "prior_no_rows": year_truth_counts[year]["prior_no"],
                "current_yes_rows": year_truth_counts[year]["current_yes"],
                "current_no_rows": year_truth_counts[year]["current_no"],
                "prior_truth_total_rows": year_truth_counts[year]["prior_yes"] + year_truth_counts[year]["prior_no"],
                "current_truth_total_rows": year_truth_counts[year]["current_yes"] + year_truth_counts[year]["current_no"],
                "prior_pred_yes_rows": year_prediction_counts[year]["prior_yes"],
                "prior_pred_no_rows": year_prediction_counts[year]["prior_no"],
                "prior_pred_invalid_rows": year_prediction_counts[year]["prior_invalid"],
                "current_pred_yes_rows": year_prediction_counts[year]["current_yes"],
                "current_pred_no_rows": year_prediction_counts[year]["current_no"],
                "current_pred_invalid_rows": year_prediction_counts[year]["current_invalid"],
                "prior_prediction_total_rows": (
                    year_prediction_counts[year]["prior_yes"]
                    + year_prediction_counts[year]["prior_no"]
                    + year_prediction_counts[year]["prior_invalid"]
                ),
                "current_prediction_total_rows": (
                    year_prediction_counts[year]["current_yes"]
                    + year_prediction_counts[year]["current_no"]
                    + year_prediction_counts[year]["current_invalid"]
                ),
                "total_active_rows": year_truth_counts[year]["total"],
                "total_prediction_rows": year_prediction_counts[year]["total"],
                "trajectory_count": len(year_trajectory_sets[year]),
            }
        )

    return {
        "trajectory_rows": trajectory_rows,
        "interval_rows": interval_rows,
        "year_rows": year_rows,
    }


def build_summary_payload(model_label: str, data: Dict[str, Any]) -> Dict[str, Any]:
    interval_rows = data["interval_rows"]
    year_rows = data["year_rows"]
    trajectory_rows = data["trajectory_rows"]
    spans = [row["active_span_years"] for row in interval_rows]
    sorted_spans = sorted(spans)
    return {
        "model": model_label,
        "evaluated_trajectory_count": len(trajectory_rows),
        "logical_question_count": len(interval_rows),
        "min_year": min((row["year"] for row in year_rows), default=None),
        "max_year": max((row["year"] for row in year_rows), default=None),
        "mean_active_span_years": (sum(spans) / len(spans)) if spans else None,
        "median_active_span_years": sorted_spans[len(sorted_spans) // 2] if sorted_spans else None,
        "max_active_rows_in_a_year": max((row["total_active_rows"] for row in year_rows), default=0),
    }


def create_visualization(model_label: str, data: Dict[str, Any], output_path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    year_rows = data["year_rows"]
    interval_rows = data["interval_rows"]
    if not year_rows:
        return

    years = [row["year"] for row in year_rows]
    prior_yes_counts = [row["prior_yes_rows"] for row in year_rows]
    prior_no_counts = [row["prior_no_rows"] for row in year_rows]
    current_yes_counts = [row["current_yes_rows"] for row in year_rows]
    current_no_counts = [row["current_no_rows"] for row in year_rows]
    current_truth_total_counts = [row["current_truth_total_rows"] for row in year_rows]
    current_yes_ratios = [
        (yes / total) if total else 0.0 for yes, total in zip(current_yes_counts, current_truth_total_counts)
    ]
    current_no_ratios = [
        (no / total) if total else 0.0 for no, total in zip(current_no_counts, current_truth_total_counts)
    ]
    prior_pred_yes_counts = [row["prior_pred_yes_rows"] for row in year_rows]
    prior_pred_no_counts = [row["prior_pred_no_rows"] for row in year_rows]
    prior_pred_invalid_counts = [row["prior_pred_invalid_rows"] for row in year_rows]
    current_pred_yes_counts = [row["current_pred_yes_rows"] for row in year_rows]
    current_pred_no_counts = [row["current_pred_no_rows"] for row in year_rows]
    current_pred_invalid_counts = [row["current_pred_invalid_rows"] for row in year_rows]
    prior_pred_total_counts = [row["prior_prediction_total_rows"] for row in year_rows]
    current_pred_total_counts = [row["current_prediction_total_rows"] for row in year_rows]
    prior_pred_yes_ratios = [
        (yes / total) if total else 0.0 for yes, total in zip(prior_pred_yes_counts, prior_pred_total_counts)
    ]
    prior_pred_no_ratios = [
        (no / total) if total else 0.0 for no, total in zip(prior_pred_no_counts, prior_pred_total_counts)
    ]
    prior_pred_invalid_ratios = [
        (invalid / total) if total else 0.0 for invalid, total in zip(prior_pred_invalid_counts, prior_pred_total_counts)
    ]
    current_pred_yes_ratios = [
        (yes / total) if total else 0.0 for yes, total in zip(current_pred_yes_counts, current_pred_total_counts)
    ]
    current_pred_no_ratios = [
        (no / total) if total else 0.0 for no, total in zip(current_pred_no_counts, current_pred_total_counts)
    ]
    current_pred_invalid_ratios = [
        (invalid / total) if total else 0.0 for invalid, total in zip(current_pred_invalid_counts, current_pred_total_counts)
    ]
    trajectory_counts = [row["trajectory_count"] for row in year_rows]
    prior_spans = [row["active_span_years"] for row in interval_rows if row["statement_source"] == "prior"]
    current_spans = [row["active_span_years"] for row in interval_rows if row["statement_source"] == "current"]

    fig, axes = plt.subplots(6, 1, figsize=(15, 21), constrained_layout=True)
    fig.patch.set_facecolor("#f6f8fb")

    axes[0].set_facecolor("#ffffff")
    axes[0].plot(years, prior_yes_counts, color="#2f5f98", linewidth=2.4, marker="o", markersize=3.5, label="Ground truth: Yes")
    axes[0].plot(years, prior_no_counts, color="#c94f4f", linewidth=2.4, marker="o", markersize=3.5, label="Ground truth: No")
    axes[0].fill_between(years, prior_yes_counts, color="#2f5f98", alpha=0.12)
    axes[0].fill_between(years, prior_no_counts, color="#c94f4f", alpha=0.10)
    axes[0].set_title("Prior-Targeted Questions: Ground-Truth Distribution", loc="left", fontsize=14, weight="bold", color="#15202b")
    axes[0].set_ylabel("Question-year rows", color="#22303c")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", color="#d9e2ec", linewidth=1.0)
    axes[0].set_axisbelow(True)

    axes[1].set_facecolor("#ffffff")
    axes[1].plot(years, current_yes_ratios, color="#d96c0f", linewidth=2.4, marker="o", markersize=3.5, label="Ground truth: Yes")
    axes[1].plot(years, current_no_ratios, color="#7a5195", linewidth=2.4, marker="o", markersize=3.5, label="Ground truth: No")
    axes[1].fill_between(years, current_yes_ratios, color="#d96c0f", alpha=0.12)
    axes[1].fill_between(years, current_no_ratios, color="#7a5195", alpha=0.10)
    axes[1].set_title("Current-Targeted Questions: Ground-Truth Ratio", loc="left", fontsize=14, weight="bold", color="#15202b")
    axes[1].set_ylabel("Ground-truth ratio", color="#22303c")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", color="#d9e2ec", linewidth=1.0)
    axes[1].set_axisbelow(True)

    axes[2].set_facecolor("#ffffff")
    axes[2].plot(years, prior_pred_yes_ratios, color="#2f5f98", linewidth=2.4, marker="o", markersize=3.5, label="Predicted: Yes")
    axes[2].plot(years, prior_pred_no_ratios, color="#c94f4f", linewidth=2.4, marker="o", markersize=3.5, label="Predicted: No")
    axes[2].plot(years, prior_pred_invalid_ratios, color="#7f8c8d", linewidth=2.0, marker="o", markersize=3.0, label="Predicted: Invalid")
    axes[2].fill_between(years, prior_pred_yes_ratios, color="#2f5f98", alpha=0.12)
    axes[2].fill_between(years, prior_pred_no_ratios, color="#c94f4f", alpha=0.10)
    axes[2].set_title("Prior-Targeted Questions: Prediction Ratio", loc="left", fontsize=14, weight="bold", color="#15202b")
    axes[2].set_ylabel("Prediction ratio", color="#22303c")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].legend(frameon=False)
    axes[2].grid(axis="y", color="#d9e2ec", linewidth=1.0)
    axes[2].set_axisbelow(True)

    axes[3].set_facecolor("#ffffff")
    axes[3].plot(years, current_pred_yes_ratios, color="#d96c0f", linewidth=2.4, marker="o", markersize=3.5, label="Predicted: Yes")
    axes[3].plot(years, current_pred_no_ratios, color="#7a5195", linewidth=2.4, marker="o", markersize=3.5, label="Predicted: No")
    axes[3].plot(years, current_pred_invalid_ratios, color="#7f8c8d", linewidth=2.0, marker="o", markersize=3.0, label="Predicted: Invalid")
    axes[3].fill_between(years, current_pred_yes_ratios, color="#d96c0f", alpha=0.12)
    axes[3].fill_between(years, current_pred_no_ratios, color="#7a5195", alpha=0.10)
    axes[3].set_title("Current-Targeted Questions: Prediction Ratio", loc="left", fontsize=14, weight="bold", color="#15202b")
    axes[3].set_ylabel("Prediction ratio", color="#22303c")
    axes[3].set_ylim(0.0, 1.0)
    axes[3].legend(frameon=False)
    axes[3].grid(axis="y", color="#d9e2ec", linewidth=1.0)
    axes[3].set_axisbelow(True)

    axes[4].set_facecolor("#ffffff")
    axes[4].plot(years, trajectory_counts, color="#2a9d8f", linewidth=2.2)
    axes[4].fill_between(years, trajectory_counts, color="#2a9d8f", alpha=0.16)
    axes[4].set_title("Trajectory Coverage By Year", loc="left", fontsize=14, weight="bold", color="#15202b")
    axes[4].set_ylabel("Trajectories contributing", color="#22303c")
    axes[4].grid(axis="y", color="#d9e2ec", linewidth=1.0)
    axes[4].set_axisbelow(True)

    axes[5].set_facecolor("#ffffff")
    max_span = max(prior_spans + current_spans) if (prior_spans or current_spans) else 1
    bins = np.arange(1, max_span + 2) - 0.5
    if prior_spans:
        axes[5].hist(prior_spans, bins=bins, alpha=0.7, color="#4c78a8", label="Prior statements")
    if current_spans:
        axes[5].hist(current_spans, bins=bins, alpha=0.7, color="#f58518", label="Current statements")
    axes[5].set_title("Active Span Length Distribution", loc="left", fontsize=14, weight="bold", color="#15202b")
    axes[5].set_xlabel("Active span length (years)", color="#22303c")
    axes[5].set_ylabel("Logical questions", color="#22303c")
    axes[5].grid(axis="y", color="#d9e2ec", linewidth=1.0)
    axes[5].set_axisbelow(True)
    if prior_spans or current_spans:
        axes[5].legend(frameon=False)

    fig.suptitle(
        f"Question Activity Statistics | {model_label}",
        x=0.055,
        y=0.995,
        ha="left",
        fontsize=18,
        weight="bold",
        color="#15202b",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    inspect_root = Path(args.inspect_root)
    output_root = Path(args.output_root)
    model_targets = resolve_requested_models(inspect_root, args.model_spec)
    if not model_targets:
        raise ValueError(f"No evaluated model directories found under {inspect_root}")

    for model_label, model_slug in model_targets:
        data = collect_model_rows(inspect_root, model_slug)
        model_output_dir = output_root / model_slug
        model_output_dir.mkdir(parents=True, exist_ok=True)

        write_csv(
            model_output_dir / "trajectory_summary.csv",
            data["trajectory_rows"],
            ["experiment_dir", "experiment_index", "active_row_count", "evaluated_row_count", "min_year", "max_year", "year_span"],
        )
        write_csv(
            model_output_dir / "question_intervals.csv",
            data["interval_rows"],
            [
                "experiment_dir",
                "experiment_index",
                "pair_id",
                "difference_idx",
                "statement_source",
                "target_pmcid",
                "target_title",
                "start_year",
                "end_year",
                "active_year_count",
                "active_span_years",
            ],
        )
        write_csv(
            model_output_dir / "year_distribution.csv",
            data["year_rows"],
            [
                "year",
                "prior_yes_rows",
                "prior_no_rows",
                "current_yes_rows",
                "current_no_rows",
                "prior_truth_total_rows",
                "current_truth_total_rows",
                "prior_pred_yes_rows",
                "prior_pred_no_rows",
                "prior_pred_invalid_rows",
                "current_pred_yes_rows",
                "current_pred_no_rows",
                "current_pred_invalid_rows",
                "prior_prediction_total_rows",
                "current_prediction_total_rows",
                "total_active_rows",
                "total_prediction_rows",
                "trajectory_count",
            ],
        )

        summary_payload = build_summary_payload(model_label, data)
        with (model_output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, indent=2, ensure_ascii=False)

        create_visualization(model_label, data, model_output_dir / "activity_statistics.png", args.dpi)
        print(f"Saved statistics for {model_label}: {model_output_dir}")


if __name__ == "__main__":
    main()
