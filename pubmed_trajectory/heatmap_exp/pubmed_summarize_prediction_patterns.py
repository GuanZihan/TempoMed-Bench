#!/usr/bin/env python3
"""Summarize per-model prediction-pattern percentages over per-difference evaluation results."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEATMAP_ROOT = PROJECT_ROOT / "pubmed_trajectory" / "heatmap_exp"
RESULTS_ROOT = HEATMAP_ROOT / "results"
PATTERN_ORDER = [
    "correct_zigzag_transition",
    "misaligned_zigzag",
    "prior_only_yes",
    "current_only_yes",
    "prior_and_current_yes",
    "all_no",
    "other",
]
PATTERN_COLORS = {
    "correct_zigzag_transition": "#4c9f70",
    "misaligned_zigzag": "#1f4e79",
    "prior_only_yes": "#c26d1a",
    "current_only_yes": "#2b7a78",
    "prior_and_current_yes": "#5b8c2a",
    "all_no": "#7a1f2b",
    "other": "#8e8e8e",
}
PATTERN_LABELS = {
    "correct_zigzag_transition": "Correct Zig-zag Transition",
    "misaligned_zigzag": "Misaligned Zig-zag",
    "prior_only_yes": "Prior-only Yes",
    "current_only_yes": "Current-only Yes",
    "prior_and_current_yes": "Prior+Current Yes",
    "all_no": "All No",
    "other": "Other",
}
LATEX_PATTERN_ORDER = [
    "prior_and_current_yes",
    "all_no",
    "current_only_yes",
    "prior_only_yes",
    "misaligned_zigzag",
    "correct_zigzag_transition",
    "other",
]
LATEX_PATTERN_HEADERS = {
    "prior_and_current_yes": r"\textit{All-True}",
    "all_no": r"\textit{All-False}",
    "current_only_yes": r"\textit{Only-Know-Latest}",
    "prior_only_yes": r"\textit{Only-Know-Prior}",
    "misaligned_zigzag": r"\textit{Wrong-Transition-Point}",
    "correct_zigzag_transition": r"\textit{Correct-Transition-Point}",
    "other": r"\textit{Other}",
}
MODEL_DISPLAY_NAMES = {
    "azure_azure-gpt-4.1": "GPT-4.1",
    "azure_azure-gpt-5": "GPT-5",
    "vllm_openai_gpt-oss-20b": "GPT-OSS-20B",
    "vllm_meta-llama_Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "vllm_Qwen_Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct",
    "vllm_Qwen_Qwen2.5-14B-Instruct": "Qwen2.5-14B-Instruct",
    "vllm_Qwen_Qwen3-0.6B": "Qwen3-0.6B",
    "vllm_Qwen_Qwen3-1.7B": "Qwen3-1.7B",
    "vllm_Qwen_Qwen3-4B": "Qwen3-4B",
    "vllm_google_medgemma-4b-it": "MedGemma-4B",
    "vllm_google_gemma-3-4b-it": "Gemma-3-4B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the percentage of prediction patterns for each model over grouped "
            "(pair_id, difference_idx) results from pubmed_inspect_questions evaluation outputs."
        )
    )
    parser.add_argument(
        "--inspect-root",
        default=RESULTS_ROOT / "inspect_questions",
        help="Root folder containing experiment_* outputs from pubmed_inspect_questions.py.",
    )
    parser.add_argument(
        "--output-csv",
        default=HEATMAP_ROOT / "prediction_pattern_summary_by_model.csv",
        help="CSV path for per-model percentages.",
    )
    parser.add_argument(
        "--output-json",
        default=HEATMAP_ROOT / "prediction_pattern_summary_by_model.json",
        help="JSON path for detailed summary.",
    )
    parser.add_argument(
        "--pie-chart-dir",
        default=HEATMAP_ROOT / "prediction_pattern_pies",
        help="Directory for per-model pie chart visualizations.",
    )
    parser.add_argument(
        "--output-latex",
        default=HEATMAP_ROOT / "prediction_pattern_summary_by_model.tex",
        help="LaTeX table path for per-model percentages.",
    )
    parser.add_argument("--left-prior-min", type=float, default=0.7)
    parser.add_argument("--left-current-max", type=float, default=0.3)
    parser.add_argument("--right-prior-max", type=float, default=0.3)
    parser.add_argument("--right-current-min", type=float, default=0.7)
    parser.add_argument(
        "--majority-threshold",
        type=float,
        default=0.9,
        help="Minimum proportion required to assign a majority-based non-zigzag pattern.",
    )
    parser.add_argument(
        "--year-start",
        type=int,
        default=None,
        help="Inclusive lower bound on question_target_year used for pattern inspection.",
    )
    parser.add_argument(
        "--year-end",
        type=int,
        default=None,
        help="Inclusive upper bound on question_target_year used for pattern inspection.",
    )
    parser.add_argument(
        "--misaligned-gap-min",
        type=int,
        default=2,
        help="Minimum absolute gap between detected split year and current guideline year to count as misaligned zig-zag.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="If set, print per-sample result paths for this model grouped by category.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def year_in_range(year: int, year_start: Optional[int], year_end: Optional[int]) -> bool:
    if year_start is not None and year < year_start:
        return False
    if year_end is not None and year > year_end:
        return False
    return True


def iter_grouped_differences(
    inspect_root: Path,
    year_start: Optional[int],
    year_end: Optional[int],
) -> Iterable[Tuple[Path, str, int, List[Dict[str, Any]]]]:
    for eval_path in inspect_root.glob("experiment_*/**/evaluation_results.json"):
        try:
            rows = read_json(eval_path)
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            pair_id = row.get("pair_id")
            difference_idx = row.get("difference_idx")
            if not pair_id or difference_idx is None:
                continue
            try:
                question_year = int(row.get("question_target_year"))
            except Exception:
                continue
            if not year_in_range(question_year, year_start, year_end):
                continue
            grouped[(pair_id, int(difference_idx))].append(row)
        for (pair_id, difference_idx), group_rows in grouped.items():
            if group_rows:
                yield eval_path, pair_id, difference_idx, group_rows


def compute_yes_ratios(rows: Sequence[Dict[str, Any]]) -> Tuple[List[int], Dict[int, Optional[float]], Dict[int, Optional[float]]]:
    years = sorted({int(row["question_target_year"]) for row in rows})
    prior_yes_ratio: Dict[int, Optional[float]] = {}
    current_yes_ratio: Dict[int, Optional[float]] = {}

    for year in years:
        prior_rows = [
            row for row in rows
            if row.get("statement_source") == "prior"
            and int(row["question_target_year"]) == year
            and row.get("prediction") in {"A", "B"}
        ]
        current_rows = [
            row for row in rows
            if row.get("statement_source") == "current"
            and int(row["question_target_year"]) == year
            and row.get("prediction") in {"A", "B"}
        ]
        prior_yes_ratio[year] = safe_mean([1.0 if row["prediction"] == "A" else 0.0 for row in prior_rows])
        current_yes_ratio[year] = safe_mean([1.0 if row["prediction"] == "A" else 0.0 for row in current_rows])
    return years, prior_yes_ratio, current_yes_ratio


def find_transition_split(
    years: Sequence[int],
    prior_yes_ratio: Dict[int, Optional[float]],
    current_yes_ratio: Dict[int, Optional[float]],
    args: argparse.Namespace,
) -> Optional[int]:
    for split_year in years[1:-1]:
        left_years = [year for year in years if year < split_year]
        right_years = [year for year in years if year >= split_year]
        if len(left_years) < 2 or len(right_years) < 2:
            continue

        left_prior = [prior_yes_ratio[year] for year in left_years if prior_yes_ratio[year] is not None]
        left_current = [current_yes_ratio[year] for year in left_years if current_yes_ratio[year] is not None]
        right_prior = [prior_yes_ratio[year] for year in right_years if prior_yes_ratio[year] is not None]
        right_current = [current_yes_ratio[year] for year in right_years if current_yes_ratio[year] is not None]
        if min(len(left_prior), len(left_current), len(right_prior), len(right_current)) < 2:
            continue

        left_prior_mean = sum(left_prior) / len(left_prior)
        left_current_mean = sum(left_current) / len(left_current)
        right_prior_mean = sum(right_prior) / len(right_prior)
        right_current_mean = sum(right_current) / len(right_current)

        if left_prior_mean < args.left_prior_min:
            continue
        if left_current_mean > args.left_current_max:
            continue
        if right_prior_mean > args.right_prior_max:
            continue
        if right_current_mean < args.right_current_min:
            continue
        return split_year
    return None


def classify_constant_pattern(rows: Sequence[Dict[str, Any]], majority_threshold: float) -> Optional[str]:
    usable = [
        row for row in rows
        if row.get("prediction") in {"A", "B"} and row.get("statement_source") in {"prior", "current"}
    ]
    if len(usable) != len(rows):
        return None

    prior_predictions = [row["prediction"] for row in usable if row.get("statement_source") == "prior"]
    current_predictions = [row["prediction"] for row in usable if row.get("statement_source") == "current"]
    if not prior_predictions or not current_predictions:
        return None

    prior_yes_ratio = sum(1 for pred in prior_predictions if pred == "A") / len(prior_predictions)
    current_yes_ratio = sum(1 for pred in current_predictions if pred == "A") / len(current_predictions)
    prior_no_ratio = 1.0 - prior_yes_ratio
    current_no_ratio = 1.0 - current_yes_ratio

    if prior_yes_ratio >= majority_threshold and current_no_ratio >= majority_threshold:
        return "prior_only_yes"
    if prior_no_ratio >= majority_threshold and current_yes_ratio >= majority_threshold:
        return "current_only_yes"
    if prior_yes_ratio >= majority_threshold and current_yes_ratio >= majority_threshold:
        return "prior_and_current_yes"
    if prior_no_ratio >= majority_threshold and current_no_ratio >= majority_threshold:
        return "all_no"
    return None


def classify_pattern(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
    years, prior_ratio, current_ratio = compute_yes_ratios(rows)
    if len(years) >= 4:
        split_year = find_transition_split(years, prior_ratio, current_ratio, args)
        if split_year is not None:
            current_years = {
                int(row["statement_guideline_year"])
                for row in rows
                if row.get("statement_source") == "current" and row.get("statement_guideline_year") is not None
            }
            if current_years:
                current_guideline_year = min(current_years)
                gap = abs(split_year - current_guideline_year)
                if gap <= args.misaligned_gap_min:
                    return "correct_zigzag_transition", {
                        "split_year": split_year,
                        "current_guideline_year": current_guideline_year,
                        "misalignment_gap": gap,
                    }
                return "misaligned_zigzag", {
                    "split_year": split_year,
                    "current_guideline_year": current_guideline_year,
                    "misalignment_gap": gap,
                }

    constant_pattern = classify_constant_pattern(rows, args.majority_threshold)
    if constant_pattern is not None:
        return constant_pattern, {}

    return "other", {}


def summarize_by_model(inspect_root: Path, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    model_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for eval_path, pair_id, difference_idx, rows in iter_grouped_differences(inspect_root, args.year_start, args.year_end):
        if not rows:
            continue
        first_row = rows[0]
        model_dir = eval_path.parent.name
        if args.model_dir and model_dir != args.model_dir:
            continue
        pattern, extra = classify_pattern(rows, args)
        model_groups[model_dir].append(
            {
                "experiment_index": first_row.get("experiment_index"),
                "pair_id": pair_id,
                "difference_idx": difference_idx,
                "pattern": pattern,
                "evaluation_results_path": str(eval_path),
                **extra,
            }
        )

    csv_rows: List[Dict[str, Any]] = []
    json_summary: Dict[str, Any] = {
        "models": {},
        "inspection_window": {"year_start": args.year_start, "year_end": args.year_end},
        "misaligned_gap_min": args.misaligned_gap_min,
    }

    for model_dir in sorted(model_groups):
        records = model_groups[model_dir]
        total = len(records)
        counter = Counter(record["pattern"] for record in records)
        model_summary = {
            "total_groups": total,
            "counts": {pattern: counter.get(pattern, 0) for pattern in PATTERN_ORDER},
            "percentages": {
                pattern: (100.0 * counter.get(pattern, 0) / total if total else 0.0)
                for pattern in PATTERN_ORDER
            },
            "records": records,
        }
        json_summary["models"][model_dir] = model_summary

        row: Dict[str, Any] = {"model_dir": model_dir, "total_groups": total}
        for pattern in PATTERN_ORDER:
            row[f"{pattern}_count"] = counter.get(pattern, 0)
            row[f"{pattern}_percentage"] = round(model_summary["percentages"][pattern], 4)
        csv_rows.append(row)

    return csv_rows, json_summary


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_dir", "total_groups"]
    for pattern in PATTERN_ORDER:
        fieldnames.append(f"{pattern}_count")
        fieldnames.append(f"{pattern}_percentage")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def format_model_name(model_dir: str) -> str:
    display = MODEL_DISPLAY_NAMES.get(model_dir, model_dir)
    return r"\texttt{%s}" % display.replace('_', r'\_')


def write_latex_table(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_dirs = sorted(summary.get("models", {}).keys())
    lines = [
        r"\begin{table}[]",
        r"    \centering",
        r"    \caption{Percentage Distribution of the Error Patterns.}",
        r"    \label{tab:percentage_distribution}",
        r"    \resizebox{\textwidth}{!}{%",
        r"    \begin{tabular}{l|ccccccc}",
        r"    \toprule",
        "        Model Name & " + " & ".join(LATEX_PATTERN_HEADERS[p] for p in LATEX_PATTERN_ORDER) + r" \\",
        r"        \midrule",
    ]
    for model_dir in model_dirs:
        model_summary = summary["models"][model_dir]
        percentages = model_summary.get("percentages", {})
        row = [format_model_name(model_dir)]
        for pattern in LATEX_PATTERN_ORDER:
            value = percentages.get(pattern, 0.0)
            row.append(f"{value:.1f}")
        lines.append("        " + " & ".join(row) + r" \\")
    lines.extend([
        r"    \bottomrule",
        r"    \end{tabular}%",
        r"    }",
        r"\end{table}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def slugify_model_dir(model_dir: str) -> str:
    return model_dir.replace("/", "_").replace(":", "_")


def make_autopct(values: Sequence[int]):
    total = sum(values)

    def _autopct(pct: float) -> str:
        if pct < 0.5 or total == 0:
            return ""
        count = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n(n={count})"

    return _autopct


def print_model_sample_paths(summary: Dict[str, Any], model_dir: str) -> None:
    models = summary.get("models", {})
    if model_dir not in models:
        print(f"Model not found in summary: {model_dir}")
        return

    print(f"Per-sample result paths for model: {model_dir}")
    records = models[model_dir].get("records", [])
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record.get("pattern", "other")].append(record)

    for pattern in PATTERN_ORDER:
        pattern_records = grouped.get(pattern, [])
        print(f"\n[{pattern}] {len(pattern_records)} samples")
        for record in pattern_records:
            extra = []
            if "split_year" in record:
                extra.append(f"split_year={record['split_year']}")
            if "current_guideline_year" in record:
                extra.append(f"current_guideline_year={record['current_guideline_year']}")
            if "misalignment_gap" in record:
                extra.append(f"misalignment_gap={record['misalignment_gap']}")
            extra_text = f" | {'; '.join(extra)}" if extra else ""
            print(
                f"- experiment_index={record.get('experiment_index')} pair_id={record.get('pair_id')} "
                f"difference_idx={record.get('difference_idx')} path=  {record.get('evaluation_results_path')}{extra_text}"
            )


def write_pie_charts(pie_chart_dir: Path, summary: Dict[str, Any]) -> List[Path]:
    pie_chart_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    saved_paths: List[Path] = []

    for model_dir, model_summary in summary.get("models", {}).items():
        counts = [model_summary["counts"].get(pattern, 0) for pattern in PATTERN_ORDER]
        if sum(counts) == 0:
            continue
        labels = [PATTERN_LABELS[pattern] for pattern in PATTERN_ORDER if model_summary["counts"].get(pattern, 0) > 0]
        values = [model_summary["counts"][pattern] for pattern in PATTERN_ORDER if model_summary["counts"].get(pattern, 0) > 0]
        colors = [PATTERN_COLORS[pattern] for pattern in PATTERN_ORDER if model_summary["counts"].get(pattern, 0) > 0]

        fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
        ax.pie(
            values,
            labels=labels,
            colors=colors,
            startangle=90,
            counterclock=False,
            autopct=make_autopct(values),
            pctdistance=0.72,
            wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
            textprops={"fontsize": 10},
        )
        ax.set_title(f"Prediction Pattern Summary | {model_dir}", fontsize=13)
        centre_circle = plt.Circle((0, 0), 0.45, fc="white")
        ax.add_artist(centre_circle)
        ax.text(0, 0, f"n={sum(values)}", ha="center", va="center", fontsize=12, weight="semibold")
        base = pie_chart_dir / f"{slugify_model_dir(model_dir)}_pattern_pie"
        png_path = base.with_suffix(".png")
        pdf_path = base.with_suffix(".pdf")
        fig.savefig(png_path, dpi=400, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        saved_paths.extend([png_path, pdf_path])
    return saved_paths


def main() -> None:
    args = parse_args()
    inspect_root = Path(args.inspect_root)
    csv_rows, json_summary = summarize_by_model(inspect_root, args)
    write_csv(Path(args.output_csv), csv_rows)
    write_json(Path(args.output_json), json_summary)
    write_latex_table(Path(args.output_latex), json_summary)
    pie_paths = write_pie_charts(Path(args.pie_chart_dir), json_summary)

    print(f"Saved CSV summary to {args.output_csv}")
    print(f"Saved JSON summary to {args.output_json}")
    print(f"Saved LaTeX table to {args.output_latex}")
    print(f"Saved {len(pie_paths)} pie-chart files to {args.pie_chart_dir}")
    for row in csv_rows:
        print(json.dumps(row, ensure_ascii=False))
    if args.model_dir:
        print_model_sample_paths(json_summary, args.model_dir)


if __name__ == "__main__":
    main()
