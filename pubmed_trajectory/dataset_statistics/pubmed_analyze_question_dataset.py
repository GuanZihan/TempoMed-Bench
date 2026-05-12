#!/usr/bin/env python3
"""Analyze the verified augmented guideline question dataset."""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "questions_2026_relaxed_4_option_augmented_verified.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "pubmed_trajectory" / "dataset_statistics" / "results" / "question_dataset"

PLOT_COLOR = "#2F4858"
SECONDARY_COLOR = "#A44A3F"
GRID_COLOR = "#d9d9d9"
TEXT_COLOR = "#1f2937"
VALID_YEAR_MIN = 1900
VALID_YEAR_MAX = 2030
REQUIRED_TOP_LEVEL_FIELDS = ("idx", "PMID_current", "Year_current", "PMID_prior", "Year_prior", "Question", "Answer")
REQUIRED_ANSWER_FIELDS = ("Choice_A", "Choice_B", "Choice_C", "Choice_D", "Correct", "Explanation")


METRICS_DESCRIPTION = {
    "size_and_coverage": "Total questions, unique current PMIDs, unique prior PMIDs, and unique current-prior pairs.",
    "temporal_structure": "Current guideline year distribution, prior guideline year distribution, and year gap distribution.",
    "pair_density": "Number of generated questions per current-prior guideline pair and per current guideline.",
    "answer_schema": "Correct-label distribution, number of answer choices per question, and presence of optional Choice_E.",
    "text_length": "Word-count distributions for question stems plus option sets, and explanations.",
    "quality_checks": "Missing fields, malformed years, duplicate idx values, duplicate question texts, and zero/missing PMIDs.",
}


def parse_json_blocks(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON list in {path}")
        return [row for row in payload if isinstance(row, dict)]

    rows = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        rows.append(json.loads(block))
    return rows


def normalize_pmid(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "0":
        return None
    return text


def normalize_year(value: Any) -> Optional[int]:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    if VALID_YEAR_MIN <= year <= VALID_YEAR_MAX:
        return year
    return None


def word_count(text: Any) -> int:
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"\b\w+\b", text))


def safe_stats(values: Iterable[int]) -> Dict[str, Optional[float]]:
    vals = list(values)
    if not vals:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": min(vals),
        "max": max(vals),
        "mean": float(mean(vals)),
        "median": float(median(vals)),
    }


def count_choices(answer: Dict[str, Any]) -> int:
    return sum(1 for label in ("A", "B", "C", "D", "E") if answer.get(f"Choice_{label}"))


def question_with_options_text(row: Dict[str, Any], answer: Dict[str, Any]) -> str:
    parts = [str(row.get("Question", ""))]
    for label in ("A", "B", "C", "D", "E"):
        choice = answer.get(f"Choice_{label}")
        if choice:
            parts.append(str(choice))
    return "\n".join(parts)


def analyze_rows(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Counter], List[Dict[str, Any]]]:
    current_pmids = set()
    prior_pmids = set()
    pairs = set()
    current_years = []
    prior_years = []
    year_gaps = []
    question_word_counts = []
    explanation_word_counts = []
    pair_counts = Counter()
    current_counts = Counter()
    correct_counts = Counter()
    choice_count_distribution = Counter()
    idx_counts = Counter()
    question_counts = Counter()
    missing_field_counts = Counter()
    malformed_year_rows = []
    zero_or_missing_pmid_counts = Counter()

    for row_i, row in enumerate(rows):
        missing_top = [field for field in REQUIRED_TOP_LEVEL_FIELDS if field not in row]
        for field in missing_top:
            missing_field_counts[f"top_level.{field}"] += 1

        answer = row.get("Answer") if isinstance(row.get("Answer"), dict) else {}
        for field in REQUIRED_ANSWER_FIELDS:
            if field not in answer:
                missing_field_counts[f"Answer.{field}"] += 1

        idx_counts[row.get("idx")] += 1
        question_text = row.get("Question", "")
        question_counts[question_text] += 1
        question_word_counts.append(word_count(question_with_options_text(row, answer)))
        explanation_word_counts.append(word_count(answer.get("Explanation", "")))
        correct_counts[str(answer.get("Correct", "MISSING")).strip().upper() or "MISSING"] += 1
        choice_count_distribution[count_choices(answer)] += 1

        current_pmid = normalize_pmid(row.get("PMID_current"))
        prior_pmid = normalize_pmid(row.get("PMID_prior"))
        if current_pmid:
            current_pmids.add(current_pmid)
        else:
            zero_or_missing_pmid_counts["current"] += 1
        if prior_pmid:
            prior_pmids.add(prior_pmid)
        else:
            zero_or_missing_pmid_counts["prior"] += 1

        current_year = normalize_year(row.get("Year_current"))
        prior_year = normalize_year(row.get("Year_prior"))
        if current_year is None or prior_year is None:
            malformed_year_rows.append(
                {
                    "row_index": row_i,
                    "idx": row.get("idx"),
                    "Year_current": row.get("Year_current"),
                    "Year_prior": row.get("Year_prior"),
                }
            )
        else:
            current_years.append(current_year)
            prior_years.append(prior_year)
            year_gaps.append(current_year - prior_year)

        if current_pmid and prior_pmid:
            pair_key = f"{current_pmid}::{prior_pmid}"
            pairs.add(pair_key)
            pair_counts[pair_key] += 1
            current_counts[current_pmid] += 1

    duplicate_idx = {str(k): v for k, v in idx_counts.items() if k is not None and v > 1}
    duplicate_questions = sum(1 for _, count in question_counts.items() if count > 1)

    summary = {
        "metrics_analyzed": METRICS_DESCRIPTION,
        "total_questions": len(rows),
        "unique_current_pmids": len(current_pmids),
        "unique_prior_pmids": len(prior_pmids),
        "unique_current_prior_pairs": len(pairs),
        "questions_per_pair": safe_stats(pair_counts.values()),
        "questions_per_current_guideline": safe_stats(current_counts.values()),
        "current_year": safe_stats(current_years),
        "prior_year": safe_stats(prior_years),
        "year_gap": safe_stats(year_gaps),
        "question_word_count": safe_stats(question_word_counts),
        "explanation_word_count": safe_stats(explanation_word_counts),
        "correct_option_distribution": dict(sorted(correct_counts.items())),
        "choice_count_distribution": dict(sorted(choice_count_distribution.items())),
        "missing_field_counts": dict(sorted(missing_field_counts.items())),
        "malformed_year_rows": len(malformed_year_rows),
        "zero_or_missing_pmid_counts": dict(zero_or_missing_pmid_counts),
        "duplicate_idx_values": duplicate_idx,
        "duplicate_question_text_count": duplicate_questions,
    }

    counters = {
        "current_year": Counter(current_years),
        "prior_year": Counter(prior_years),
        "year_gap": Counter(year_gaps),
        "pair_counts": pair_counts,
        "current_counts": current_counts,
        "correct_option": correct_counts,
        "choice_count": choice_count_distribution,
        "question_word_count": Counter(question_word_counts),
    }
    return summary, counters, malformed_year_rows


def configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "Andale Mono",
            "figure.dpi": 140,
            "axes.labelsize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
        }
    )


def style_ax(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_counter_bar(ax, counter: Counter, xlabel: str, ylabel: str, color: str = PLOT_COLOR) -> None:
    style_ax(ax)
    if not counter:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color=TEXT_COLOR)
        ax.set_xlabel(xlabel, color=TEXT_COLOR)
        ax.set_ylabel(ylabel, color=TEXT_COLOR)
        return
    keys = sorted(counter)
    values = [counter[key] for key in keys]
    ax.bar(keys, values, color=color, edgecolor=color, alpha=0.9, width=0.85)
    ax.set_xlabel(xlabel, color=TEXT_COLOR)
    ax.set_ylabel(ylabel, color=TEXT_COLOR)


def save_single_counter_figure(
    counter: Counter,
    output_dir: Path,
    stem: str,
    xlabel: str,
    ylabel: str,
    color: str = PLOT_COLOR,
    width: float = 0.85,
) -> List[Path]:
    configure_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    style_ax(ax)

    if counter:
        keys = sorted(counter)
        values = [counter[key] for key in keys]
        ax.bar(keys, values, color=color, edgecolor=color, alpha=0.9, width=width)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color=TEXT_COLOR)

    ax.set_xlabel(xlabel, color=TEXT_COLOR)
    ax.set_ylabel(ylabel, color=TEXT_COLOR)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def create_summary_figures(counters: Dict[str, Counter], output_dir: Path) -> List[Path]:
    saved_paths: List[Path] = []
    saved_paths.extend(
        save_single_counter_figure(
            counters["current_year"],
            output_dir,
            "current_year_distribution",
            "Up-to-date Guideline Year",
            "Questions",
            PLOT_COLOR,
        )
    )
    saved_paths.extend(
        save_single_counter_figure(
            counters["prior_year"],
            output_dir,
            "prior_year_distribution",
            "Outdated Guideline Year",
            "Questions",
            SECONDARY_COLOR,
        )
    )
    saved_paths.extend(
        save_single_counter_figure(
            counters["year_gap"],
            output_dir,
            "year_gap_distribution",
            "Year Gap",
            "Questions",
            PLOT_COLOR,
        )
    )
    pair_count_distribution = Counter(counters["pair_counts"].values())
    saved_paths.extend(
        save_single_counter_figure(
            pair_count_distribution,
            output_dir,
            "questions_per_current_prior_pair_distribution",
            "Questions per Current-Prior Pair",
            "Pairs",
            SECONDARY_COLOR,
        )
    )
    saved_paths.extend(
        save_single_counter_figure(
            counters["question_word_count"],
            output_dir,
            "question_word_count_distribution",
            "MCQ Word Count",
            "Questions",
            PLOT_COLOR,
            width=1.0,
        )
    )
    return saved_paths


def write_counter_csv(counter: Counter, path: Path, key_name: str, count_name: str = "count") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([key_name, count_name])
        for key, count in sorted(counter.items(), key=lambda item: item[0]):
            writer.writerow([key, count])


def write_pair_counts(counter: Counter, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["PMID_current", "PMID_prior", "question_count"])
        for pair, count in counter.most_common():
            current, prior = pair.split("::", 1)
            writer.writerow([current, prior, count])


def write_outputs(summary: Dict[str, Any], counters: Dict[str, Counter], malformed_rows: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "question_dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "malformed_year_rows.json").write_text(
        json.dumps(malformed_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_counter_csv(counters["current_year"], output_dir / "current_year_distribution.csv", "year")
    write_counter_csv(counters["prior_year"], output_dir / "prior_year_distribution.csv", "year")
    write_counter_csv(counters["year_gap"], output_dir / "year_gap_distribution.csv", "year_gap")
    write_counter_csv(counters["correct_option"], output_dir / "correct_option_distribution.csv", "option")
    write_counter_csv(counters["choice_count"], output_dir / "choice_count_distribution.csv", "choice_count")
    write_counter_csv(counters["question_word_count"], output_dir / "question_word_count_distribution.csv", "word_count")
    write_pair_counts(counters["pair_counts"], output_dir / "current_prior_pair_question_counts.csv")
    create_summary_figures(counters, output_dir)


def print_report(summary: Dict[str, Any], output_dir: Path) -> None:
    print("\nMetrics analyzed:")
    for name, description in summary["metrics_analyzed"].items():
        print(f"  - {name}: {description}")

    print("\nQuestion dataset summary:")
    print(f"  Total questions             : {summary['total_questions']}")
    print(f"  Unique current PMIDs        : {summary['unique_current_pmids']}")
    print(f"  Unique prior PMIDs          : {summary['unique_prior_pmids']}")
    print(f"  Unique current-prior pairs  : {summary['unique_current_prior_pairs']}")
    print(f"  Mean questions / pair       : {summary['questions_per_pair']['mean']:.3f}")
    print(f"  Median questions / pair     : {summary['questions_per_pair']['median']:.3f}")
    print(f"  Current year range          : {summary['current_year']['min']} - {summary['current_year']['max']}")
    print(f"  Prior year range            : {summary['prior_year']['min']} - {summary['prior_year']['max']}")
    print(f"  Mean year gap               : {summary['year_gap']['mean']:.3f}")
    print(f"  Malformed year rows         : {summary['malformed_year_rows']}")
    print(f"  Output directory            : {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze verified augmented guideline question dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows = parse_json_blocks(args.input)
    if not rows:
        raise RuntimeError(f"No question records loaded from {args.input}")

    summary, counters, malformed_rows = analyze_rows(rows)
    write_outputs(summary, counters, malformed_rows, args.output_dir)
    print_report(summary, args.output_dir)


if __name__ == "__main__":
    main()
