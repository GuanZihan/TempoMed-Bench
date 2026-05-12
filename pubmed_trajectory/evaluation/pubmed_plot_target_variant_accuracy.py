#!/usr/bin/env python3
"""Compare current-targeted and prior-targeted accuracy from 5-option evaluation results."""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

CURRENT_COLOR = "#2F4858"
PRIOR_COLOR = "#A44A3F"
GRID_COLOR = "#d9d9d9"
TEXT_COLOR = "#1f2937"
VARIANT_ORDER = ("current_guideline", "prior_guideline")
VARIANT_LABELS = {
    "current_guideline": "Current-targeted",
    "prior_guideline": "Prior-targeted",
}


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().upper()
        if text in {"TRUE", "T", "YES", "Y", "1"}:
            return True
        if text in {"FALSE", "F", "NO", "N", "0"}:
            return False
    return False


def load_results(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Input JSON must contain a list of records: {path}")

    parsed: List[Dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        if "correct" not in row:
            continue
        parsed.append(
            {
                "correct": _to_bool(row["correct"]),
                "question_variant": row.get("question_variant", "current_guideline"),
            }
        )
    return parsed


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    denominator = 1 + (z**2) / total
    center = (phat + (z**2) / (2 * total)) / denominator
    margin = z * math.sqrt((phat * (1 - phat) / total) + (z**2) / (4 * total**2)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def model_name_from_path(path: Path) -> str:
    name = path.name
    name = re.sub(r"^evaluation_results_5_option_", "", name)
    name = re.sub(r"_augmented\.json$", "", name)
    name = re.sub(r"\.json$", "", name)
    return name


def pretty_model_name(name: str) -> str:
    replacements = {
        "azure-gpt-5": "GPT-5",
        "azure-gpt-4.1": "GPT-4.1",
        "azure-gpt-4o": "GPT-4o",
        "google_medgemma-4b-it": "MedGemma-4B",
        "google_medgemma-27b-text-it": "MedGemma-27B",
        "google_gemma-3-4b-it": "Gemma-3-4B",
        "meta-llama_Llama-3.1-8B-Instruct": "Llama-3.1-8B",
        "meta-llama_Llama-3.2-3B-Instruct": "Llama-3.2-3B",
        "meta-llama_Llama-2-13b-chat-hf": "Llama-2-13B",
        "meta-llama_Llama-2-7b-chat-hf": "Llama-2-7B",
        "openai_gpt-oss-20b": "GPT-OSS-20B",
    }
    if name in replacements:
        return replacements[name]
    return name.replace("Qwen_", "").replace("_", "/")


def summarize_file(path: Path) -> Dict[str, Any]:
    records = load_results(path)
    grouped = defaultdict(lambda: {"correct": 0, "n": 0})
    for row in records:
        variant = row.get("question_variant", "current_guideline")
        if variant not in VARIANT_ORDER:
            continue
        grouped[variant]["correct"] += int(row["correct"])
        grouped[variant]["n"] += 1

    model = model_name_from_path(path)
    summary = {"model": model, "model_label": pretty_model_name(model), "path": str(path)}
    for variant in VARIANT_ORDER:
        correct = grouped[variant]["correct"]
        n = grouped[variant]["n"]
        ci_low, ci_high = wilson_interval(correct, n)
        summary[variant] = {
            "correct": correct,
            "n": n,
            "accuracy": correct / n if n else 0.0,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    return summary


def configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "Andale Mono",
            "figure.dpi": 140,
            "axes.labelsize": 80,
            "xtick.labelsize": 80,
            "ytick.labelsize": 80,
            "legend.fontsize": 80,
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
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))


def create_figure(rows: Sequence[Dict[str, Any]], output_path: Path) -> List[Path]:
    configure_style()
    rows = sorted(rows, key=lambda row: row["current_guideline"]["accuracy"], reverse=True)
    labels = [row["model_label"] for row in rows]
    x = np.arange(len(rows)) * 1.75
    width = 0.16
    offset = 0.13

    current_y = [row["current_guideline"]["accuracy"] for row in rows]
    prior_y = [row["prior_guideline"]["accuracy"] for row in rows]
    current_err = [
        [max(0.0, row["current_guideline"]["accuracy"] - row["current_guideline"]["ci_low"]) for row in rows],
        [max(0.0, row["current_guideline"]["ci_high"] - row["current_guideline"]["accuracy"]) for row in rows],
    ]
    prior_err = [
        [max(0.0, row["prior_guideline"]["accuracy"] - row["prior_guideline"]["ci_low"]) for row in rows],
        [max(0.0, row["prior_guideline"]["ci_high"] - row["prior_guideline"]["accuracy"]) for row in rows],
    ]

    fig_width = max(22.0, min(52.0, 2.8 * len(rows)))
    fig, ax = plt.subplots(figsize=(fig_width, 11.0), constrained_layout=True)
    fig.patch.set_facecolor("white")
    style_ax(ax)

    ax.bar(
        x - offset,
        current_y,
        width,
        yerr=current_err,
        label=VARIANT_LABELS["current_guideline"],
        color=CURRENT_COLOR,
        edgecolor=CURRENT_COLOR,
        linewidth=0.8,
        error_kw={"elinewidth": 1.0, "capsize": 3, "capthick": 1.0},
        alpha=0.92,
    )
    ax.bar(
        x + offset,
        prior_y,
        width,
        yerr=prior_err,
        label=VARIANT_LABELS["prior_guideline"],
        color=PRIOR_COLOR,
        edgecolor=PRIOR_COLOR,
        linewidth=0.8,
        error_kw={"elinewidth": 1.0, "capsize": 3, "capthick": 1.0},
        alpha=0.92,
    )

    ax.set_ylabel("Average Accuracy", color=TEXT_COLOR)
    # ax.set_xlabel("Model", color=TEXT_COLOR)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.legend(frameon=False, loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def write_summary_csv(rows: Sequence[Dict[str, Any]], output_path: Path) -> Path:
    csv_path = output_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "model_label",
                "current_correct",
                "current_total",
                "current_accuracy",
                "current_ci_low",
                "current_ci_high",
                "prior_correct",
                "prior_total",
                "prior_accuracy",
                "prior_ci_low",
                "prior_ci_high",
                "source_path",
            ]
        )
        for row in rows:
            current = row["current_guideline"]
            prior = row["prior_guideline"]
            writer.writerow(
                [
                    row["model"],
                    row["model_label"],
                    current["correct"],
                    current["n"],
                    f"{current['accuracy']:.6f}",
                    f"{current['ci_low']:.6f}",
                    f"{current['ci_high']:.6f}",
                    prior["correct"],
                    prior["n"],
                    f"{prior['accuracy']:.6f}",
                    f"{prior['ci_low']:.6f}",
                    f"{prior['ci_high']:.6f}",
                    row["path"],
                ]
            )
    return csv_path


def resolve_inputs(args: argparse.Namespace) -> List[Path]:
    if args.input:
        return [Path(item) for item in args.input]
    return sorted(Path(args.input_dir).glob(args.glob))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare current-targeted and prior-targeted accuracy.")
    parser.add_argument(
        "--input",
        nargs="*",
        default=None,
        help="One or more evaluation result JSON files. If omitted, --input-dir/--glob are used.",
    )
    parser.add_argument(
        "--input-dir",
        default="pubmed_trajectory/results",
        help="Directory used when --input is omitted.",
    )
    parser.add_argument(
        "--glob",
        default="evaluation_results_5_option_*_augmented.json",
        help="Glob pattern used inside --input-dir when --input is omitted.",
    )
    parser.add_argument(
        "--output",
        default="pubmed_trajectory/evaluation/results/pubmed_target_variant_accuracy_comparison",
        help="Output path stem for PNG/PDF/CSV files.",
    )
    args = parser.parse_args()

    paths = resolve_inputs(args)
    if not paths:
        raise FileNotFoundError("No evaluation result JSON files found.")

    rows = [summarize_file(path) for path in paths]
    saved = create_figure(rows, Path(args.output))
    csv_path = write_summary_csv(rows, Path(args.output))
    for path in saved:
        print(f"Saved figure: {path}")
    print(f"Saved summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
