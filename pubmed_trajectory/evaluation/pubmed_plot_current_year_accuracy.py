#!/usr/bin/env python3
"""Plot mean accuracy by current guideline year from 5-option evaluation results."""

import argparse
import json
import math
import numpy as np
from collections import defaultdict
from statistics import mean
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

CURRENT_COLOR = "#2F4858"
TREND_COLOR = "#D97706"
GRID_COLOR = "#d9d9d9"
TEXT_COLOR = "#1f2937"


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


def load_results(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list of evaluation records.")
    if not data:
        raise ValueError("Input JSON has no records.")

    parsed: List[Dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        if not {"year_current", "year_previous", "correct"}.issubset(row.keys()):
            continue
        try:
            year_current = int(row["year_current"])
            year_previous = int(row["year_previous"])
        except (TypeError, ValueError):
            continue
        parsed.append(
            {
                "year_current": year_current,
                "year_previous": year_previous,
                "year_gap": year_current - year_previous,
                "correct": _to_bool(row["correct"]),
                "question_variant": row.get("question_variant", "current_guideline"),
            }
        )

    if not parsed:
        raise ValueError("No valid rows after parsing year_current/year_previous/correct.")
    return parsed


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    denominator = 1 + (z ** 2) / total
    center = (phat + (z ** 2) / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt((phat * (1 - phat) / total) + (z ** 2) / (4 * total ** 2))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def summarize_accuracy(records: List[Dict[str, Any]], key_name: str, start_year: int | None = None) -> List[Dict[str, Any]]:
    grouped = defaultdict(lambda: {"correct_sum": 0, "n": 0, "years": []})
    before_key = None
    if start_year is not None:
        before_key = f"Pre-{start_year}"

    for record in records:
        year = record[key_name]
        key = year
        if start_year is not None and year <= start_year:
            key = before_key
        grouped[key]["correct_sum"] += int(record["correct"])
        grouped[key]["n"] += 1
        grouped[key]["years"].append(year)

    rows = []
    sorted_keys = sorted(k for k in grouped if k != before_key)
    if before_key is not None and before_key in grouped:
        sorted_keys = [before_key] + sorted_keys

    for display_key in sorted_keys:
        payload = grouped[display_key]
        correct_sum = payload["correct_sum"]
        n = payload["n"]
        ci_low, ci_high = wilson_interval(correct_sum, n)
        numeric_year = float(mean(payload["years"])) if payload["years"] else None
        rows.append(
            {
                "key": display_key,
                "tick_label": str(display_key),
                "regression_x": numeric_year,
                "accuracy": correct_sum / n,
                "n": n,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return rows


def filter_records_by_question_variant(records: List[Dict[str, Any]], question_variant: str) -> List[Dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("question_variant", "current_guideline") == question_variant
    ]


def configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "Andale Mono",
            "figure.dpi": 300,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 18,
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


def create_figure(summary_rows: List[Dict[str, Any]], output_path: Path) -> List[Path]:
    configure_style()
    x_positions = list(range(len(summary_rows)))
    tick_labels = [row["tick_label"] for row in summary_rows]
    regression_x = [row["regression_x"] for row in summary_rows]
    y = [row["accuracy"] for row in summary_rows]
    yerr_lower = [max(0.0, min(1.0, row["accuracy"] - row["ci_low"])) for row in summary_rows]
    yerr_upper = [max(0.0, min(1.0, row["ci_high"] - row["accuracy"])) for row in summary_rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    style_ax(ax)
    ax.errorbar(
        x_positions,
        y,
        yerr=[yerr_lower, yerr_upper],
        fmt="o-",
        color=CURRENT_COLOR,
        ecolor=CURRENT_COLOR,
        elinewidth=1.2,
        capsize=3,
        capthick=1.2,
        linewidth=2.0,
        markersize=6.0,
        markerfacecolor="white",
        markeredgewidth=1.4,
        zorder=3,
    )
    ax.fill_between(x_positions, y, color=CURRENT_COLOR, alpha=0.08, zorder=1)

    if len(x_positions) >= 2:
        coeffs = np.polyfit(np.asarray(regression_x, dtype=float), np.asarray(y, dtype=float), deg=1)
        trend = np.poly1d(coeffs)
        y_start = float(np.clip(trend(min(regression_x)), 0.0, 1.0))
        y_end = float(np.clip(trend(max(regression_x)), 0.0, 1.0))
        ax.plot(
            [x_positions[0], x_positions[-1]],
            [y_start, y_end],
            linestyle=":",
            linewidth=5,
            color=TREND_COLOR,
            alpha=0.95,
            zorder=2,
            label="Linear trend",
        )
        ax.legend(frameon=False, loc="best")

    ax.set_xlabel("Current Guideline Year", color=TEXT_COLOR)
    ax.set_ylabel("Mean Accuracy", color=TEXT_COLOR)
    ax.set_ylim(0, 1.02)

    tick_positions = []
    tick_display_labels = []
    derived_start_year = None

    for row in summary_rows:
        label = row["tick_label"]
        if isinstance(row["key"], str) and label.startswith("Pre-"):
            tick_positions.append(x_positions[summary_rows.index(row)])
            tick_display_labels.append(label)
            try:
                derived_start_year = int(label.split("Pre-", 1)[1])
            except ValueError:
                derived_start_year = None
            break

    for pos, row in zip(x_positions, summary_rows):
        key = row["key"]
        label = row["tick_label"]
        if isinstance(key, int):
            if derived_start_year is None:
                if not tick_positions or (key - summary_rows[0]["key"]) % 4 == 0:
                    tick_positions.append(pos)
                    tick_display_labels.append(label)
            elif key > derived_start_year and (key - derived_start_year) % 4 == 0:
                tick_positions.append(pos)
                tick_display_labels.append(label)

    if not tick_positions and summary_rows:
        tick_positions = x_positions
        tick_display_labels = tick_labels

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_display_labels)

    for row, pos in zip(summary_rows, x_positions):
        ax.annotate(
            f'n={row["n"]}',
            (pos, row["accuracy"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8.5,
            color=TEXT_COLOR,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mean accuracy by current guideline year.")
    parser.add_argument(
        "--input",
        default="pubmed_trajectory/results/evaluation_results_5_option_Qwen_Qwen2.5-14B-Instruct_augmented.json",
        help="Path to evaluation results JSON.",
    )
    parser.add_argument(
        "--output",
        default="pubmed_trajectory/evaluation/results/pubmed_current_year_accuracy",
        help="Output path stem for the figure (without extension required).",
    )
    parser.add_argument(
        "--start_year",
        type=int,
        default=None,
        help="Optional lower bound on current guideline year to include in the plot.",
    )
    args = parser.parse_args()

    records = load_results(args.input)
    current_target_records = filter_records_by_question_variant(records, "current_guideline")
    summary_rows = summarize_accuracy(current_target_records or records, "year_current", start_year=args.start_year)
    if not summary_rows:
        raise ValueError("No rows remain after applying the start_year filter.")
    saved_paths = create_figure(summary_rows, Path(args.output))
    for path in saved_paths:
        print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
