#!/usr/bin/env python3
"""Export a LaTeX table for current-targeted 5-option evaluation summaries."""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List


DEFAULT_INPUT_DIR = Path("pubmed_trajectory/results")
DEFAULT_OUTPUT = Path("pubmed_trajectory/evaluation/results/current_targeted_results_table.tex")
DEFAULT_MODEL_SCRIPT = Path("evaluate_5_option.sh")


def parse_models_from_shell_script(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    in_models = False
    models: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not in_models:
            if line.startswith("models=("):
                in_models = True
            continue
        if line == ")":
            break
        models.extend(re.findall(r'"([^"]+)"', raw_line))
    return models


def slug_from_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def summary_path_for_model(input_dir: Path, model_name: str) -> Path:
    slug = slug_from_model_name(model_name)
    return input_dir / f"visualize_5_option_{slug}_augmented_question_summary.csv"


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
        "meta-llama_Llama-2-70b-chat-hf": "Llama-2-70B",
        "meta-llama_Llama-3.1-70B-Instruct": "Llama-3.1-70B",
        "openai_gpt-oss-20b": "GPT-OSS-20B",
        "Qwen_Qwen3-4B-Instruct-2507": "Qwen3-4B-2507",
        "allenai_Olmo-3-7B-Instruct": "OLMo-3-7B",
        "Intelligent-Internet_II-Medical-8B": "II-Medical-8B",
    }
    if name in replacements:
        return replacements[name]
    return name.replace("Qwen_", "").replace("_", "/")


def load_current_targeted_summary(path: Path) -> Dict[str, float]:
    total_variants = 0.0
    weighted_accuracy = 0.0
    counts = {
        "current": 0.0,
        "prior": 0.0,
        "distractor": 0.0,
        "invalid": 0.0,
        "unknown": 0.0,
    }

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("question_variant") != "current_guideline":
                continue
            n = float(row["n_variants"])
            total_variants += n
            weighted_accuracy += float(row["mean_accuracy"]) * n
            counts["current"] += float(row["current_guideline_count"])
            counts["prior"] += float(row["prior_guideline_count"])
            counts["distractor"] += float(row["interference_count"])
            counts["invalid"] += float(row["invalid_count"])
            counts["unknown"] += float(row["unknown_count"])

    if total_variants <= 0:
        raise ValueError(f"No current_guideline rows found in {path}")

    total_predictions = sum(counts.values())
    valid_predictions = counts["current"] + counts["prior"] + counts["distractor"]

    return {
        "accuracy": weighted_accuracy / total_variants,
        "accuracy_valid": (counts["current"] / valid_predictions) if valid_predictions > 0 else 0.0,
        "current": counts["current"] / total_predictions if total_predictions > 0 else 0.0,
        "prior": counts["prior"] / total_predictions if total_predictions > 0 else 0.0,
        "distractor": counts["distractor"] / total_predictions if total_predictions > 0 else 0.0,
        "invalid": counts["invalid"] / total_predictions if total_predictions > 0 else 0.0,
        "unknown": counts["unknown"] / total_predictions if total_predictions > 0 else 0.0,
    }


def pct(value: float) -> str:
    return f"{value * 100:.1f}"


def build_table(rows: List[Dict[str, str]], caption: str, label: str) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Model & Acc. (\%) & Acc.$_{\text{valid}}$ (\%) & Current (\%) & Prior (\%) & Distractor (\%) & Invalid (\%) & Unknown (\%) " + r"\\",
        r"\midrule",
    ]

    for row in rows:
        lines.append(
            f"{row['model']} & {row['accuracy']} & {row['accuracy_valid']} & {row['current']} & {row['prior']} & {row['distractor']} & {row['invalid']} & {row['unknown']} " + r"\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a LaTeX table from current-targeted 5-option summaries.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--model-script", type=Path, default=DEFAULT_MODEL_SCRIPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--caption",
        default=(
            "Current-targeted evaluation results for different models on the 5-option benchmark. "
            "Acc. is the overall current-targeted accuracy. "
            "Acc.$_{\\text{valid}}$ excludes invalid and unknown predictions from the denominator. "
            "Prediction distributions are normalized over all current-targeted predictions."
        ),
    )
    parser.add_argument("--label", default="tab:current_targeted_results")
    args = parser.parse_args()

    models = parse_models_from_shell_script(args.model_script)
    if not models:
        raise FileNotFoundError(f"No model entries found in {args.model_script}")

    rows: List[Dict[str, str]] = []
    missing_models: List[str] = []

    for model_name in models:
        slug = slug_from_model_name(model_name)
        pretty_name = pretty_model_name(slug)
        summary_path = summary_path_for_model(args.input_dir, model_name)
        if summary_path.exists():
            summary = load_current_targeted_summary(summary_path)
            rows.append(
                {
                    "model": rf"\texttt{{{pretty_name}}}",
                    "accuracy": pct(summary["accuracy"]),
                    "accuracy_valid": pct(summary["accuracy_valid"]),
                    "current": pct(summary["current"]),
                    "prior": pct(summary["prior"]),
                    "distractor": pct(summary["distractor"]),
                    "invalid": pct(summary["invalid"]),
                    "unknown": pct(summary["unknown"]),
                }
            )
        else:
            missing_models.append(model_name)
            rows.append(
                {
                    "model": rf"\texttt{{{pretty_name}}}",
                    "accuracy": "N/A",
                    "accuracy_valid": "N/A",
                    "current": "N/A",
                    "prior": "N/A",
                    "distractor": "N/A",
                    "invalid": "N/A",
                    "unknown": "N/A",
                }
            )

    latex = build_table(rows, args.caption, args.label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(latex, encoding="utf-8")
    print(f"Saved LaTeX table: {args.output}")
    if missing_models:
        print("Missing summary files for the following models:")
        for model_name in missing_models:
            print(f"  - {model_name}")


if __name__ == "__main__":
    main()
