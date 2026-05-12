import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMM_DIR = PROJECT_ROOT / "comm_guideline_trajectory_2026_relaxed_augmented_year_calibrated"
DEFAULT_NONCOMM_DIR = PROJECT_ROOT / "noncomm_guideline_trajectory_2026_relaxed_augmented_year_calibrated"
DEFAULT_OTHER_DIR = PROJECT_ROOT / "other_guideline_trajectory_2026_relaxed_augmented_year_calibrated"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "pubmed_trajectory" / "dataset_statistics" / "results"
MIN_VALID_YEAR = 1900
MAX_VALID_YEAR = 2030

SUBSETS = ["comm", "noncomm", "other"]
SUBSET_COLORS = {
    "comm": "#1f4e79",
    "noncomm": "#b24a0b",
    "other": "#2d7f5e",
}
SUBSET_LABELS = {
    "comm": "COMM",
    "noncomm": "NONCOMM",
    "other": "OTHER",
}


@dataclass
class TrajectoryStats:
    subset: str
    file_path: str
    current_year: Optional[int]
    guideline_years: List[int]
    trajectory_length: int
    timespan: int


def normalize_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        parsed = value
    else:
        try:
            parsed = int(str(value).strip())
        except Exception:
            return None
    if parsed == 0:
        return None
    if parsed < MIN_VALID_YEAR or parsed > MAX_VALID_YEAR:
        return None
    return parsed


def iter_json_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*/*.json"))


def collect_subset_stats(subset: str, root: Path) -> List[TrajectoryStats]:
    paths = list(iter_json_files(root))
    rows: List[TrajectoryStats] = []
    for path in tqdm(paths, desc=f"Load {subset} trajectories"):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        current_year = normalize_int(payload.get("year_of_current_guidance"))
        years: List[int] = []
        if current_year is not None:
            years.append(current_year)
        priors = payload.get("prior_guidelines") or []
        for prior in priors:
            if not isinstance(prior, dict):
                continue
            prior_year = normalize_int(prior.get("year"))
            if prior_year is not None:
                years.append(prior_year)
        timespan = (max(years) - min(years)) if years else 0
        rows.append(
            TrajectoryStats(
                subset=subset,
                file_path=str(path),
                current_year=current_year,
                guideline_years=years,
                trajectory_length=1 + sum(1 for prior in priors if isinstance(prior, dict)),
                timespan=timespan,
            )
        )
    return rows


def load_all_rows(comm_dir: str, noncomm_dir: str, other_dir: str) -> List[TrajectoryStats]:
    rows: List[TrajectoryStats] = []
    rows.extend(collect_subset_stats("comm", Path(comm_dir)))
    rows.extend(collect_subset_stats("noncomm", Path(noncomm_dir)))
    rows.extend(collect_subset_stats("other", Path(other_dir)))
    return rows


def bins_from_values(values: List[int]) -> np.ndarray:
    if not values:
        return np.array([0, 1])
    lo = min(values)
    hi = max(values)
    if lo == hi:
        return np.array([lo - 0.5, hi + 0.5])
    return np.arange(lo - 0.5, hi + 1.5, 1)


def style_ax(ax, xlabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")


def plot_histogram_by_subset(data: Dict[str, List[int]], title: str, xlabel: str, output_path: Path) -> List[Path]:
    all_values = [value for values in data.values() for value in values]
    bins = bins_from_values(all_values)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    for subset in SUBSETS:
        values = data.get(subset, [])
        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.hist(values, bins=bins, color=SUBSET_COLORS[subset], alpha=0.92, edgecolor="white", linewidth=0.6)
        style_ax(ax, xlabel)
        subset_png = output_path.with_name(f"{output_path.stem}_{subset}.png")
        subset_pdf = output_path.with_name(f"{output_path.stem}_{subset}.pdf")
        fig.savefig(subset_png, dpi=400, bbox_inches="tight")
        fig.savefig(subset_pdf, bbox_inches="tight")
        plt.close(fig)
        saved_paths.extend([subset_png, subset_pdf])
    return saved_paths


def configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "Andale Mono",
            "figure.dpi": 140,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )


def build_argument_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--comm-dir", default=DEFAULT_COMM_DIR)
    parser.add_argument("--noncomm-dir", default=DEFAULT_NONCOMM_DIR)
    parser.add_argument("--other-dir", default=DEFAULT_OTHER_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser
