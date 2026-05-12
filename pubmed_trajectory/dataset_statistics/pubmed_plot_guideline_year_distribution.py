from pathlib import Path

from _dataset_stats_common import (
    SUBSETS,
    build_argument_parser,
    configure_plot_style,
    load_all_rows,
    plot_histogram_by_subset,
)


def main() -> None:
    parser = build_argument_parser("Plot the guideline year distribution across dataset subsets.")
    args = parser.parse_args()

    configure_plot_style()
    rows = load_all_rows(args.comm_dir, args.noncomm_dir, args.other_dir)
    data = {subset: [year for row in rows if row.subset == subset for year in row.guideline_years] for subset in SUBSETS}
    output_path = Path(args.output_dir) / "guideline_year_distribution.png"
    saved_paths = plot_histogram_by_subset(data, "Guideline Year Distribution", "Guideline Year (Current + Prior)", output_path)
    for saved_path in saved_paths:
        print(f"Saved plot to: {saved_path}")


if __name__ == "__main__":
    main()
