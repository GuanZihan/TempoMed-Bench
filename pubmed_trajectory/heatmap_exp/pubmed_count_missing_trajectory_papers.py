import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTRACTED_DIRS = [
    PROJECT_ROOT / "pmc_oa_comm_extracted_2026_relaxed",
    PROJECT_ROOT / "pmc_oa_noncomm_extracted_2026_relaxed",
    PROJECT_ROOT / "pmc_oa_other_extracted_2026_relaxed",
]

DEFAULT_TRAJECTORY_SUMMARY = PROJECT_ROOT / "pubmed_trajectory" / "heatmap_exp" / "results" / "trajectory_browser_2026_relaxed_clean_summary.json"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "pubmed_trajectory" / "heatmap_exp" / "results" / "trajectory_missing_papers_clean_summary.json"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "pubmed_trajectory" / "heatmap_exp" / "results" / "trajectory_missing_papers_clean_summary.csv"


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def strip_namespace(tag):
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def extract_pmid_from_xml(xml_path):
    try:
        for _, elem in ET.iterparse(str(xml_path), events=("end",)):
            if strip_namespace(elem.tag) != "article-id":
                continue
            pub_id_type = elem.attrib.get("pub-id-type", "").lower()
            if pub_id_type != "pmid":
                elem.clear()
                continue
            text = (elem.text or "").strip()
            pmid = to_int(text)
            elem.clear()
            if pmid is not None:
                return pmid
    except ET.ParseError:
        return None
    return None


def build_pmid_to_pmcid_map(extracted_dirs):
    pmid_to_pmcids = defaultdict(set)
    available_pmcids = set()
    xml_file_count = 0
    missing_pmid_xml_count = 0

    for extracted_dir in extracted_dirs:
        root = Path(extracted_dir)
        if not root.exists():
            continue
        for xml_path in root.rglob("*.xml"):
            xml_file_count += 1
            pmcid = xml_path.stem
            available_pmcids.add(pmcid)
            pmid = extract_pmid_from_xml(xml_path)
            if pmid is None:
                missing_pmid_xml_count += 1
                continue
            pmid_to_pmcids[pmid].add(pmcid)

    return {
        "pmid_to_pmcids": {pmid: sorted(pmcids) for pmid, pmcids in pmid_to_pmcids.items()},
        "available_pmcids": sorted(available_pmcids),
        "xml_file_count": xml_file_count,
        "missing_pmid_xml_count": missing_pmid_xml_count,
    }


def load_trajectories(summary_path):
    with open(summary_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    trajectories = []
    for record in data.get("records", []):
        current_node = {
            "role": "current",
            "pmid": to_int(record.get("pmid")),
            "title": record.get("title"),
            "year": to_int(record.get("year")),
        }
        prior_nodes = []
        for prior in record.get("prior_guidelines", []):
            prior_nodes.append(
                {
                    "role": "prior",
                    "pmid": to_int(prior.get("pmid")),
                    "title": prior.get("title"),
                    "year": to_int(prior.get("year")),
                }
            )
        trajectories.append(
            {
                "category": record.get("category"),
                "category_label": record.get("category_label"),
                "pmc_id": record.get("pmc_id"),
                "title": record.get("title"),
                "topic": record.get("topic"),
                "year": to_int(record.get("year")),
                "node_count": 1 + len(prior_nodes),
                "nodes": [current_node] + prior_nodes,
            }
        )
    return trajectories


def analyze_trajectories(trajectories, pmid_to_pmcids):
    results = []
    pmid_lookup = {to_int(k): v for k, v in pmid_to_pmcids.items()}

    for trajectory in trajectories:
        found_count = 0
        missing_count = 0
        reason_counts = Counter()
        node_results = []

        for node in trajectory["nodes"]:
            pmid = node["pmid"]
            pmcids = []
            status = "found"
            if pmid in (None, 0):
                status = "missing_pmid"
                reason_counts[status] += 1
                missing_count += 1
            else:
                pmcids = pmid_lookup.get(pmid, [])
                if pmcids:
                    found_count += 1
                else:
                    status = "pmid_not_in_extracted_db"
                    reason_counts[status] += 1
                    missing_count += 1

            node_results.append(
                {
                    "role": node["role"],
                    "pmid": pmid,
                    "resolved_pmcids": pmcids,
                    "status": status,
                    "title": node["title"],
                    "year": node["year"],
                }
            )

        results.append(
            {
                "category": trajectory["category"],
                "category_label": trajectory["category_label"],
                "pmc_id": trajectory["pmc_id"],
                "title": trajectory["title"],
                "topic": trajectory["topic"],
                "year": trajectory["year"],
                "node_count": trajectory["node_count"],
                "found_paper_count": found_count,
                "missing_paper_count": missing_count,
                "all_nodes_found": missing_count == 0,
                "missing_fraction": missing_count / trajectory["node_count"] if trajectory["node_count"] else 0.0,
                "missing_reason_counts": dict(reason_counts),
                "nodes": node_results,
            }
        )

    results.sort(
        key=lambda row: (
            -row["missing_paper_count"],
            -row["missing_fraction"],
            row["category"] or "",
            row["pmc_id"] or "",
        )
    )
    return results


def write_outputs(results, map_summary, output_json, output_csv):
    all_nodes_found_trajectory_count = sum(1 for row in results if row["all_nodes_found"])
    summary = {
        "mapping_summary": {
            "xml_file_count": map_summary["xml_file_count"],
            "missing_pmid_xml_count": map_summary["missing_pmid_xml_count"],
            "mapped_pmid_count": len(map_summary["pmid_to_pmcids"]),
            "available_pmcid_count": len(map_summary["available_pmcids"]),
        },
        "trajectory_count": len(results),
        "all_nodes_found_trajectory_count": all_nodes_found_trajectory_count,
        "all_nodes_found_trajectory_fraction": (
            all_nodes_found_trajectory_count / len(results) if results else 0.0
        ),
        "results": results,
    }

    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    fieldnames = [
        "category",
        "category_label",
        "pmc_id",
        "title",
        "topic",
        "year",
        "node_count",
        "found_paper_count",
        "missing_paper_count",
        "all_nodes_found",
        "missing_fraction",
        "missing_reason_counts",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, escapechar="\\")
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "category": row["category"],
                    "category_label": row["category_label"],
                    "pmc_id": row["pmc_id"],
                    "title": row["title"],
                    "topic": row["topic"],
                    "year": row["year"],
                    "node_count": row["node_count"],
                    "found_paper_count": row["found_paper_count"],
                    "missing_paper_count": row["missing_paper_count"],
                    "all_nodes_found": row["all_nodes_found"],
                    "missing_fraction": f"{row['missing_fraction']:.6f}",
                    "missing_reason_counts": json.dumps(row["missing_reason_counts"], ensure_ascii=False, sort_keys=True),
                }
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count how many papers in each trajectory cannot be resolved into the local extracted PMC OA database."
    )
    parser.add_argument(
        "--trajectory-summary",
        default=DEFAULT_TRAJECTORY_SUMMARY,
        help="Path to the trajectory summary JSON to analyze.",
    )
    parser.add_argument(
        "--extracted-dir",
        action="append",
        dest="extracted_dirs",
        default=None,
        help="Extracted PMC OA directory. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    extracted_dirs = args.extracted_dirs or DEFAULT_EXTRACTED_DIRS
    map_summary = build_pmid_to_pmcid_map(extracted_dirs)
    trajectories = load_trajectories(args.trajectory_summary)
    results = analyze_trajectories(trajectories, map_summary["pmid_to_pmcids"])

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    write_outputs(results, map_summary, args.output_json, args.output_csv)

    all_nodes_found_trajectory_count = sum(1 for row in results if row["all_nodes_found"])

    print(f"Wrote JSON to {args.output_json}")
    print(f"Wrote CSV to {args.output_csv}")
    print(f"Analyzed {len(results)} trajectories")
    print(f"Trajectories with all nodes found: {all_nodes_found_trajectory_count}")


if __name__ == "__main__":
    main()
