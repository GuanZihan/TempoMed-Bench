#!/usr/bin/env python3
"""Verify title-year mismatches for guideline questions and optionally update them.

The question file is expected to contain repeated JSON objects separated by
whitespace, even if the filename ends with .jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import requests

NCBI_EUTILS_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
QUESTION_YEAR_PATTERN_TEMPLATE = r"(guideline\s+issued\s+in\s+){}\b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Query PubMed titles by PMID and print cases where a year found in "
            "the title differs from the recorded Year_current or Year_prior. "
            "Verification mode can update the stored year fields and, for the "
            "current guideline, the year mentioned in the question and explanation."
        )
    )
    parser.add_argument(
        "--questions-file",
        required=True,
        help="Path to the generated question file containing repeated JSON objects.",
    )
    parser.add_argument(
        "--cache-file",
        default=None,
        help=(
            "Optional JSON cache for PMID-to-title lookups. Defaults to "
            "<questions-file>.pmid_title_cache.json."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Number of PMIDs to request per PubMed ESummary call.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.34,
        help="Delay between NCBI requests when API key is not used.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP timeout for each NCBI request.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("NCBI_API_KEY"),
        help="Optional NCBI API key. Defaults to NCBI_API_KEY env var if set.",
    )
    parser.add_argument(
        "--email",
        default=os.environ.get("NCBI_EMAIL"),
        help="Optional contact email sent to NCBI. Defaults to NCBI_EMAIL env var.",
    )
    parser.add_argument(
        "--tool-name",
        default="medical_knowledge_update_title_year_checker",
        help="Tool name sent to NCBI E-utilities.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Show mismatches one by one and ask whether to apply updates.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help=(
            "Where to write the updated question file in verification mode. "
            "Defaults to overwriting --questions-file."
        ),
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Write a .bak copy of the original file before saving updates.",
    )
    return parser.parse_args()


def iter_json_objects(path: Path) -> Iterator[dict]:
    decoder = json.JSONDecoder()
    text = path.read_text()
    index = 0
    length = len(text)

    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        obj, next_index = decoder.raw_decode(text, index)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected a JSON object at character offset {index}")
        yield obj
        index = next_index


def load_cache(cache_path: Path) -> Dict[str, str]:
    if not cache_path.exists():
        return {}
    with cache_path.open() as infile:
        data = json.load(infile)
    if not isinstance(data, dict):
        raise ValueError(f"Cache file is not a JSON object: {cache_path}")
    return {str(key): str(value) for key, value in data.items()}


def save_cache(cache_path: Path, cache: Dict[str, str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as outfile:
        json.dump(dict(sorted(cache.items())), outfile, indent=2, sort_keys=True)


def chunked(items: List[str], size: int) -> Iterator[List[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def fetch_titles(pmids: Iterable[str], args: argparse.Namespace, cache: Dict[str, str]) -> Dict[str, str]:
    unique_pmids = sorted({pmid for pmid in pmids if pmid and pmid not in cache})
    if not unique_pmids:
        return cache

    session = requests.Session()
    headers = {"User-Agent": f"{args.tool_name}/1.0"}

    for batch in chunked(unique_pmids, args.batch_size):
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "json",
            "tool": args.tool_name,
        }
        if args.api_key:
            params["api_key"] = args.api_key
        if args.email:
            params["email"] = args.email

        response = session.get(
            NCBI_EUTILS_ESUMMARY_URL,
            params=params,
            headers=headers,
            timeout=args.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        result = payload.get("result", {})

        for pmid in batch:
            summary = result.get(str(pmid), {})
            title = summary.get("title", "")
            if not title:
                title = summary.get("sorttitle", "")
            cache[str(pmid)] = title.strip()

        save_cache(Path(args.cache_file), cache)
        if not args.api_key:
            time.sleep(args.sleep_seconds)

    return cache


def find_title_years(title: str) -> List[int]:
    return sorted({int(match.group(0)) for match in YEAR_PATTERN.finditer(title)})


def collect_mismatches(records: Iterable[dict], title_cache: Dict[str, str]) -> List[dict]:
    mismatches: List[dict] = []

    for record_position, record in enumerate(records):
        idx = record.get("idx")
        for pmid_field, year_field, label in (
            ("PMID_current", "Year_current", "current"),
            ("PMID_prior", "Year_prior", "prior"),
        ):
            pmid = str(record.get(pmid_field, "")).strip()
            recorded_year = record.get(year_field)
            if not pmid or recorded_year in (None, ""):
                continue

            title = title_cache.get(pmid, "")
            title_years = find_title_years(title)
            mismatching_years = [year for year in title_years if year != int(recorded_year)]
            if mismatching_years:
                mismatches.append(
                    {
                        "record_position": record_position,
                        "idx": idx,
                        "label": label,
                        "pmid": pmid,
                        "year_field": year_field,
                        "recorded_year": int(recorded_year),
                        "title_years": title_years,
                        "mismatching_title_years": mismatching_years,
                        "title": title,
                    }
                )

    return mismatches


def prompt_for_choice(mismatch: dict) -> int | None:
    years = mismatch["mismatching_title_years"]
    while True:
        print()
        print(f"idx={mismatch['idx']} label={mismatch['label']} pmid={mismatch['pmid']}")
        print(f"recorded year: {mismatch['recorded_year']}")
        print(f"title years: {mismatch['title_years']}")
        print(f"title: {mismatch['title']}")
        if len(years) == 1:
            candidate = years[0]
            answer = input(f"Apply update to {candidate}? [y]es/[n]o/[q]uit: ").strip().lower()
            if answer in {"y", "yes"}:
                return candidate
            if answer in {"n", "no", "", "skip", "s"}:
                return None
            if answer in {"q", "quit"}:
                raise KeyboardInterrupt
            print("Please answer y, n, or q.")
            continue

        answer = input(
            "Enter one title year to apply, or [s]kip, or [q]uit: "
        ).strip().lower()
        if answer in {"s", "skip", "", "n", "no"}:
            return None
        if answer in {"q", "quit"}:
            raise KeyboardInterrupt
        try:
            selected_year = int(answer)
        except ValueError:
            print("Please enter one of the listed years, s, or q.")
            continue
        if selected_year not in years:
            print("Selected year is not in the title year list.")
            continue
        return selected_year


def replace_first_year(text: str, old_year: int, new_year: int) -> Tuple[str, bool]:
    pattern = re.compile(rf"\b{old_year}\b")
    updated_text, count = pattern.subn(str(new_year), text, count=1)
    return updated_text, count > 0


def replace_question_guideline_year(text: str, old_year: int, new_year: int) -> Tuple[str, bool]:
    pattern = re.compile(QUESTION_YEAR_PATTERN_TEMPLATE.format(old_year), flags=re.IGNORECASE)
    updated_text, count = pattern.subn(rf"\g<1>{new_year}", text, count=1)
    return updated_text, count > 0


def apply_update_to_record(record: dict, mismatch: dict, new_year: int) -> List[str]:
    notes: List[str] = []
    old_year = int(mismatch["recorded_year"])
    year_field = mismatch["year_field"]
    record[year_field] = new_year
    notes.append(f"updated {year_field} from {old_year} to {new_year}")

    if mismatch["label"] == "current":
        question = record.get("Question")
        if isinstance(question, str):
            updated_question, changed = replace_question_guideline_year(question, old_year, new_year)
            if not changed:
                updated_question, changed = replace_first_year(question, old_year, new_year)
            if changed:
                record["Question"] = updated_question
                notes.append("updated Question year")
            else:
                notes.append("Question year not updated automatically")

        answer = record.get("Answer")
        if isinstance(answer, dict):
            explanation = answer.get("Explanation")
            if isinstance(explanation, str):
                updated_explanation, changed = replace_first_year(explanation, old_year, new_year)
                if changed:
                    answer["Explanation"] = updated_explanation
                    notes.append("updated Explanation year")
                else:
                    notes.append("Explanation year not updated automatically")

    return notes


def write_records(path: Path, records: List[dict], backup: bool) -> None:
    if backup and path.exists():
        backup_path = path.with_suffix(path.suffix + ".bak")
        backup_path.write_text(path.read_text())

    rendered = "\n\n\n".join(json.dumps(record, ensure_ascii=False, indent=2) for record in records)
    rendered += "\n"
    path.write_text(rendered)


def run_verification(records: List[dict], mismatches: List[dict], output_path: Path, backup: bool) -> int:
    updates_applied = 0

    try:
        for mismatch in mismatches:
            record = records[mismatch["record_position"]]
            selected_year = prompt_for_choice(mismatch)
            if selected_year is None:
                continue
            notes = apply_update_to_record(record, mismatch, selected_year)
            updates_applied += 1
            print("Applied update:")
            for note in notes:
                print(f"  - {note}")
    except KeyboardInterrupt:
        print("\nVerification interrupted. Saving applied updates so far.", file=sys.stderr)

    if updates_applied:
        write_records(output_path, records, backup=backup)

    print(
        json.dumps(
            {
                "mismatches_reviewed": len(mismatches),
                "updates_applied": updates_applied,
                "output_file": str(output_path) if updates_applied else None,
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )
    return updates_applied


def main() -> int:
    args = parse_args()
    questions_path = Path(args.questions_file).expanduser().resolve()
    cache_path = (
        Path(args.cache_file).expanduser().resolve()
        if args.cache_file
        else questions_path.with_name(f"{questions_path.name}.pmid_title_cache.json")
    )
    output_path = (
        Path(args.output_file).expanduser().resolve()
        if args.output_file
        else questions_path
    )
    args.cache_file = str(cache_path)

    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file does not exist: {questions_path}")

    records = list(iter_json_objects(questions_path))
    pmids: List[str] = []
    for record in records:
        for field in ("PMID_current", "PMID_prior"):
            pmid = str(record.get(field, "")).strip()
            if pmid:
                pmids.append(pmid)

    cache = load_cache(cache_path)
    cache = fetch_titles(pmids, args, cache)
    mismatches = collect_mismatches(records, cache)

    if args.verify:
        run_verification(records, mismatches, output_path, backup=args.backup)
        return 0

    for mismatch in mismatches:
        print(json.dumps(mismatch, ensure_ascii=False))

    print(
        json.dumps(
            {
                "records_checked": len(records),
                "unique_pmids": len(set(pmids)),
                "mismatches_found": len(mismatches),
                "cache_file": str(cache_path),
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
