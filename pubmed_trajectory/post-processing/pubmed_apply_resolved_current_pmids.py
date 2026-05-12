import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET_DIR = PROJECT_ROOT / "comm_guideline_trajectory_2026_relaxed_augmented"
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def safe_request(session: requests.Session, url: str, params: Dict[str, Any], sleep_seconds: float = 0.34, max_retries: int = 4) -> requests.Response:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = session.get(url, params=params, timeout=60)
            response.raise_for_status()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            return response
        except Exception as exc:
            last_error = exc
            time.sleep(min(6, 1.5 ** attempt))
    raise RuntimeError(f"Request failed for {url} params={params}: {last_error}")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(normalize_text(v) for v in value if normalize_text(v))
    return str(value).strip()


def normalize_title_for_match(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return text.strip()


def should_update_current_pmid(payload: Dict[str, Any]) -> bool:
    current_pmid = payload.get("PMID")
    meta = payload.get("augmentation_metadata") or {}
    resolution = meta.get("current_pmid_resolution") or {}
    resolved_pmid = resolution.get("resolved_pmid")
    return (
        current_pmid in (0, "0")
        and resolution.get("resolved") is True
        and resolved_pmid not in (None, "", 0, "0")
    )


def updated_current_pmid_value(payload: Dict[str, Any]) -> Any:
    resolved_pmid = (payload.get("augmentation_metadata") or {}).get("current_pmid_resolution", {}).get("resolved_pmid")
    resolved_text = str(resolved_pmid)
    return int(resolved_text) if resolved_text.isdigit() else resolved_pmid


def fetch_esummary_title(session: requests.Session, pmid: str, sleep_seconds: float) -> str:
    response = safe_request(
        session,
        ESUMMARY_URL,
        {"db": "pubmed", "id": pmid, "retmode": "json"},
        sleep_seconds,
    )
    result = response.json().get("result", {})
    return normalize_text((result.get(str(pmid)) or {}).get("title"))


def resolve_pmid_by_exact_title(session: requests.Session, title: str, sleep_seconds: float, cache: Dict[str, Optional[Any]]) -> Optional[Any]:
    normalized_title = normalize_title_for_match(title)
    if not normalized_title:
        return None
    if normalized_title in cache:
        return cache[normalized_title]

    response = safe_request(
        session,
        ESEARCH_URL,
        {
            "db": "pubmed",
            "term": f'"{title}[TITLE]"',
            "retmode": "json",
            "retmax": 10,
        },
        sleep_seconds,
    )
    id_list = response.json().get("esearchresult", {}).get("idlist", [])
    resolved: Optional[Any] = None
    for pmid in id_list:
        candidate_title = fetch_esummary_title(session, str(pmid), sleep_seconds)
        if normalize_title_for_match(candidate_title) == normalized_title:
            resolved = int(pmid) if str(pmid).isdigit() else pmid
            break
    cache[normalized_title] = resolved
    return resolved


def iter_json_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*/*.json"))


def maybe_update_zero_title_pmid(session: requests.Session, entry: Dict[str, Any], title_key: str, sleep_seconds: float, cache: Dict[str, Optional[Any]]) -> Optional[Any]:
    if entry.get("PMID") not in (0, "0"):
        return None
    title = normalize_text(entry.get(title_key))
    if not title:
        return None
    resolved = resolve_pmid_by_exact_title(session, title, sleep_seconds, cache)
    if resolved in (None, "", 0, "0"):
        return None
    entry["PMID"] = resolved
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace current PMID=0 with recorded resolved PMID, then fill remaining zero PMIDs by exact-title PubMed search."
    )
    parser.add_argument(
        "--target-dir",
        default=DEFAULT_TARGET_DIR,
        help="Root directory containing augmented trajectory JSON files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report matching files without modifying them.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.34,
        help="Delay after each PubMed API request.",
    )
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    session = requests.Session()
    session.headers.update({"User-Agent": "guideline-pmid-fixup/1.0"})
    title_cache: Dict[str, Optional[Any]] = {}

    updated_files = 0
    examined = 0
    resolved_current_from_metadata = 0
    resolved_current_from_title = 0
    resolved_prior_from_title = 0

    for path in iter_json_files(target_dir):
        examined += 1
        payload = load_json(path)
        changed = False

        if should_update_current_pmid(payload):
            new_pmid = updated_current_pmid_value(payload)
            old_pmid = payload.get("PMID")
            payload["PMID"] = new_pmid
            changed = True
            resolved_current_from_metadata += 1
            print(f"RESOLVED-CURRENT-META\t{path}\t{old_pmid}\t{new_pmid}")

        resolved_current = maybe_update_zero_title_pmid(
            session=session,
            entry=payload,
            title_key="Title",
            sleep_seconds=args.sleep_seconds,
            cache=title_cache,
        )
        if resolved_current is not None:
            changed = True
            resolved_current_from_title += 1
            print(f"RESOLVED-CURRENT-TITLE\t{path}\t{resolved_current}")

        priors = payload.get("prior_guidelines") or []
        for idx, prior in enumerate(priors):
            if not isinstance(prior, dict):
                continue
            resolved_prior = maybe_update_zero_title_pmid(
                session=session,
                entry=prior,
                title_key="title",
                sleep_seconds=args.sleep_seconds,
                cache=title_cache,
            )
            if resolved_prior is not None:
                changed = True
                resolved_prior_from_title += 1
                print(f"RESOLVED-PRIOR-TITLE\t{path}\tindex={idx}\t{resolved_prior}")

        if changed:
            if not args.dry_run:
                write_json(path, payload)
            updated_files += 1

    print(f"Examined files: {examined}")
    print(f"Updated files: {updated_files}")
    print(f"Resolved current from metadata: {resolved_current_from_metadata}")
    print(f"Resolved current from title search: {resolved_current_from_title}")
    print(f"Resolved prior from title search: {resolved_prior_from_title}")
    print(f"Mode: {'dry-run' if args.dry_run else 'write'}")


if __name__ == "__main__":
    main()
