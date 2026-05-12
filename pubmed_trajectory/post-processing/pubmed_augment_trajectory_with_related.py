import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TypedDict
from xml.etree import ElementTree as ET

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(root_dir)
sys.path.append(root_dir)

from utils import config

ELINK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PMCID_CONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"

DEFAULT_INPUT_DIR = os.path.join(project_root, "other_guideline_trajectory_2026_relaxed")
DEFAULT_OUTPUT_DIR = os.path.join(project_root, "other_guideline_trajectory_2026_relaxed_augmented")


_THREAD_LOCAL = threading.local()


class PriorVersionDecision(BaseModel):
    is_prior_version: bool = Field(description="Whether the candidate is a prior version of the current guideline.")
    confidence: int = Field(description="Confidence from 0 to 100.")
    same_topic: bool = Field(description="Whether the candidate addresses the same clinical topic.")
    same_or_related_organization: bool = Field(description="Whether the issuing organization is the same or closely related.")
    is_guideline_like: bool = Field(description="Whether the candidate appears to be a guideline/consensus/statement/recommendation paper.")
    reason: str = Field(description="Short rationale for the decision.")


class ClassifierState(TypedDict):
    system_prompt: str
    user_prompt: str
    decision: Dict[str, Any]


@dataclass
class CandidatePaper:
    pmid: str
    score: Optional[float]
    title: str
    year: Optional[int]
    abstract: str
    organizations: List[str]
    raw_affiliations: List[str]

    def to_prompt_block(self) -> str:
        org_text = "; ".join(self.organizations[:8]) if self.organizations else ""
        return (
            f"Candidate PMID: {self.pmid}\n"
            f"Candidate title: {self.title or 'N/A'}\n"
            f"Candidate publication year: {self.year if self.year is not None else 'N/A'}\n"
            f"Candidate organizations/affiliations: {org_text or 'N/A'}\n"
            f"Candidate abstract: {self.abstract or 'N/A'}\n"
            f"Neighbor score: {self.score if self.score is not None else 'N/A'}"
        )


def build_azure_llm(deployment: Optional[str] = None) -> AzureChatOpenAI:
    chosen_deployment = deployment or config.AZURE_DEPLOYMENT
    return AzureChatOpenAI(
        azure_deployment=chosen_deployment,
        api_key=config.AZURE_OPENAI_API_KEY,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_version=config.AZURE_OPENAI_API_VERSION,
        temperature=0 if chosen_deployment not in ["azure-gpt-5"] else 1,
    )


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(normalize_text(v) for v in value if normalize_text(v))
    return str(value).strip()


def normalize_org_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]
    cleaned: List[str] = []
    seen = set()
    for item in items:
        text = normalize_text(item)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned


_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def extract_year(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = normalize_text(value)
    match = _YEAR_RE.search(text)
    if match:
        return int(match.group(0))
    return None


def chunked(seq: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def safe_request(session: requests.Session, url: str, params: Dict[str, Any], sleep_seconds: float, max_retries: int = 4) -> requests.Response:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = session.get(url, params=params, timeout=60)
            response.raise_for_status()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            return response
        except Exception as exc:  # pragma: no cover - network-dependent
            last_error = exc
            time.sleep(min(6, 1.5 ** attempt))
    raise RuntimeError(f"Request failed for {url} params={params}: {last_error}")


def fetch_neighbor_scores(session: requests.Session, current_pmid: str, max_related: int, sleep_seconds: float) -> List[Dict[str, Any]]:
    response = safe_request(
        session,
        ELINK_URL,
        {
            "dbfrom": "pubmed",
            "db": "pubmed",
            "id": current_pmid,
            "cmd": "neighbor_score",
            "retmode": "json",
        },
        sleep_seconds,
    )
    payload = response.json()
    linksets = payload.get("linksets", [])
    if not linksets:
        return []

    scored = []
    for db_block in linksets[0].get("linksetdbs", []):
        for entry in db_block.get("links", []):
            if isinstance(entry, dict):
                pmid = str(entry.get("id") or "").strip()
                score = entry.get("score")
            else:
                pmid = str(entry).strip()
                score = None
            if pmid and pmid != str(current_pmid):
                scored.append({"pmid": pmid, "score": score})

    deduped = []
    seen = set()
    for item in scored:
        pmid = item["pmid"]
        if pmid in seen:
            continue
        seen.add(pmid)
        deduped.append(item)
    return deduped[:max_related]


def fetch_esummary_map(session: requests.Session, pmids: List[str], sleep_seconds: float) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for batch in chunked(pmids, 100):
        response = safe_request(
            session,
            ESUMMARY_URL,
            {"db": "pubmed", "id": ",".join(batch), "retmode": "json"},
            sleep_seconds,
        )
        payload = response.json().get("result", {})
        for pmid in batch:
            record = payload.get(str(pmid), {})
            if record:
                summary[str(pmid)] = record
    return summary


def fetch_efetch_details(session: requests.Session, pmids: List[str], sleep_seconds: float) -> Dict[str, Dict[str, Any]]:
    details: Dict[str, Dict[str, Any]] = {}
    for batch in chunked(pmids, 50):
        response = safe_request(
            session,
            EFETCH_URL,
            {"db": "pubmed", "id": ",".join(batch), "retmode": "xml", "rettype": "abstract"},
            sleep_seconds,
        )
        root = ET.fromstring(response.text)
        for article in root.findall(".//PubmedArticle"):
            pmid_text = article.findtext(".//MedlineCitation/PMID")
            if not pmid_text:
                continue
            abstract_texts = []
            for elem in article.findall(".//Abstract/AbstractText"):
                text = " ".join(t.strip() for t in elem.itertext() if t and t.strip())
                label = elem.attrib.get("Label")
                if label and text:
                    text = f"{label}: {text}"
                if text:
                    abstract_texts.append(text)
            affiliations = []
            for aff in article.findall(".//AffiliationInfo/Affiliation"):
                text = " ".join(t.strip() for t in aff.itertext() if t and t.strip())
                if text:
                    affiliations.append(text)
            details[str(pmid_text)] = {
                "abstract": "\n".join(abstract_texts),
                "affiliations": dedupe_preserve_order(affiliations),
            }
    return details


def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output






def normalize_title_for_match(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return text.strip()


def resolve_current_pmid_from_source_pmcid(
    session: requests.Session,
    source_pmcid: str,
    record_title: str,
    sleep_seconds: float,
) -> Dict[str, Any]:
    if not source_pmcid:
        return {
            "status": "unresolved_missing_source_pmcid",
            "resolved": False,
            "resolved_pmid": None,
        }

    pmcid = source_pmcid if source_pmcid.upper().startswith("PMC") else f"PMC{source_pmcid}"
    response = safe_request(
        session,
        PMCID_CONV_URL,
        {"ids": pmcid, "format": "json"},
        sleep_seconds,
    )
    records = response.json().get("records", [])
    if not records:
        return {
            "status": "unresolved_no_idconv_record",
            "resolved": False,
            "source_pmcid": pmcid,
            "resolved_pmid": None,
        }

    converted = records[0]
    pmid = str(converted.get("pmid") or "").strip()
    if not pmid:
        return {
            "status": "unresolved_no_pmid_from_pmcid",
            "resolved": False,
            "source_pmcid": pmcid,
            "resolved_pmid": None,
        }

    summary = fetch_esummary_map(session, [pmid], sleep_seconds).get(pmid, {})
    resolved_title = normalize_text(summary.get("title"))
    matches = normalize_title_for_match(resolved_title) == normalize_title_for_match(record_title)
    return {
        "status": "resolved_from_source_pmcid" if matches else "pmcid_to_pmid_title_mismatch",
        "resolved": matches,
        "source_pmcid": pmcid,
        "resolved_pmid": pmid if matches else None,
        "candidate_pmid": pmid,
        "resolved_title": resolved_title,
        "title_matches_record": matches,
    }


def build_candidate_papers(session: requests.Session, related_items: List[Dict[str, Any]], sleep_seconds: float) -> List[CandidatePaper]:
    pmids = [item["pmid"] for item in related_items]
    summary_map = fetch_esummary_map(session, pmids, sleep_seconds)
    details_map = fetch_efetch_details(session, pmids, sleep_seconds)
    papers: List[CandidatePaper] = []
    score_map = {item["pmid"]: item.get("score") for item in related_items}
    for pmid in pmids:
        summary = summary_map.get(pmid, {})
        details = details_map.get(pmid, {})
        title = normalize_text(summary.get("title"))
        year = extract_year(summary.get("pubdate"))
        affiliations = details.get("affiliations", [])
        organizations = affiliations[:10]
        papers.append(
            CandidatePaper(
                pmid=pmid,
                score=score_map.get(pmid),
                title=title,
                year=year,
                abstract=normalize_text(details.get("abstract")),
                organizations=organizations,
                raw_affiliations=affiliations,
            )
        )
    return papers


def build_existing_prior_pmids(record: Dict[str, Any]) -> set[str]:
    pmids = set()
    for prior in record.get("prior_guidelines", []) or []:
        pmid = prior.get("PMID")
        if pmid not in (None, "", 0, "0"):
            pmids.add(str(pmid))
    return pmids


def build_llm_prompt(record: Dict[str, Any], candidate: CandidatePaper) -> str:
    current_orgs = normalize_org_list(record.get("Organization"))
    prior_lines = []
    for prior in record.get("prior_guidelines", []) or []:
        prior_lines.append(
            f"- PMID={prior.get('PMID')} | Year={prior.get('year')} | Organization={normalize_text(prior.get('Organization'))} | Title={normalize_text(prior.get('title'))}"
        )
    existing_priors = "\n".join(prior_lines) if prior_lines else "None"
    return (
        "Determine whether the candidate paper is a PRIOR VERSION of the current guideline paper.\n\n"
        "Return JSON only with these keys:\n"
        "is_prior_version: true/false\n"
        "confidence: integer 0-100\n"
        "same_topic: true/false\n"
        "same_or_related_organization: true/false\n"
        "is_guideline_like: true/false\n"
        "reason: short string\n\n"
        "Decision rule for true:\n"
        "- candidate is older than current guideline\n"
        "- candidate is on the same clinical topic\n"
        "- candidate appears to be a guideline/consensus/statement/recommendation/practice guidance paper\n"
        "- candidate is likely an earlier version or closely preceding edition from the same or clearly related issuing organization\n"
        "- do not mark true for general reviews, background studies, or unrelated society papers\n\n"
        f"Current guideline PMID: {record.get('PMID')}\n"
        f"Current guideline title: {normalize_text(record.get('Title'))}\n"
        f"Current guideline year: {record.get('year_of_current_guidance')}\n"
        f"Current guideline topic: {normalize_text(record.get('Topic'))}\n"
        f"Current guideline organizations: {current_orgs}\n"
        f"Known existing prior guidelines:\n{existing_priors}\n\n"
        f"{candidate.to_prompt_block()}\n"
    )


def build_classifier_graph(llm: AzureChatOpenAI):
    structured_llm = llm.with_structured_output(PriorVersionDecision)

    def classify_node(state: ClassifierState) -> Dict[str, Any]:
        decision = structured_llm.invoke(
            [
                SystemMessage(content=state["system_prompt"]),
                HumanMessage(content=state["user_prompt"]),
            ]
        )
        if isinstance(decision, PriorVersionDecision):
            payload = decision.model_dump()
        else:
            payload = dict(decision)
        return {"decision": payload}

    graph = StateGraph(ClassifierState)
    graph.add_node("classify", classify_node)
    graph.set_entry_point("classify")
    graph.add_edge("classify", END)
    return graph.compile()


def classify_candidates_azure(classifier_graph: Any, record: Dict[str, Any], candidates: List[CandidatePaper]) -> List[Dict[str, Any]]:
    system_prompt = (
        "You are a strict biomedical literature linkage verifier. "
        "Be conservative about claiming prior-version relationships and follow the structured output schema exactly."
    )
    results = []
    for candidate in candidates:
        state = {
            "system_prompt": system_prompt,
            "user_prompt": build_llm_prompt(record, candidate),
            "decision": {},
        }
        output_state = classifier_graph.invoke(state)
        results.append(output_state["decision"])
    return results


def build_augmented_prior_entry(candidate: CandidatePaper, decision: Dict[str, Any], current_record: Dict[str, Any]) -> Dict[str, Any]:
    organizations: Any
    if candidate.organizations:
        organizations = candidate.organizations if len(candidate.organizations) > 1 else candidate.organizations[0]
    else:
        organizations = ""
    return {
        "year": candidate.year,
        "PMID": int(candidate.pmid) if candidate.pmid.isdigit() else candidate.pmid,
        "Organization": organizations,
        "title": candidate.title,
        "reason": normalize_text(decision.get("reason")),
        "source": "related_work_llm_augmentation",
        "augmentation_metadata": {
            "current_pmid": current_record.get("PMID"),
            "current_pmid_used_for_augmentation": current_record.get("_augmentation_current_pmid"),
            "neighbor_score": candidate.score,
            "same_topic": bool(decision.get("same_topic")),
            "same_or_related_organization": bool(decision.get("same_or_related_organization")),
            "is_guideline_like": bool(decision.get("is_guideline_like")),
            "confidence": decision.get("confidence"),
            "raw_affiliations": candidate.raw_affiliations[:10],
        },
    }


def should_accept_candidate(candidate: CandidatePaper, decision: Dict[str, Any], current_year: Optional[int]) -> bool:
    if not bool(decision.get("is_prior_version")):
        return False
    confidence = decision.get("confidence")
    try:
        confidence_value = int(confidence)
    except Exception:
        confidence_value = 0
    if confidence_value < 60:
        return False
    if current_year is not None and candidate.year is not None and candidate.year >= current_year:
        return False
    return True


def augment_record(
    record: Dict[str, Any],
    session: requests.Session,
    classifier_graph: Any,
    max_related: int,
    sleep_seconds: float,
    source_pmcid: Optional[str] = None,
) -> Dict[str, Any]:
    current_pmid = record.get("PMID")
    resolved_current_pmid = None if current_pmid in (None, "", 0, "0") else str(current_pmid)
    pmid_resolution = {
        "status": "original_pmid_present" if resolved_current_pmid else "requires_source_pmcid_resolution",
        "resolved": bool(resolved_current_pmid),
        "original_pmid": current_pmid,
        "resolved_pmid": resolved_current_pmid,
        "source_pmcid": source_pmcid,
    }

    if not resolved_current_pmid:
        pmid_resolution = resolve_current_pmid_from_source_pmcid(
            session,
            source_pmcid=source_pmcid or "",
            record_title=normalize_text(record.get("Title")),
            sleep_seconds=sleep_seconds,
        )
        pmid_resolution["original_pmid"] = current_pmid
        resolved_value = pmid_resolution.get("resolved_pmid")
        resolved_current_pmid = str(resolved_value) if resolved_value not in (None, "") else None

    if not resolved_current_pmid:
        result = dict(record)
        result["augmentation_metadata"] = {
            "status": "skipped_missing_pmid",
            "added_prior_guidelines": 0,
            "current_pmid_resolution": pmid_resolution,
            "current_pmid_used_for_augmentation": None,
        }
        return result

    existing_prior_pmids = build_existing_prior_pmids(record)
    related_items = fetch_neighbor_scores(session, resolved_current_pmid, max_related, sleep_seconds)
    candidate_papers = build_candidate_papers(session, related_items, sleep_seconds)
    filtered_candidates = [
        candidate
        for candidate in candidate_papers
        if candidate.pmid != resolved_current_pmid and candidate.pmid not in existing_prior_pmids
    ]

    if not filtered_candidates:
        result = dict(record)
        result["augmentation_metadata"] = {
            "status": "no_new_candidates",
            "related_candidates_considered": 0,
            "added_prior_guidelines": 0,
            "current_pmid_resolution": pmid_resolution,
            "current_pmid_used_for_augmentation": resolved_current_pmid,
        }
        return result

    prompt_record = dict(record)
    prompt_record["PMID"] = resolved_current_pmid
    prompt_record["_augmentation_current_pmid"] = resolved_current_pmid
    decisions = classify_candidates_azure(classifier_graph, prompt_record, filtered_candidates)

    current_year = extract_year(record.get("year_of_current_guidance"))
    accepted_entries = []
    candidate_audit = []
    for candidate, decision in zip(filtered_candidates, decisions):
        accepted = should_accept_candidate(candidate, decision, current_year)
        candidate_audit.append(
            {
                "pmid": candidate.pmid,
                "title": candidate.title,
                "year": candidate.year,
                "neighbor_score": candidate.score,
                "accepted": accepted,
                "decision": decision,
            }
        )
        if accepted:
            accepted_entries.append(build_augmented_prior_entry(candidate, decision, prompt_record))

    result = dict(record)
    result["_augmentation_current_pmid"] = resolved_current_pmid
    priors = list(record.get("prior_guidelines", []) or [])
    priors.extend(accepted_entries)
    priors.sort(key=lambda item: (extract_year(item.get("year")) or 9999, normalize_text(item.get("title")).lower()))
    result["prior_guidelines"] = priors
    result["augmentation_metadata"] = {
        "status": "augmented" if accepted_entries else "reviewed_no_additions",
        "current_pmid_resolution": pmid_resolution,
        "current_pmid_used_for_augmentation": resolved_current_pmid,
        "related_candidates_retrieved": len(related_items),
        "related_candidates_considered": len(filtered_candidates),
        "added_prior_guidelines": len(accepted_entries),
        "added_prior_pmids": [entry["PMID"] for entry in accepted_entries],
        "candidate_audit": candidate_audit,
    }
    result.pop("_augmentation_current_pmid", None)
    return result


def load_record(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def collect_input_files(input_dir: Path) -> List[Path]:
    return sorted(input_dir.glob("*/*.json"), reverse=True)


def get_thread_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": "guideline-trajectory-augmentation/1.0"})
        _THREAD_LOCAL.session = session
    return session


def get_thread_classifier_graph(azure_deployment: Optional[str]) -> Any:
    deployment_key = azure_deployment or config.AZURE_DEPLOYMENT
    cache_key = f"classifier_graph::{deployment_key}"
    classifier_graph = getattr(_THREAD_LOCAL, cache_key, None)
    if classifier_graph is None:
        azure_llm = build_azure_llm(azure_deployment)
        classifier_graph = build_classifier_graph(azure_llm)
        setattr(_THREAD_LOCAL, cache_key, classifier_graph)
    return classifier_graph


def process_single_file(
    path: Path,
    input_dir: Path,
    output_dir: Path,
    azure_deployment: Optional[str],
    max_related: int,
    sleep_seconds: float,
    force: bool,
) -> Dict[str, Any]:
    relative = path.relative_to(input_dir)
    out_path = output_dir / relative
    if out_path.exists() and not force:
        augmented = load_record(out_path)
        original = load_record(path)
        return build_summary_row(path, out_path, original, augmented)

    original = load_record(path)
    session = get_thread_session()
    classifier_graph = get_thread_classifier_graph(azure_deployment)
    try:
        augmented = augment_record(
            original,
            session=session,
            classifier_graph=classifier_graph,
            max_related=max_related,
            sleep_seconds=sleep_seconds,
            source_pmcid=path.stem,
        )
    except Exception as exc:
        print(exc)
        augmented = dict(original)
        augmented["augmentation_metadata"] = {
            "status": "error",
            "error": str(exc),
            "added_prior_guidelines": 0,
        }
    write_json(out_path, augmented)
    return build_summary_row(path, out_path, original, augmented)


def build_summary_row(source_path: Path, output_path: Path, original: Dict[str, Any], augmented: Dict[str, Any]) -> Dict[str, Any]:
    meta = augmented.get("augmentation_metadata", {})
    return {
        "source_file": str(source_path),
        "output_file": str(output_path),
        "pmid": original.get("PMID"),
        "title": original.get("Title"),
        "year": original.get("year_of_current_guidance"),
        "status": meta.get("status"),
        "original_prior_count": len(original.get("prior_guidelines", []) or []),
        "augmented_prior_count": len(augmented.get("prior_guidelines", []) or []),
        "added_prior_guidelines": meta.get("added_prior_guidelines", 0),
        "added_prior_pmids": meta.get("added_prior_pmids", []),
        "related_candidates_retrieved": meta.get("related_candidates_retrieved", 0),
        "related_candidates_considered": meta.get("related_candidates_considered", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment guideline trajectories using PubMed related-work candidates and an LLM verifier.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Input trajectory directory.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for augmented trajectories.")
    parser.add_argument("--azure-deployment", default=None, help="Optional Azure deployment override.")
    parser.add_argument("--max-related", type=int, default=25, help="Maximum related PubMed candidates per current PMID.")
    parser.add_argument("--limit-files", type=int, default=None, help="Optional cap on number of trajectory JSON files to process.")
    parser.add_argument("--sleep-seconds", type=float, default=0.34, help="Delay after each E-utilities request.")
    parser.add_argument("--max-workers", type=int, default=4, help="Number of files to process in parallel.")
    parser.add_argument("--force", action="store_true", help="Overwrite output files even if they already exist.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = collect_input_files(input_dir)
    if args.limit_files is not None:
        files = files[: args.limit_files]

    summary_rows = []
    total_added = 0
    max_workers = max(1, int(args.max_workers))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(
                process_single_file,
                path,
                input_dir,
                output_dir,
                args.azure_deployment,
                args.max_related,
                args.sleep_seconds,
                args.force,
            ): path
            for path in files
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(future_to_path), desc="Augment trajectories"):
            summary_row = future.result()
            summary_rows.append(summary_row)
            total_added += int(summary_row.get("added_prior_guidelines", 0) or 0)

    summary_rows.sort(key=lambda row: row["source_file"], reverse=True)

    summary_path = output_dir / "augmentation_summary.json"
    audit_path = output_dir / "augmentation_summary.jsonl"
    write_json(
        summary_path,
        {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "llm_backend": "azure",
            "azure_deployment": args.azure_deployment or config.AZURE_DEPLOYMENT,
            "processed_files": len(summary_rows),
            "total_added_prior_guidelines": total_added,
            "rows": summary_rows,
        },
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8") as f:
        for row in summary_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Processed files: {len(summary_rows)}")
    print(f"Total added prior guidelines: {total_added}")
    print(f"Augmented trajectories saved to: {output_dir}")
    print(f"Summary JSON: {summary_path}")
    print(f"Summary JSONL: {audit_path}")


if __name__ == "__main__":
    main()
