#!/usr/bin/env python3
"""
Inspect statement-based matrix questions for one trajectory or for a batch of trajectories.

Single trajectory mode:
1. find internal diff pairs whose current/prior PMCIDs are both present in the trajectory
2. generate statement-agreement question templates matching pubmed_generate_questions_2d_matrix.py
3. instantiate those questions across years on the x-axis
4. evaluate one or more LLMs on the resulting yes/no questions
5. write per-trajectory heatmaps and summaries

Batch mode (--run-all):
1. iterate over all trajectories in the input summary
2. run the same per-trajectory pipeline
3. average per-trajectory heatmap cells by target rank and question year
4. save aggregate heatmaps and CSV summaries
"""

import argparse
import asyncio
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import re
import sys
import textwrap
import pdb

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import BadRequestError
from pydantic import BaseModel, Field
from tqdm import tqdm

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
PROJECT_ROOT = ROOT_DIR.parent
sys.path.append(str(ROOT_DIR))

from utils import config

DEFAULT_INPUT_JSON = (
    PROJECT_ROOT / "pubmed_trajectory" / "heatmap_exp" / "results" / "trajectory_missing_papers_clean_summary.json"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "pubmed_trajectory" / "heatmap_exp" / "results" / "inspect_questions"
)

DIFF_SEARCH_ROOTS = [
    PROJECT_ROOT / "results_2026_relaxed_with_post_processing",
    PROJECT_ROOT / "results_2026_relaxed_with_post_processing_flat",
]
TRAJECTORY_DIRS = [
    PROJECT_ROOT / "comm_guideline_trajectory_2026_relaxed_augmented_year_calibrated",
    PROJECT_ROOT / "noncomm_guideline_trajectory_2026_relaxed_augmented_year_calibrated",
    PROJECT_ROOT / "other_guideline_trajectory_2026_relaxed_augmented_year_calibrated",
]
SYSTEM_PROMPT_EVAL = "You are a medical QA model. Answer the binary multiple-choice question."
FALLBACK_ANSWER = "##Answer: INVALID\n##Explanation: Request failed."


class GuidelinePaper(object):
    def __init__(self, pmcid, title, year, topic, organization, pmid, role, status):
        self.pmcid = pmcid
        self.title = title
        self.year = year
        self.topic = topic
        self.organization = organization
        self.pmid = pmid
        self.role = role
        self.status = status

    def to_dict(self):
        return {
            "pmcid": self.pmcid,
            "title": self.title,
            "year": self.year,
            "topic": self.topic,
            "organization": self.organization,
            "pmid": self.pmid,
            "role": self.role,
            "status": self.status,
        }


class AgreementQuestionResponse(BaseModel):
    question_with_prior_statement: str = Field(
        description=(
            "A full question template for the PRIOR recommendation that contains the literal token {year} and follows: "
            "'According to the most recent guideline on xxx published by the xxx on or before {year}, do you agree that xxx?'"
        )
    )
    question_with_current_statement: str = Field(
        description=(
            "A full question template for the CURRENT recommendation that contains the literal token {year} and follows: "
            "'According to the most recent guideline on xxx published by the xxx on or before {year}, do you agree that xxx?'"
        )
    )
    explanation_for_prior_statement: str = Field(
        description=(
            "A concise explanation for the PRIOR-statement question that compares the prior and current recommendations, "
            "mentions the prior/current guideline years, and clarifies why the answer is Yes or No depending on the target year."
        )
    )
    explanation_for_current_statement: str = Field(
        description=(
            "A concise explanation for the CURRENT-statement question that compares the prior and current recommendations, "
            "mentions the prior/current guideline years, and clarifies why the answer is Yes or No depending on the target year."
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and inspect statement-based matrix questions for one trajectory or a batch of trajectories."
    )
    parser.add_argument(
        "--input-json",
        default=DEFAULT_INPUT_JSON,
        help="Input trajectory summary JSON. Supports both trajectory_missing_papers_clean_summary.json and trajectory_browser_all_nodes_found_summary.json.",
    )
    parser.add_argument(
        "--experiment-index",
        type=int,
        default=None,
        help="Index within the input trajectory list for single-trajectory mode.",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all trajectories in the input summary and generate aggregate averaged heatmaps.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output directory.",
    )
    parser.add_argument(
        "--generator-model",
        default=config.AZURE_DEPLOYMENT,
        help="Azure deployment used to generate question templates.",
    )
    parser.add_argument(
        "--eval-model",
        action="append",
        default=None,
        help=(
            "Evaluation model spec. Repeatable. Format: azure:<deployment> or vllm:<model_name>. "
            "Default: azure:%s" % config.AZURE_DEPLOYMENT
        ),
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Optional inclusive first year on the x-axis.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Optional inclusive last year on the x-axis.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=6,
        help="Max Azure concurrency for evaluation.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=320,
        help="Heatmap DPI.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Generate questions and metadata without evaluating models or aggregate heatmaps.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Reuse existing generated_questions.jsonl in the experiment output folder and only run evaluation.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Reuse existing evaluation_results.json files and regenerate summaries/heatmaps without rerunning evaluation.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def normalize_organization(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(normalize_text(v) for v in value if normalize_text(v))
    return normalize_text(value)


def normalize_recommendation(value: Any) -> str:
    return normalize_text(value)


def normalize_reference_text(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"\bguideline\b$", "", text, flags=re.IGNORECASE).strip(" ,;:-")
    return normalize_text(text)


def normalize_statement_text(value: Any) -> str:
    text = normalize_text(value).strip()
    text = text.rstrip(" .!?;:")
    return normalize_text(text)


def recommendations_match(a: str, b: str) -> bool:
    return normalize_recommendation(a).lower() == normalize_recommendation(b).lower()


def slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "artifact"


def resolve_trajectory_path(pmcid: str) -> Optional[Path]:
    for root_name in TRAJECTORY_DIRS:
        matches = list((ROOT_DIR / root_name).rglob("%s.json" % pmcid))
        if matches:
            return matches[0]
    return None


def load_trajectory_record(pmcid: str) -> Dict[str, Any]:
    path = resolve_trajectory_path(pmcid)
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload["_path"] = str(path)
    return payload if isinstance(payload, dict) else {}


def build_trajectory_pmid_index() -> Dict[int, Dict[str, Any]]:
    index: Dict[int, Dict[str, Any]] = {}
    for root_name in TRAJECTORY_DIRS:
        root = ROOT_DIR / root_name
        if not root.exists():
            continue
        for path in root.rglob("PMC*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            pmid = payload.get("PMID")
            if pmid in (None, ""):
                continue
            try:
                pmid = int(pmid)
            except Exception:
                continue
            index[pmid] = {
                "pmcid": path.stem,
                "path": str(path),
                "title": normalize_text(payload.get("Title")),
                "topic": normalize_text(payload.get("Topic")),
                "organization": normalize_organization(payload.get("Organization")),
                "year": payload.get("year_of_current_guidance"),
            }
    return index


def load_experiments(input_json: str) -> List[Dict[str, Any]]:
    with open(input_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "results" in payload:
        experiments = []
        for index, row in enumerate(payload.get("results", [])):
            experiment = dict(row)
            experiment["_experiment_index"] = index
            experiments.append(experiment)
        return experiments

    if isinstance(payload, dict) and "records" in payload:
        pmid_index = build_trajectory_pmid_index()
        experiments = []
        for index, record in enumerate(payload.get("records", [])):
            current_pmcid = record.get("pmc_id")
            current_node = {
                "role": "current",
                "pmid": record.get("pmid"),
                "resolved_pmcids": [current_pmcid] if current_pmcid else [],
                "status": "found" if current_pmcid else "missing_pmcid",
                "title": record.get("title"),
                "year": record.get("year"),
            }
            nodes = [current_node]
            missing_count = 0
            for prior in record.get("prior_guidelines", []):
                pmid = prior.get("pmid")
                resolved = []
                status = "missing_pmid"
                if pmid not in (None, ""):
                    try:
                        pmid_int = int(pmid)
                    except Exception:
                        pmid_int = None
                    if pmid_int in pmid_index:
                        resolved = [pmid_index[pmid_int]["pmcid"]]
                        status = "found"
                    else:
                        status = "pmid_not_resolved_to_pmcid"
                if not resolved:
                    missing_count += 1
                nodes.append(
                    {
                        "role": "prior",
                        "pmid": pmid,
                        "resolved_pmcids": resolved,
                        "status": status,
                        "title": prior.get("title"),
                        "year": prior.get("year"),
                    }
                )
            experiment = {
                "category": record.get("category"),
                "category_label": record.get("category_label"),
                "pmc_id": current_pmcid,
                "title": record.get("title"),
                "topic": record.get("topic"),
                "year": record.get("year"),
                "node_count": len(nodes),
                "found_paper_count": len(nodes) - missing_count,
                "missing_paper_count": missing_count,
                "nodes": nodes,
                "_experiment_index": index,
            }
            if experiment["node_count"] != 2: continue
            experiments.append(experiment)
        return experiments

    raise ValueError("Unsupported input summary schema: expected dict with results or records.")


def build_guideline_papers(experiment: Dict[str, Any]) -> Tuple[Dict[str, GuidelinePaper], List[Dict[str, Any]]]:
    # papers contain the guideline papers for the given trajectory (experiment)
    papers: Dict[str, GuidelinePaper] = {}
    skipped_nodes: List[Dict[str, Any]] = []

    # for guideline nodes in the trajectories
    for node in experiment.get("nodes", []):
        resolved = list(node.get("resolved_pmcids") or [])
        # if they do not have pmcids, then skip
        if not resolved:
            skipped_nodes.append(
                {
                    "title": node.get("title"),
                    "year": node.get("year"),
                    "pmid": node.get("pmid"),
                    "status": node.get("status"),
                    "reason": "no_resolved_pmcid",
                }
            )
            continue
        
        # else, proceed to analyze them
        pmcid = resolved[0]
        # get the record based on the given pmcid of the node
        record = load_trajectory_record(pmcid)
        title = normalize_text(record.get("Title") or node.get("title"))
        year = record.get("year_of_current_guidance") or node.get("year")
        topic = normalize_text(record.get("Topic") or experiment.get("topic"))
        organization = normalize_organization(record.get("Organization") or experiment.get("category_label"))
        pmid = record.get("PMID") if record else node.get("pmid")
        # another filter after retrieving the information
        if not title or year is None:
            skipped_nodes.append(
                {
                    "title": node.get("title"),
                    "year": node.get("year"),
                    "pmid": node.get("pmid"),
                    "status": node.get("status"),
                    "reason": "missing_title_or_year",
                }
            )
            continue
        # construct a guidelinepaper instance
        papers[pmcid] = GuidelinePaper(
            pmcid=pmcid,
            title=title,
            year=int(year),
            topic=topic,
            organization=organization,
            pmid=int(pmid) if pmid not in (None, "") else None,
            role=normalize_text(node.get("role") or "prior"),
            status=normalize_text(node.get("status") or ""),
        )
    return papers, skipped_nodes


def build_target_years(papers: Dict[str, GuidelinePaper], start_year: Optional[int], end_year: Optional[int]) -> List[int]:
    if not papers:
        raise ValueError("No resolved guideline papers were found for the selected experiment.")
    min_year = min(p.year for p in papers.values())
    max_year = max(p.year for p in papers.values())
    first_year = start_year if start_year is not None else min_year
    last_year = end_year if end_year is not None else max_year
    if first_year > last_year:
        raise ValueError("start-year must be <= end-year")
    return list(range(first_year, last_year + 1))


def build_target_rank_map(papers: Dict[str, GuidelinePaper]) -> Dict[str, int]:
    ordered = sorted(papers.values(), key=lambda paper: (paper.year, paper.title.lower()), reverse=True)
    rank_map = {}
    for rank, paper in enumerate(ordered, start=1):
        rank_map[paper.pmcid] = rank
    return rank_map


def find_diff_path(current_pmcid: str, prior_pmcid: str) -> Optional[Path]:
    filename = "%s_%s_extracted_diffs.json" % (current_pmcid, prior_pmcid)
    for root in DIFF_SEARCH_ROOTS:
        if not root.exists():
            continue
        matches = list(root.rglob(filename))
        if matches:
            return matches[0]
    return None


def flatten_available_diffs(papers: Dict[str, GuidelinePaper]) -> List[Dict[str, Any]]:
    pmcids = sorted(papers.keys(), key=lambda value: int(value.replace("PMC", "")))
    flattened: List[Dict[str, Any]] = []

    # for each current and prior pmcid, go to the loop to find the difference file.
    for current_pmcid in pmcids:
        for prior_pmcid in pmcids:
            if current_pmcid == prior_pmcid:
                continue
            diff_path = find_diff_path(current_pmcid, prior_pmcid)
            
            if diff_path is None:
                continue
            with diff_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, list):
                continue
            for diff_index, diff in enumerate(payload):
                if not isinstance(diff, dict):
                    continue
                flattened.append(
                    {
                        "diff_path": str(diff_path),
                        "pair_id": "%s__%s" % (current_pmcid, prior_pmcid),
                        "difference_idx": diff_index,
                        "current_pmcid": current_pmcid,
                        "prior_pmcid": prior_pmcid,
                        "diff": diff,
                    }
                )
    return flattened


def build_azure_llm(model_name: str) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=model_name,
        api_key=config.AZURE_OPENAI_API_KEY,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_version=config.AZURE_OPENAI_API_VERSION,
        temperature=0 if model_name not in ["azure-gpt-5"] else 1,
        max_retries=5,
        timeout=60.0,
    )


def build_agreement_question_prompt(
    topic: str,
    clinical_focus: str,
    prior_recommendation: str,
    prior_raw_text: str,
    current_recommendation: str,
    current_raw_text: str,
    organization: str,
    current_title: str,
    prior_title: str,
    current_year: int,
    prior_year: int,
) -> Tuple[str, str]:
    system = """You are generating a two-dimensional guideline-comparison question package.

Return exactly four fields:
- question_with_prior_statement
- question_with_current_statement
- explanation_for_prior_statement
- explanation_for_current_statement

--------------------------------
Step 1 — Identify the Core Difference (MANDATORY)
--------------------------------
You MUST first identify the SINGLE most important actionable difference between the prior and current recommendations.

Definition of actionable difference:
- A difference in clinical decision or recommendation strength
- Examples:
  - recommended vs not recommended
  - routine vs selective
  - first-line vs alternative
  - should vs may vs should not

DO NOT select:
- background information (e.g., controversy, statistics, supporting evidence)
- descriptive summaries that do not affect decision-making

--------------------------------
Step 2 — Construct Paired Statements
--------------------------------
You MUST construct TWO statements (prior and current) such that:

1. They reflect the SAME clinical scenario and decision point
2. They differ ONLY in recommendation/decision (not topic or scope)
3. They are mutually incompatible under the same interpretation
4. A model confusing the two guideline versions would likely answer at least one incorrectly
5. `question_with_prior_statement` should be correct only after the years that the prior guidelines are activated; `question_with_current_statement` should be correct only after the years that the current guidelines are activated

CRITICAL CONSTRAINT:
- The two statements MUST NOT both be correct under the same guideline version
- If they can both be agreed with, you MUST revise them

Each statement MUST:
- Express a clear clinical stance (e.g., "should", "should not", "is recommended", "is not routinely necessary")
- Avoid vague or descriptive-only wording (e.g., "is controversial", "studies suggest")
- Be falsifiable (clearly true or false under a specific guideline version)

Preferred contrast patterns:
- routine vs selective
- recommended vs not recommended
- should vs should not
- first-line vs alternative

--------------------------------
Step 3 — Generate Questions
--------------------------------
Generate two questions using EXACTLY this template:

"According to the most recent guideline on xxx published by the xxx on or before {year}, do you agree that xxx?"

Requirements:
- Each must be a full question
- Each must contain the literal token {year}
- The first xxx = topic
- The second xxx = organization
- The third xxx = the statement

- question_with_prior_statement:
  MUST use the PRIOR recommendation

- question_with_current_statement:
  MUST use the CURRENT recommendation

- Do NOT mention "prior" or "current" in the question text
- Do NOT invent content not supported by the recommendations
- Your question MUST accurately capture the key difference.

--------------------------------
Step 4 — Generate Explanations
--------------------------------
For each explanation (2–4 sentences):

You MUST:
- Explicitly mention BOTH guideline years
- Clearly explain the key decision-level difference
- Explicitly state whether the statement in the question is correct or not for that year
- Highlight how the recommendation evolved
- Carefully consider the raw texts from the current and the prior guidelines, to avoid misuderstanding!

--------------------------------
Step 5 — Final Quality Check (STRICT)
--------------------------------
Before output, verify:

1. The two statements create a REAL contrast in clinical decision
2. They are NOT both acceptable under the same guideline version
3. The difference is NOT about background/context, but about recommendation
4. The questions are answerable with YES/NO and require distinguishing guideline versions

If ANY condition fails, revise the statements.
"""
    user = """Topic:
{topic}

Clinical focus:
{clinical_focus}

Current guideline title:
{current_title}

Prior guideline title:
{prior_title}

Prior recommendation (summary):
{prior_recommendation}

Prior recommendation (raw text):
{prior_raw_text}

Current recommendation (summary):
{current_recommendation}

Current recommendation (raw text):
{current_raw_text}

Guideline organization:
{organization}

Current guideline year:
{current_year}

Prior guideline year:
{prior_year}
""".format(
        topic=topic,
        clinical_focus=clinical_focus,
        current_title=current_title,
        prior_title=prior_title,
        prior_recommendation=prior_recommendation,
        prior_raw_text=prior_raw_text,
        current_recommendation=current_recommendation,
        current_raw_text=current_raw_text,
        organization=organization,
        current_year=current_year,
        prior_year=prior_year,
    )
    return system, user


def build_reference_fallback(topic: str, clinical_focus: str) -> str:
    parts = []
    for value in [topic, clinical_focus]:
        text = normalize_reference_text(value)
        if text and text.lower() not in " ".join(parts).lower():
            parts.append(text)
    return " ".join(parts).strip() or "relevant topic"


def build_organization_fallback(organization: str) -> str:
    return normalize_reference_text(organization) or "relevant organization"


def build_statement_fallback(recommendation: str, raw_text: str) -> str:
    return normalize_statement_text(recommendation) or normalize_statement_text(raw_text) or "the recommendation applies"


def build_question_template(topic_reference: str, organization_reference: str, statement_text: str) -> str:
    question_template = (
        "According to the most recent guideline on %s published by the %s on or before {year}, do you agree with the statement that %s?"
        % (
            normalize_reference_text(topic_reference) or "relevant topic",
            normalize_reference_text(organization_reference) or "relevant organization",
            normalize_statement_text(statement_text) or "the recommendation applies",
        )
    )
    return normalize_text(question_template)


def ensure_question_template(question_template: str, fallback_template: str) -> str:
    template = normalize_text(question_template)
    if "{year}" not in template:
        template = fallback_template
    if not template.endswith("?"):
        template += "?"
    return normalize_text(template)


def build_question_from_template(question_template: str, year: int) -> str:
    return ensure_question_template(question_template, question_template).replace("{year}", str(year))


def build_year_states(
    target_years: Sequence[int],
    prior_year: int,
    current_year: int,
    prior_pmid: Optional[int],
    current_pmid: Optional[int],
    prior_recommendation: str,
    current_recommendation: str,
) -> List[Dict[str, Any]]:
    same_state = recommendations_match(prior_recommendation, current_recommendation)
    assignments = []
    for year in target_years:
        if year < prior_year:
            assignment = {
                "year": year,
                "label": "C",
                "recommendation": None,
                "recommendation_source": None,
                "latest_guideline_year": None,
                "latest_guideline_pmid": None,
            }
        elif year < current_year:
            assignment = {
                "year": year,
                "label": "B",
                "recommendation": prior_recommendation,
                "recommendation_source": "prior",
                "latest_guideline_year": prior_year,
                "latest_guideline_pmid": prior_pmid,
            }
        else:
            assignment = {
                "year": year,
                "label": "A",
                "recommendation": current_recommendation if not same_state else prior_recommendation,
                "recommendation_source": "current" if not same_state else "prior",
                "latest_guideline_year": current_year,
                "latest_guideline_pmid": current_pmid,
            }
        assignments.append(assignment)
    return assignments


def build_yes_no_answer(assignment: Dict[str, Any], statement_recommendation: str) -> Optional[str]:
    if assignment["label"] == "C" or assignment["recommendation"] is None:
        return "C"
    return "A" if recommendations_match(assignment["recommendation"], statement_recommendation) else "B"


def build_answer_payload(
    assignment: Dict[str, Any],
    correct_label: Optional[str],
    explanation: str,
    statement_source: str,
    statement_guideline_year: int,
) -> Optional[Dict[str, Any]]:
    if correct_label is None:
        return None
    latest_year = assignment.get("latest_guideline_year")
    year = assignment.get("year")
    statement_origin = "%s guideline (%s)" % (statement_source, statement_guideline_year)
    if correct_label == "C":
        latest_sentence = (
            "For year %s, there is no guideline available on or before that year, so Choice_C is correct."
            % year
        )
    else:
        agreement_word = "agrees" if correct_label == "A" else "does not agree"
        latest_sentence = (
            "For year %s, the latest available guideline is from %s, and it %s with the statement drawn from the %s, so Choice_%s is correct."
            % (year, latest_year, agreement_word, statement_origin, correct_label)
        )
    explanation_text = normalize_text(explanation)
    if explanation_text:
        explanation_text = latest_sentence + " " + explanation_text
    else:
        explanation_text = latest_sentence
    return {
        "Choice_A": "Yes",
        "Choice_B": "No",
        "Choice_C": "I do not know",
        "Correct": correct_label,
        "Explanation": explanation_text,
    }


def parse_agreement_question_response(raw_text: str) -> Optional[AgreementQuestionResponse]:
    if not raw_text:
        return None
    text = raw_text.strip()
    try:
        payload = json.loads(text)
        return AgreementQuestionResponse.model_validate(payload)
    except Exception:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        try:
            payload = json.loads(fenced_match.group(1))
            return AgreementQuestionResponse.model_validate(payload)
        except Exception:
            pass

    json_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(1))
            return AgreementQuestionResponse.model_validate(payload)
        except Exception:
            pass
    return None


def generate_question_packages(
    diff_records: Sequence[Dict[str, Any]],
    papers: Dict[str, GuidelinePaper],
    generator_model: str,
) -> List[Dict[str, Any]]:
    llm = build_azure_llm(generator_model)
    packages: List[Dict[str, Any]] = []

    try:
        for record in tqdm(diff_records, desc="Generating diff question packages", unit="diff", leave=False):
            current_paper = papers[record["current_pmcid"]]
            prior_paper = papers[record["prior_pmcid"]]
            diff = record["diff"]
            topic = current_paper.topic or prior_paper.topic
            organization = current_paper.organization or prior_paper.organization
            clinical_focus = normalize_text(diff.get("clinical_focus"))
            prior_recommendation = normalize_recommendation(diff.get("prior_recommendation"))
            current_recommendation = normalize_recommendation(diff.get("current_recommendation"))
            prior_raw_text = normalize_text(diff.get("prior_recommendation_raw_text"))
            current_raw_text = normalize_text(diff.get("current_recommendation_raw_text"))

            system_prompt, user_prompt = build_agreement_question_prompt(
                topic=topic,
                clinical_focus=clinical_focus,
                prior_recommendation=prior_recommendation,
                prior_raw_text=prior_raw_text,
                current_recommendation=current_recommendation,
                current_raw_text=current_raw_text,
                organization=organization,
                current_title=current_paper.title,
                prior_title=prior_paper.title,
                current_year=current_paper.year,
                prior_year=prior_paper.year,
            )

            fallback_topic_reference = build_reference_fallback(topic, clinical_focus)
            fallback_organization_reference = build_organization_fallback(organization)
            fallback_prior_statement = build_statement_fallback(prior_recommendation, prior_raw_text)
            fallback_current_statement = build_statement_fallback(current_recommendation, current_raw_text)
            prior_fallback_template = build_question_template(
                fallback_topic_reference,
                fallback_organization_reference,
                fallback_prior_statement,
            )
            current_fallback_template = build_question_template(
                fallback_topic_reference,
                fallback_organization_reference,
                fallback_current_statement,
            )

            try:
                response = llm.invoke(
                    [
                        SystemMessage(content=system_prompt + "\nReturn valid JSON only with exactly the four required keys."),
                        HumanMessage(content=user_prompt),
                    ]
                )
                parsed = parse_agreement_question_response(
                    response.content if hasattr(response, "content") else str(response)
                )
                if parsed is None:
                    raise ValueError("Could not parse generator response as AgreementQuestionResponse JSON")
                prior_question_template = ensure_question_template(
                    parsed.question_with_prior_statement,
                    prior_fallback_template,
                )
                current_question_template = ensure_question_template(
                    parsed.question_with_current_statement,
                    current_fallback_template,
                )
                prior_explanation = normalize_text(parsed.explanation_for_prior_statement)
                current_explanation = normalize_text(parsed.explanation_for_current_statement)
            except Exception as exc:
                prior_question_template = prior_fallback_template
                current_question_template = current_fallback_template
                prior_explanation = "Fallback template used because question generation failed: %s" % normalize_text(exc)
                current_explanation = prior_explanation

            packages.append(
                {
                    **record,
                    "topic": topic,
                    "organization": organization,
                    "clinical_focus": clinical_focus,
                    "prior_recommendation": prior_recommendation,
                    "current_recommendation": current_recommendation,
                    "prior_raw_text": prior_raw_text,
                    "current_raw_text": current_raw_text,
                    "question_with_prior_statement": prior_question_template,
                    "question_with_current_statement": current_question_template,
                    "explanation_for_prior_statement": prior_explanation,
                    "explanation_for_current_statement": current_explanation,
                }
            )
    finally:
        close_azure_llm(llm)
    return packages


def build_matrix_questions(
    question_packages: Sequence[Dict[str, Any]],
    papers: Dict[str, GuidelinePaper],
    experiment: Dict[str, Any],
    target_years: Sequence[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    idx = 0

    for package in question_packages:
        current_paper = papers[package["current_pmcid"]]
        prior_paper = papers[package["prior_pmcid"]]
        year_assignments = build_year_states(
            target_years,
            prior_year=prior_paper.year,
            current_year=current_paper.year,
            prior_pmid=prior_paper.pmid,
            current_pmid=current_paper.pmid,
            prior_recommendation=package["prior_recommendation"],
            current_recommendation=package["current_recommendation"],
        )

        statement_entries = [
            {
                "statement_source": "prior",
                "target_pmcid": prior_paper.pmcid,
                "target_title": prior_paper.title,
                "target_year": prior_paper.year,
                "target_role": prior_paper.role,
                "target_status": prior_paper.status,
                "target_pmid": prior_paper.pmid,
                "statement_guideline_year": prior_paper.year,
                "statement_guideline_pmid": prior_paper.pmid,
                "statement_recommendation": package["prior_recommendation"],
                "question_template": package["question_with_prior_statement"], # question_with_prior_statement prior_question_template
                "question_explanation": package["explanation_for_prior_statement"], # explanation_for_prior_statement prior_explanation
            },
            {
                "statement_source": "current",
                "target_pmcid": current_paper.pmcid,
                "target_title": current_paper.title,
                "target_year": current_paper.year,
                "target_role": current_paper.role,
                "target_status": current_paper.status,
                "target_pmid": current_paper.pmid,
                "statement_guideline_year": current_paper.year,
                "statement_guideline_pmid": current_paper.pmid,
                "statement_recommendation": package["current_recommendation"],
                "question_template": package["question_with_current_statement"], # question_with_current_statement current_question_template
                "question_explanation": package["explanation_for_current_statement"], # explanation_for_current_statement current_explanation
            },
        ]
        
        for statement_entry in statement_entries:
            for assignment in year_assignments:
                correct_label = build_yes_no_answer(assignment, statement_entry["statement_recommendation"])
                question_text = (
                    build_question_from_template(statement_entry["question_template"], assignment["year"])
                )
                rows.append(
                    {
                        "idx": idx,
                        "experiment_index": experiment["_experiment_index"],
                        "current_pmcid": package["current_pmcid"],
                        "prior_pmcid": package["prior_pmcid"],
                        "pair_id": package["pair_id"],
                        "difference_idx": package["difference_idx"],
                        "topic": package["topic"],
                        "organization": package["organization"],
                        "clinical_focus": package["clinical_focus"],
                        "target_pmcid": statement_entry["target_pmcid"],
                        "target_title": statement_entry["target_title"],
                        "target_year": statement_entry["target_year"],
                        "target_role": statement_entry["target_role"],
                        "target_status": statement_entry["target_status"],
                        "target_pmid": statement_entry["target_pmid"],
                        "statement_source": statement_entry["statement_source"],
                        "statement_guideline_year": statement_entry["statement_guideline_year"],
                        "statement_guideline_pmid": statement_entry["statement_guideline_pmid"],
                        "statement_text": statement_entry["statement_recommendation"],
                        "question_target_year": assignment["year"],
                        "latest_guideline_year": assignment["latest_guideline_year"],
                        "latest_guideline_pmid": assignment["latest_guideline_pmid"],
                        "recommendation_source": assignment["recommendation_source"],
                        "available": True,
                        "Question": question_text,
                        "Answer": build_answer_payload(
                            assignment,
                            correct_label,
                            statement_entry["question_explanation"],
                            statement_entry["statement_source"],
                            statement_entry["statement_guideline_year"],
                        ),
                    }
                )
                idx += 1

    return rows


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def parse_model_spec(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        return "azure", spec
    backend, model_name = spec.split(":", 1)
    backend = backend.strip().lower()
    model_name = model_name.strip()
    if backend not in {"azure", "vllm"}:
        raise ValueError("Unsupported backend in model spec: %s" % spec)
    if not model_name:
        raise ValueError("Empty model name in spec: %s" % spec)
    return backend, model_name


def _run_aclose_safely(aclose_callable: Any) -> None:
    try:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(aclose_callable())
        finally:
            loop.close()
    except Exception:
        pass


def close_azure_llm(llm: Any) -> None:
    seen_ids = set()
    stack = [llm]
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen_ids:
            continue
        seen_ids.add(id(obj))

        for attr_name in (
            "client",
            "async_client",
            "root_client",
            "root_async_client",
            "http_client",
            "http_async_client",
            "_client",
            "_async_client",
            "_root_client",
            "_root_async_client",
        ):
            child = getattr(obj, attr_name, None)
            if child is not None:
                stack.append(child)

        close_fn = getattr(obj, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

        aclose_fn = getattr(obj, "aclose", None)
        if callable(aclose_fn):
            _run_aclose_safely(aclose_fn)


def build_eval_prompt(question: str) -> str:
    return """
Question:
%s

A) Yes
B) No
C) I do not know

# Do not fabricate evidence or unsupported facts.
# Please answer with the following format:
##Answer: [Your Choice Here (A/B/C)]
##Explanation: [Your explanation here]
""" % question


def _sync_azure_invoke(azure_llm: AzureChatOpenAI, prompt: str) -> str:
    try:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT_EVAL),
            HumanMessage(content=prompt),
        ]
        response = azure_llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)
    except BadRequestError as exc:
        print("[WARN] Azure prompt rejected: %s" % exc)
        return FALLBACK_ANSWER
    except Exception as exc:
        print("[ERROR] Azure prompt failed: %s" % exc)
        return FALLBACK_ANSWER


def parallel_azure_query(model_name: str, prompts: Sequence[str], max_concurrency: int) -> List[str]:
    azure_llm = build_azure_llm(model_name)
    outputs = [FALLBACK_ANSWER] * len(prompts)
    try:
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            future_map = {}
            for index, prompt in enumerate(prompts):
                future = executor.submit(_sync_azure_invoke, azure_llm, prompt)
                future_map[future] = index
            for future in tqdm(as_completed(future_map), total=len(future_map), desc="Azure evaluation", leave=False):
                outputs[future_map[future]] = future.result()
    finally:
        close_azure_llm(azure_llm)
    return outputs


def query_model(backend: str, model_name: str, prompts: Sequence[str], max_concurrency: int) -> List[str]:
    if backend == "azure":
        return parallel_azure_query(model_name, prompts, max_concurrency)
    from utils.model_utils import get_response_with_vllm
    return get_response_with_vllm(model_name, list(prompts), system_prompt=SYSTEM_PROMPT_EVAL)


def extract_choice_label(text: str) -> str:
    if not text:
        return "INVALID"
    cleaned = text.strip()
    match = re.search(r"##Answer:\s*(A|B)\b", cleaned, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    first_line = cleaned.splitlines()[0].strip().upper() if cleaned.splitlines() else ""
    if first_line in {"A", "B", "C"}:
        return first_line
    tokens = re.findall(r"\b(A|B)\b", cleaned.upper())
    if len(tokens) == 1:
        return tokens[0]
    return "INVALID"


def evaluate_questions(
    question_rows: Sequence[Dict[str, Any]],
    model_spec: str,
    max_concurrency: int,
) -> List[Dict[str, Any]]:
    backend, model_name = parse_model_spec(model_spec)
    evaluable_rows = [row for row in question_rows if row.get("available") and row.get("Question")]
    prompts = [build_eval_prompt(row["Question"]) for row in evaluable_rows]
    responses = query_model(backend, model_name, prompts, max_concurrency)

    results = []
    for row, response in zip(evaluable_rows, responses):
        prediction = extract_choice_label(response)
        gold = row["Answer"]["Correct"]
        results.append(
            {
                **row,
                "model_spec": model_spec,
                "model_backend": backend,
                "model_name": model_name,
                "prompt": build_eval_prompt(row["Question"]),
                "model_answer": response,
                "prediction": prediction,
                "correct": prediction == gold,
            }
        )
    return results


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = float(successes) / total
    denominator = 1 + (z ** 2) / total
    center = (phat + (z ** 2) / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt((phat * (1 - phat) / total) + (z ** 2) / (4 * total ** 2))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def summarize_results(
    evaluation_rows: Sequence[Dict[str, Any]],
    papers: Dict[str, GuidelinePaper],
    target_years: Sequence[int],
    target_rank_map: Dict[str, int],
    experiment_index: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in evaluation_rows:
        grouped[(row["statement_source"], row["question_target_year"])].append(row)

    summary: List[Dict[str, Any]] = []
    for statement_source in ["prior", "current"]:
        for year in target_years:
            rows = grouped.get((statement_source, year), [])
            total = len(rows)
            correct = sum(int(bool(row["correct"])) for row in rows)
            invalid = sum(int(row["prediction"] == "INVALID") for row in rows)
            accuracy = (float(correct) / total) if total else None
            ci_low, ci_high = wilson_interval(correct, total) if total else (None, None)
            summary.append(
                {
                    "experiment_index": experiment_index,
                    "statement_source": statement_source,
                    "question_target_year": year,
                    "total": total,
                    "correct": correct,
                    "invalid": invalid,
                    "accuracy": accuracy,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return summary


def write_summary_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            normalized = dict(row)
            for key in ["accuracy", "ci_low", "ci_high", "mean_accuracy", "std_accuracy"]:
                if key in normalized:
                    normalized[key] = "" if normalized[key] is None else "%.6f" % normalized[key]
            writer.writerow(normalized)


def wrap_label(text: str, width: int = 34) -> str:
    lines = textwrap.wrap(normalize_text(text), width=width)
    return "\n".join(lines[:3]) if lines else text


HEATMAP_CURRENT_COLOR = "#2F4858"
HEATMAP_PRIOR_COLOR = "#A44A3F"
HEATMAP_CURRENT_MARKER_COLOR = "#8FB7C9"
HEATMAP_GRID_COLOR = "#d9d9d9"
HEATMAP_TEXT_COLOR = "#1f2937"
HEATMAP_UNKNOWN_COLOR = "#d7dde5"
HEATMAP_EMPTY_COLOR = "#eef2f7"


def configure_heatmap_style() -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "Andale Mono",
            "axes.labelsize": 20,
            "axes.titlesize": 18,
            "xtick.labelsize": 13,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
        }
    )


def style_heatmap_axis(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=HEATMAP_TEXT_COLOR)


def save_figure_variants(fig, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())


def create_heatmap(
    summary_rows: Sequence[Dict[str, Any]],
    papers: Dict[str, GuidelinePaper],
    experiment: Dict[str, Any],
    model_spec: str,
    output_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch

    configure_heatmap_style()
    ordered_sources = ["prior", "current"]
    ordered_years = sorted({row["question_target_year"] for row in summary_rows})
    row_index = {source: idx for idx, source in enumerate(ordered_sources)}
    col_index = {year: idx for idx, year in enumerate(ordered_years)}

    matrix = np.full((len(ordered_sources), len(ordered_years)), np.nan, dtype=float)
    totals = np.zeros((len(ordered_sources), len(ordered_years)), dtype=int)
    for row in summary_rows:
        i = row_index[row["statement_source"]]
        j = col_index[row["question_target_year"]]
        if row["accuracy"] is not None:
            matrix[i, j] = row["accuracy"]
        totals[i, j] = row["total"]

    cmap = LinearSegmentedColormap.from_list(
        "inspect_accuracy",
        [HEATMAP_EMPTY_COLOR, "#d8e5ec", "#8ba8b8", HEATMAP_CURRENT_COLOR],
        N=256,
    ).copy()
    cmap.set_bad(color=HEATMAP_EMPTY_COLOR)

    fig_height = 4.8
    fig, ax = plt.subplots(figsize=(1.0 * len(ordered_years) + 5.2, fig_height))
    fig.patch.set_facecolor("white")

    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    style_heatmap_axis(ax)

    ax.set_xticks(np.arange(len(ordered_years)))
    ax.set_yticks(np.arange(len(ordered_sources)))
    ax.set_xticklabels([str(year) for year in ordered_years], fontsize=13, color=HEATMAP_TEXT_COLOR)
    ax.set_yticklabels(
        ["Prior Statements", "Current Statements"],
        fontsize=15,
        color=HEATMAP_TEXT_COLOR,
    )
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True, length=0)
    ax.set_xticks(np.arange(-0.5, len(ordered_years), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ordered_sources), 1), minor=True)
    ax.grid(which="minor", color=HEATMAP_GRID_COLOR, linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            total = totals[i, j]
            value = matrix[i, j]
            if total <= 0 or np.isnan(value):
                ax.text(j, i, "NA", ha="center", va="center", fontsize=9.5, color="#6b7280")
                continue
            text_color = "#ffffff" if value >= 0.62 else HEATMAP_TEXT_COLOR
            ax.text(
                j,
                i,
                "%d%%\n(n=%d)" % (round(value * 100), total),
                ha="center",
                va="center",
                fontsize=9.0,
                color=text_color,
                weight="semibold",
            )

    ax.set_xlabel("Evaluation Time (Year)", fontsize=20, color=HEATMAP_TEXT_COLOR, labelpad=12)
    ax.set_ylabel("Question Statement Type", fontsize=20, color=HEATMAP_TEXT_COLOR, labelpad=12)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=12, colors=HEATMAP_TEXT_COLOR)
    colorbar.set_label("Accuracy", fontsize=16, color=HEATMAP_TEXT_COLOR)

    fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.98])
    save_figure_variants(fig, output_path, dpi)
    plt.close(fig)


def create_prediction_heatmaps(
    evaluation_rows: Sequence[Dict[str, Any]],
    papers: Dict[str, GuidelinePaper],
    experiment: Dict[str, Any],
    model_spec: str,
    output_dir: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    configure_heatmap_style()
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in evaluation_rows:
        grouped[(row["pair_id"], row["difference_idx"])].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    cmap = ListedColormap(["#ffffff", HEATMAP_CURRENT_COLOR, HEATMAP_UNKNOWN_COLOR])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    for (pair_id, difference_idx), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        ordered_years = sorted({int(row["question_target_year"]) for row in rows})
        row_index = {"prior": 0, "current": 1}
        col_index = {year: idx for idx, year in enumerate(ordered_years)}
        matrix = np.full((2, len(ordered_years)), np.nan, dtype=float)
        gold_matrix = [[None for _ in ordered_years] for _ in range(2)]

        first_row = rows[0]
        current_paper = papers[first_row["current_pmcid"]]
        prior_paper = papers[first_row["prior_pmcid"]]

        for row in rows:
            i = row_index[row["statement_source"]]
            j = col_index[int(row["question_target_year"])]
            prediction = row.get("prediction")
            if prediction == "A":
                matrix[i, j] = 1.0
            elif prediction == "B":
                matrix[i, j] = 0.0
            else:
                matrix[i, j] = 2.0
            gold_matrix[i][j] = row.get("Answer", {}).get("Correct")

        fig, ax = plt.subplots(figsize=(max(8.5, 0.72 * len(ordered_years) + 4.8), 4.6))
        fig.patch.set_facecolor("white")

        masked = np.ma.masked_invalid(matrix)
        ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        style_heatmap_axis(ax)

        ax.set_xticks(np.arange(len(ordered_years)))
        ax.set_yticks(np.arange(2))
        ax.set_xticklabels([str(year) for year in ordered_years], fontsize=13, color=HEATMAP_TEXT_COLOR)
        ax.set_yticklabels(["Prior Statement", "Current Statement"], fontsize=15, color=HEATMAP_TEXT_COLOR)
        ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True, length=0)
        ax.set_xticks(np.arange(-0.5, len(ordered_years), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
        ax.grid(which="minor", color=HEATMAP_GRID_COLOR, linestyle="-", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)

        for year_value, label, color in [
            (prior_paper.year, "Prior Guideline", HEATMAP_PRIOR_COLOR),
            (current_paper.year, "Current Guideline", HEATMAP_CURRENT_MARKER_COLOR),
        ]:
            if year_value in col_index:
                x = col_index[year_value]
                ax.axvline(x=x, color=color, linewidth=2.0, linestyle="--", alpha=0.9)
                ax.text(x, -0.72, label, ha="center", va="bottom", fontsize=10.5, color=color, weight="bold")

        for i in range(2):
            for j in range(len(ordered_years)):
                value = matrix[i, j]
                if np.isnan(value):
                    ax.text(j, i, "NA", ha="center", va="center", fontsize=9.5, color="#6b7280")
                    continue
                prediction_text = "Yes" if value == 1.0 else "No" if value == 0.0 else "IDK"
                gold = gold_matrix[i][j]
                gold_text = "Yes" if gold == "A" else "No" if gold == "B" else "IDK" if gold == "C" else "?"
                text_color = "#ffffff" if value == 1.0 else HEATMAP_TEXT_COLOR
                ax.text(
                    j,
                    i,
                    "%s\nGT:%s" % (prediction_text, gold_text),
                    ha="center",
                    va="center",
                    fontsize=9.0,
                    color=text_color,
                    weight="semibold",
                )

        ax.set_xlabel("Evaluation Time (Year)", fontsize=20, color=HEATMAP_TEXT_COLOR, labelpad=12)
        ax.set_ylabel("Question Statement Type", fontsize=20, color=HEATMAP_TEXT_COLOR, labelpad=12)

        fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.98])
        file_name = "difference_%02d__%s.png" % (difference_idx, slugify(pair_id))
        save_figure_variants(fig, output_dir / file_name, dpi)
        plt.close(fig)


def aggregate_summary_rows(all_summary_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in all_summary_rows:
        if row.get("accuracy") is None:
            continue
        grouped[(row["statement_source"], row["question_target_year"])].append(row)

    aggregated = []
    for (statement_source, question_year), rows in sorted(grouped.items()):
        values = [float(row["accuracy"]) for row in rows]
        mean_accuracy = sum(values) / len(values)
        variance = sum((value - mean_accuracy) ** 2 for value in values) / len(values)
        std_accuracy = math.sqrt(variance)
        aggregated.append(
            {
                "statement_source": statement_source,
                "question_target_year": question_year,
                "mean_accuracy": mean_accuracy,
                "std_accuracy": std_accuracy,
                "n_trajectories": len(set(row["experiment_index"] for row in rows)),
                "n_cells": len(rows),
            }
        )
    return aggregated


def create_aggregate_heatmap(
    aggregate_rows: Sequence[Dict[str, Any]],
    model_spec: str,
    output_path: Path,
    dpi: int,
    title_suffix: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch

    configure_heatmap_style()
    sources = ["prior", "current"]
    years = sorted({row["question_target_year"] for row in aggregate_rows})
    row_index = {source: idx for idx, source in enumerate(sources)}
    col_index = {year: idx for idx, year in enumerate(years)}

    matrix = np.full((len(sources), len(years)), np.nan, dtype=float)
    counts = np.zeros((len(sources), len(years)), dtype=int)
    for row in aggregate_rows:
        i = row_index[row["statement_source"]]
        j = col_index[row["question_target_year"]]
        matrix[i, j] = row["mean_accuracy"]
        counts[i, j] = row["n_trajectories"]

    cmap = LinearSegmentedColormap.from_list(
        "aggregate_accuracy",
        [HEATMAP_EMPTY_COLOR, "#d8e5ec", "#8ba8b8", HEATMAP_CURRENT_COLOR],
        N=256,
    ).copy()
    cmap.set_bad(color=HEATMAP_EMPTY_COLOR)

    fig_height = 4.8
    fig, ax = plt.subplots(figsize=(1.0 * len(years) + 4.8, fig_height))
    fig.patch.set_facecolor("white")

    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    style_heatmap_axis(ax)

    ax.set_xticks(np.arange(len(years)))
    ax.set_yticks(np.arange(len(sources)))
    ax.set_xticklabels([str(year) for year in years], fontsize=13, color=HEATMAP_TEXT_COLOR)
    ax.set_yticklabels(["Prior Statements", "Current Statements"], fontsize=15, color=HEATMAP_TEXT_COLOR)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True, length=0)
    ax.set_xticks(np.arange(-0.5, len(years), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(sources), 1), minor=True)
    ax.grid(which="minor", color=HEATMAP_GRID_COLOR, linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            count = counts[i, j]
            if np.isnan(value) or count == 0:
                ax.text(j, i, "NA", ha="center", va="center", fontsize=9.5, color="#6b7280")
                continue
            text_color = "#ffffff" if value >= 0.62 else HEATMAP_TEXT_COLOR
            ax.text(
                j,
                i,
                "%d%%\n(n=%d)" % (round(value * 100), count),
                ha="center",
                va="center",
                fontsize=9.0,
                color=text_color,
                weight="semibold",
            )

    ax.set_xlabel("Evaluation Time (Year)", fontsize=20, color=HEATMAP_TEXT_COLOR, labelpad=12)
    ax.set_ylabel("Question Statement Type", fontsize=20, color=HEATMAP_TEXT_COLOR, labelpad=12)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=12, colors=HEATMAP_TEXT_COLOR)
    colorbar.set_label("Mean Accuracy", fontsize=16, color=HEATMAP_TEXT_COLOR)

    fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.98])
    save_figure_variants(fig, output_path, dpi)
    plt.close(fig)


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    experiment: Dict[str, Any],
    papers: Dict[str, GuidelinePaper],
    skipped_nodes: Sequence[Dict[str, Any]],
    diff_records: Sequence[Dict[str, Any]],
    target_years: Sequence[int],
    question_path: Path,
) -> None:
    payload = {
        "experiment_index": experiment["_experiment_index"],
        "input_json": args.input_json,
        "current_pmcid": experiment.get("pmc_id"),
        "current_title": experiment.get("title"),
        "topic": experiment.get("topic"),
        "node_count": experiment.get("node_count"),
        "found_paper_count": experiment.get("found_paper_count"),
        "missing_paper_count": experiment.get("missing_paper_count"),
        "generator_model": args.generator_model,
        "eval_models": args.eval_model,
        "target_years": list(target_years),
        "resolved_target_papers": [paper.to_dict() for paper in sorted(papers.values(), key=lambda p: (p.year, p.title))],
        "skipped_nodes": list(skipped_nodes),
        "diff_pairs": [
            {
                "pair_id": row["pair_id"],
                "current_pmcid": row["current_pmcid"],
                "prior_pmcid": row["prior_pmcid"],
                "difference_idx": row["difference_idx"],
                "diff_path": row["diff_path"],
            }
            for row in diff_records
        ],
        "question_jsonl": str(question_path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def prepare_experiment_bundle(
    experiment: Dict[str, Any],
    args: argparse.Namespace,
    output_root: Path,
) -> Dict[str, Any]:
    papers, skipped_nodes = build_guideline_papers(experiment)
    if not papers:
        return {
            "status": "skipped",
            "reason": "no_resolved_papers",
            "experiment_index": experiment["_experiment_index"],
            "pmc_id": experiment.get("pmc_id"),
        }

    target_years = build_target_years(papers, args.start_year, args.end_year)
    target_rank_map = build_target_rank_map(papers)
    experiment_dir = output_root / ("experiment_%04d_%s" % (experiment["_experiment_index"], experiment["pmc_id"]))
    question_path = experiment_dir / "generated_questions.jsonl"
    package_path = experiment_dir / "question_packages.json"
    metadata_path = experiment_dir / "metadata.json"

    if args.eval_only:
        if not question_path.exists():
            return {
                "status": "skipped",
                "reason": "missing_generated_questions",
                "experiment_index": experiment["_experiment_index"],
                "pmc_id": experiment.get("pmc_id"),
            }
        matrix_questions = read_jsonl(question_path)
    else:
        diff_records = flatten_available_diffs(papers)
        if not diff_records:
            return {
                "status": "skipped",
                "reason": "no_internal_diff_pairs",
                "experiment_index": experiment["_experiment_index"],
                "pmc_id": experiment.get("pmc_id"),
            }
        question_packages = generate_question_packages(diff_records, papers, args.generator_model)
        matrix_questions = build_matrix_questions(question_packages, papers, experiment, target_years)
        write_jsonl(question_path, matrix_questions)
        package_path.parent.mkdir(parents=True, exist_ok=True)
        with package_path.open("w", encoding="utf-8") as handle:
            json.dump(question_packages, handle, indent=2, ensure_ascii=False)
        write_metadata(metadata_path, args, experiment, papers, skipped_nodes, diff_records, target_years, question_path)

    return {
        "status": "ok",
        "experiment_index": experiment["_experiment_index"],
        "pmc_id": experiment.get("pmc_id"),
        "experiment": experiment,
        "papers": papers,
        "target_years": target_years,
        "target_rank_map": target_rank_map,
        "experiment_dir": experiment_dir,
        "matrix_questions": matrix_questions,
        "per_model_summary": {},
    }


def load_existing_evaluation_rows(bundle: Dict[str, Any], model_spec: str) -> List[Dict[str, Any]]:
    raw_path = bundle["experiment_dir"] / slugify(model_spec) / "evaluation_results.json"
    if not raw_path.exists():
        raise FileNotFoundError("Missing evaluation results for %s: %s" % (model_spec, raw_path))
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("evaluation_results.json must contain a list: %s" % raw_path)
    return payload


def write_experiment_model_outputs(
    bundle: Dict[str, Any],
    model_spec: str,
    evaluation_rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    model_dir = bundle["experiment_dir"] / slugify(model_spec)
    summary_rows = summarize_results(
        evaluation_rows,
        bundle["papers"],
        bundle["target_years"],
        bundle["target_rank_map"],
        bundle["experiment_index"],
    )
    raw_path = model_dir / "evaluation_results.json"
    csv_path = model_dir / "heatmap_summary.csv"
    heatmap_dir = model_dir / "difference_heatmaps"

    model_dir.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8") as handle:
        json.dump(list(evaluation_rows), handle, indent=2, ensure_ascii=False)
    write_summary_csv(
        csv_path,
        summary_rows,
        [
            "experiment_index",
            "statement_source",
            "question_target_year",
            "total",
            "correct",
            "invalid",
            "accuracy",
            "ci_low",
            "ci_high",
        ],
    )
    create_prediction_heatmaps(evaluation_rows, bundle["papers"], bundle["experiment"], model_spec, heatmap_dir, args.dpi)
    return summary_rows


def run_single_experiment(
    experiment: Dict[str, Any],
    args: argparse.Namespace,
    output_root: Path,
) -> Dict[str, Any]:
    result = prepare_experiment_bundle(experiment, args, output_root)
    if result.get("status") != "ok":
        return result

    if args.skip_eval:
        return result

    if args.render_only:
        for model_spec in args.eval_model:
            evaluation_rows = load_existing_evaluation_rows(result, model_spec)
            summary_rows = write_experiment_model_outputs(result, model_spec, evaluation_rows, args)
            result["per_model_summary"][model_spec] = summary_rows
        return result

    for model_spec in args.eval_model:
        evaluation_rows = evaluate_questions(result["matrix_questions"], model_spec, args.max_concurrency)
        summary_rows = write_experiment_model_outputs(result, model_spec, evaluation_rows, args)
        result["per_model_summary"][model_spec] = summary_rows

    return result


def run_batch(experiments: Sequence[Dict[str, Any]], args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    batch_label = slugify(Path(args.input_json).stem)
    aggregate_root = output_root / ("aggregate_%s" % batch_label)

    aggregate_by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    batch_status = []
    prepared_bundles: List[Dict[str, Any]] = []

    for experiment in tqdm(experiments, desc="Preparing trajectories", unit="trajectory"):
        try:
            result = prepare_experiment_bundle(experiment, args, output_root)
        except Exception as exc:
            print(exc)
            result = {
                "status": "failed",
                "reason": repr(exc),
                "experiment_index": experiment.get("_experiment_index"),
                "pmc_id": experiment.get("pmc_id"),
            }
        batch_status.append(result)
        if result.get("status") == "ok":
            prepared_bundles.append(result)
    # pdb.set_trace()
    if not args.skip_eval:
        if args.render_only:
            for model_spec in args.eval_model:
                for bundle in prepared_bundles:
                    bundle_results = load_existing_evaluation_rows(bundle, model_spec)
                    summary_rows = write_experiment_model_outputs(bundle, model_spec, bundle_results, args)
                    bundle["per_model_summary"][model_spec] = summary_rows
                    aggregate_by_model[model_spec].extend(summary_rows)
        else:
            for model_spec in args.eval_model:
                combined_questions: List[Dict[str, Any]] = []
                bundle_question_counts: List[int] = []
                for bundle in prepared_bundles:
                    rows = bundle["matrix_questions"]
                    combined_questions.extend(rows)
                    bundle_question_counts.append(len(rows))

                combined_results = evaluate_questions(combined_questions, model_spec, args.max_concurrency)
                offset = 0
                for bundle, question_count in zip(prepared_bundles, bundle_question_counts):
                    evaluable_count = sum(
                        1 for row in bundle["matrix_questions"] if row.get("available") and row.get("Question")
                    )
                    bundle_results = combined_results[offset:offset + evaluable_count]
                    offset += evaluable_count
                    summary_rows = write_experiment_model_outputs(bundle, model_spec, bundle_results, args)
                    bundle["per_model_summary"][model_spec] = summary_rows
                    aggregate_by_model[model_spec].extend(summary_rows)

    serializable_batch_status = []
    for row in batch_status:
        serializable_batch_status.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"experiment", "papers", "target_rank_map", "experiment_dir", "matrix_questions"}
            }
        )

    batch_status_path = aggregate_root / "batch_status.json"
    batch_status_path.parent.mkdir(parents=True, exist_ok=True)
    with batch_status_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable_batch_status, handle, indent=2, ensure_ascii=False)

    if args.skip_eval:
        print("Saved batch status: %s" % batch_status_path)
        return

    title_suffix = "Average over %d trajectories" % len([row for row in serializable_batch_status if row.get("status") == "ok"])
    for model_spec, rows in aggregate_by_model.items():
        aggregate_rows = aggregate_summary_rows(rows)
        model_dir = aggregate_root / slugify(model_spec)
        csv_path = model_dir / "aggregate_heatmap_summary.csv"
        heatmap_path = model_dir / "aggregate_heatmap.png"
        write_summary_csv(
            csv_path,
            aggregate_rows,
            ["statement_source", "question_target_year", "mean_accuracy", "std_accuracy", "n_trajectories", "n_cells"],
        )
        create_aggregate_heatmap(aggregate_rows, model_spec, heatmap_path, args.dpi, title_suffix)
        print("Saved aggregate outputs for %s: %s" % (model_spec, model_dir))

    print("Saved batch status: %s" % batch_status_path)


def main() -> None:
    args = parse_args()
    if not args.eval_model:
        args.eval_model = ["azure:%s" % config.AZURE_DEPLOYMENT]

    experiments = load_experiments(args.input_json)
    if args.run_all:
        run_batch(experiments, args)
        return

    if args.experiment_index is None:
        raise ValueError("Provide --experiment-index for single mode, or use --run-all for batch mode.")

    experiment = experiments[args.experiment_index]
    result = run_single_experiment(experiment, args, Path(args.output_root))
    if result.get("status") != "ok":
        raise ValueError("Experiment %d skipped: %s" % (args.experiment_index, result.get("reason")))

    print(
        "Completed experiment %d (%s). Outputs are under %s"
        % (args.experiment_index, experiment.get("pmc_id"), args.output_root)
    )


if __name__ == "__main__":
    main()
