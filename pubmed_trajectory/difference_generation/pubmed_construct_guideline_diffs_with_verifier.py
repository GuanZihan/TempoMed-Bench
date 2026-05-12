import requests
from typing import List, Dict, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json
import os
import sys
from pydantic import BaseModel, Field

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(root_dir)
sys.path.append(root_dir)
from utils import config
from tqdm import tqdm
import glob
from functools import wraps

from openai import RateLimitError, APITimeoutError, APIError
from langchain_core.exceptions import LangChainException
import time
import random

class GuidelineDifference(BaseModel):
    clinical_focus: str = Field(
        ...,
        description="The specific clinical aspect being discussed (e.g., patient selection, treatment threshold, procedural requirement)."
    )

    current_recommendation: str = Field(
        ...,
        description="The recommendation stated in the CURRENT (newest) guideline."
    )

    current_recommendation_raw_text: str = Field(
        ...,
        description="The raw text of the relevant recommendations in the current guideline paper. Just copy paste from the given XML file for the current guideline"
    )

    prior_recommendation: str = Field(
        ...,
        description="The corresponding recommendation in the PRIOR guideline."
    )

    prior_recommendation_raw_text: str = Field(
        ...,
        description="The raw text of the relevant recommendations in the prior guideline paper. Just copy paste from the given XML file for the prior guideline"
    )

    change_summary: str = Field(
        ...,
        description="A concise explanation of how or why the recommendation changed."
    )

    # reasoning_head_to_head_compairson: str = Field(
    #     ...,
    #     description="A justification for why the prior and current recommendations should be regarded as a head-to-head comparison."
    # )

class ItemList(BaseModel):
    items: List[GuidelineDifference] = Field(
        ...,
        description="A list of clinically meaningful recommendation changes between the current and prior guideline."
    )

class GuidelineQuestion(BaseModel):
    question: str = Field(
        ...,
        description="A clear, specific question targeting the identified guideline change."
    )

    current_answer: str = Field(
        ...,
        description="The answer according to the CURRENT (newest) guideline."
    )

    prior_answer: str = Field(
        ...,
        description="The answer according to the PRIOR guideline."
    )

class QuestionItem(BaseModel):
    item: GuidelineQuestion

class GuidelineState(TypedDict):
    json_path: str
    topic: str
    organization: List[str]
    current_year: int
    current_pmid: int
    prior_guidelines: List[Dict]

    current_pmcid: Optional[str]
    prior_pmcids: List[str]

    current_xml: Optional[str]
    prior_xmls: List[str]

    questions: List[Dict]
    extracted_diffs: List[str]
    __end__: bool = False


def retry_on_ratelimit(
    max_retries: int = 5,
    base_delay: float = 5.0,
    max_delay: float = 70.0,
):
    """
    Retry decorator for LLM calls with exponential backoff + jitter.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)

                except (
                    RateLimitError,
                    APITimeoutError,
                    APIError,
                    LangChainException,
                ) as e:
                    # Only retry on rate-limit / transient errors
                    if attempt == max_retries - 1:
                        raise

                    sleep_time = min(
                        base_delay * (2 ** attempt) + random.uniform(0, 1),
                        max_delay,
                    )
                    print(
                        f"[RateLimit] {fn.__name__} failed "
                        f"(attempt {attempt + 1}/{max_retries}), "
                        f"sleeping {sleep_time:.1f}s"
                    )
                    time.sleep(sleep_time)

        return wrapper
    return decorator

def pmid_to_pmcid(pmid: str):
    url = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
    params = {
        "ids": pmid,
        "format": "json"
    }
    # print(pmid)
    for _ in range(3):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            recs = data.get("records", [])
            if recs and "pmcid" in recs[0]:
                return recs[0]["pmcid"]
            return None
        except Exception:
            time.sleep(1 + random.random())
    return None

def pmcid_to_pmid(pmcid: str) -> int | None:
    url = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
    params = {
        "ids": pmcid,
        "format": "json"
    }
    for _ in range(3):
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            records = data.get("records", [])
            if records and "pmid" in records[0]:
                return records[0]["pmid"]
            return None
        except Exception:
            time.sleep(1 + random.random())
    return None


model = AzureChatOpenAI(
    azure_deployment=config.AZURE_DEPLOYMENT,
    api_key=config.AZURE_OPENAI_API_KEY,
    azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
    api_version=config.AZURE_OPENAI_API_VERSION,
    temperature=1 if "gpt-5" in str(config.AZURE_DEPLOYMENT).lower() else 0,
)

EXTRACT_RECOMMENDATION_RUNS = int(
    os.getenv("EXTRACT_RECOMMENDATION_RUNS", "1")
)


def load_json(state: GuidelineState):
    with open(state["json_path"], "r") as f:
        data = json.load(f)

    if not data:
        return {}

    record = data
    return {
        "topic": record.get("Topic"),
        "organization": record.get("Organization", [""]),
        "current_year": record.get("year_of_current_guidance", 0),
        "current_pmid": record.get("PMID", []),
        "prior_guidelines": record.get("prior_guidelines", []),
    }


def validate_json(state: GuidelineState):
    if (
        not state.get("prior_guidelines")
        or not state.get("topic")
        or not state.get("organization")
        or not state.get("current_pmid")
    ):
        return {"__end__": True}

    return state


def validation_router(state):
    if "__end__" in state:
        return END
    return "resolve_pmids"



def resolve_pmids(state: GuidelineState):
    current_pmcid = pmid_to_pmcid(state["current_pmid"])

    prior_pmcids = []
    for g in state["prior_guidelines"]:
        pmcid = pmid_to_pmcid(g.get("PMID"))
        if pmcid:
            prior_pmcids.append(pmcid)

    return {
        "current_pmcid": current_pmcid,
        "prior_pmcids": prior_pmcids,
    }


def load_pmc_xml_from_db(pmcid: str) -> str:
    """
    Load PMC XML from OA directory structure.
    Handles PMC IDs >= 10 million correctly.
    """
    assert pmcid.startswith("PMC"), f"Invalid PMCID: {pmcid}"

    pmc_num = int(pmcid.replace("PMC", ""))          # e.g., 11567890
    bucket = pmc_num // 1_000_000                    # e.g., 11
    bucket_dir = f"PMC{bucket:03d}xxxxxx"            # PMC011xxxxxx

    dir_list = [
        os.path.join(project_root, "pmc_oa_noncomm_extracted_2026_relaxed"),
        os.path.join(project_root, "pmc_oa_comm_extracted_2026_relaxed"),
        os.path.join(project_root, "pmc_oa_other_extracted_2026_relaxed"),
    ]
    for base_dir in dir_list:
        xml_path = os.path.join(base_dir, bucket_dir, f"{pmcid}.xml")
        if not os.path.exists(xml_path):
            continue
        with open(xml_path, "r", encoding="utf-8") as f:
            return f.read()
    
    return ""

def load_pmc_xml(state: GuidelineState):
    if not state["current_pmcid"] or not state["prior_pmcids"]:
        return {"__end__": True}

    current_xml = load_pmc_xml_from_db(state["current_pmcid"])
    prior_xmls = []
    for pid in state["prior_pmcids"]:
        prior_xml = load_pmc_xml_from_db(pid)
        if prior_xml == "":
            return {"__end__": True}
        prior_xmls.append(prior_xml)

    return {
        "current_xml": current_xml,
        "prior_xmls": prior_xmls,
    }


def validate_pmid(state: GuidelineState):
    if "__end__" in state:
        return END
    return "extract_recommendations"

extract_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a medical guideline analysis expert."),
    ("human",
     """
Compare the CURRENT guideline with the PRIOR guideline.

Topic: {topic}
Organization: {organization}

=============================CURRENT GUIDELINE XML=============================
{current_xml}

=============================PRIOR GUIDELINE XML=============================
{prior_xml}

=============================Instructions=============================
1. Please Extract:
    - The key clinical recommendation in the CURRENT guideline.
    - The corresponding recommendation in the PRIOR guideline.
2. For each extracted change, the topic must be **identical** between the prior and current recommendations (i.e., a strict head-to-head comparison).
    - For example, the prior recommendation and the current recommendation are different treatments based on the **SAME disease**
3. Focus only on recommendations that have changed substantially, for example:
    - The prior guideline recommends therapy A for disease C, whereas the current guideline recommends therapy B for the **SAME disease** C based on new clinical evidence. 
    - Each pair of `current_recommendation` and `prior_recommendation` must represent a direct **head-to-head comparison** and be **meaningfully different**.
4. **MUST exclude minor or incremental updates**, including but not limited to:
    - The prior guideline lacked specific guidance, and the current guideline merely adds clarification or detail.
    - Both guidelines recommend the same intervention, with the current guideline only expanding on context, rationale, or implementation details.
5. If the prior and current guidelines address different clinical topics, return an empty list.
6. Each generated `current_recommendation` and `prior_recommendation` must be self-contained, precise, and clinically interpretable, explicitly stating any necessary background conditions or patient populations. Do not use vague words and try to be more specific and detailed.
7. Your recommendation text in `current_recommendation` and `prior_recommendation` MUST be faithful to the corresponding raw texts. DO NOT fabricate any new clinical terms or conditions.
8. Identify as many differences as possible.
""")
])

@retry_on_ratelimit(max_retries=6)
def extract_recommendations(state: GuidelineState):
    outputs = []
    extract_model = model.with_structured_output(ItemList)
    for idx, prior_xml in enumerate(state["prior_xmls"]):
        diffs = []
        for _ in range(EXTRACT_RECOMMENDATION_RUNS):
            resp = extract_model.invoke(
                extract_prompt.format_messages(
                    topic=state["topic"],
                    organization=state["organization"],
                    current_xml=state["current_xml"],
                    prior_xml=prior_xml,
                )
            )
            diffs.extend(resp.items)
        output = {
            "current_pmcid": state["current_pmcid"],
            "prior_pmcid": state["prior_pmcids"][idx],
            "difference": diffs,
            "organization": state["organization"],
        }
        outputs.append(output)
    return {"extracted_diffs": outputs}


def save_pair_outputs(
    base_dir: str,
    current_pmid: str,
    prior_pmid: str,
    extracted_diffs: list
):
    os.makedirs(os.path.join(base_dir, current_pmid), exist_ok=True)

    d_path = os.path.join(
        base_dir,
        current_pmid,
        f"{current_pmid}_{prior_pmid}_extracted_diffs.json"
    )

    with open(d_path, "w", encoding="utf-8") as f:
        json.dump(extracted_diffs, f, indent=2, ensure_ascii=False)

def persist_outputs(state: GuidelineState):
    base_dir = os.path.join(project_root, "results_2026_relaxed_with_post_processing")

    for diff_entry in state["extracted_diffs"]:
        current_pmcid = diff_entry["current_pmcid"]
        prior_pmcid = diff_entry["prior_pmcid"]

        save_pair_outputs(
            base_dir=base_dir,
            current_pmid=current_pmcid,
            prior_pmid=prior_pmcid,
            extracted_diffs=[
                d.model_dump() if isinstance(d, BaseModel) else d
                for d in diff_entry["difference"]
            ]
        )

    return {}


    # B) NEW STRATIFIER HANDLING:
    # Carefully analyze the case where the current adds extra risk factors/stratifiers (e.g., age, sex). Do not reject (Bad, score=0) if:
    #     (i) the BASELINE recommendation for the shared population is comparable, AND
    #     (ii) the new stratifiers are presented as examples/considerations for selective use, not as a completely new primary target population.

@retry_on_ratelimit(max_retries=6)
def verify_difference(state: GuidelineState):
    prompt_template = """
### Assignment

You are evaluating whether the **prior** and **current** clinical guideline recommendations shown below exhibit a **strict head-to-head change**.

Your task is to determine whether the current recommendation truly replaces, updates, or meaningfully modifies the prior recommendation **for the same explicit clinical decision unit**, and assign a score accordingly.

---

## SCORING

- GOOD = 1
- OK = 0.5
- BAD = 0

---

## STRICT HEAD-TO-HEAD REQUIREMENTS

### 1. SAME EXPLICIT CLINICAL DECISION UNIT (MANDATORY)

The two recommendations must address the **same explicitly stated clinical unit**, including:

- the same disease or condition (as explicitly named)
- the same patient population
- the same disease stage / severity / control status
- the same clinical decision point  
  (e.g., first-line therapy, step-up therapy, elective surgery after non-operative management)

IMPORTANT:
- Clinical reasoning, medical ontology, or pathophysiologic hierarchy MUST NOT be used to infer equivalence.
- The clinical decision unit in the two recommendations must be explicitly the same. Wording may differ slightly ONLY if the two recommendations refer to the same explicitly stated disease, population, and decision point, but they must refer to the same decision unit as named in the text.
- **Overlap, subset, or superset relationships do NOT qualify as the same unit**.
- If the wording in the prior and the current recommendation are **NOT the exactly the SAME** and you think they are refering to the SAME clinical decision unit, YOU MUST CLEARLY EXPLAIN WHY in your `rationale` with a starting hashtag <reason>!!!

A broader condition in the prior guideline CANNOT be treated as implicitly covering a more specific subtype or complication introduced only in the current guideline.

Example (NOT head-to-head):
- prior: “complicated diverticulitis”
- current: “pelvic abscess treated with percutaneous drainage”

Even if the subtype is clinically part of the broader category, this is considered a **decision-unit mismatch**.

---

### 2. POPULATION ALIGNMENT (TEXT-EXPLICIT, NOT INFERRED)

- Population descriptions **MUST not differ in wording**, otherwise should be rejected (BAD: 0.0).
- Any stratification used in the **current** recommendation (e.g., uncontrolled disease, refractory cases, abscess subtype, treatment pathway, eligibility subgroup) MUST also be explicitly present in the prior recommendation.
- You may NOT assume that the prior recommendation applies to a subgroup unless it is explicitly mentioned.

#### Allowed Exception: Refinement of Enumerated Categories

If the prior recommendation **explicitly enumerates multiple categories**, procedures, or settings (e.g., THA, TKA, HFS; mild/moderate/severe disease), the current recommendation MAY provide more specific guidance for those same enumerated items.

This is considered a valid refinement ONLY if no new category, subtype, or decision axis is introduced.

---

### 3. REFINEMENT VS. FRAMEWORK CHANGE

Refinement is allowed ONLY along axes that **already exist** in the prior guideline.

- Valid refinement:
  - prior explicitly lists A, B, C
  - current provides more specific guidance for A and/or B

- Invalid refinement (framework change):
  - current introduces new disease subtypes, complications, treatment pathways, or eligibility gates
  - these elements were NOT explicitly defined in the prior recommendation
  - This type of refinement should be **directly rejected** (Bad: 0.0)

Framework changes are NOT head-to-head comparisons, even if clinically reasonable.

---

### 4. TRUE CHANGE IN GUIDANCE

To qualify as a meaningful change:

- The current recommendation must replace, restrict, expand, or clearly supersede the prior guidance for the same decision unit.

The following do NOT qualify as substantive change:
- increased certainty without strategy change
- stronger wording without new clinical action
- background explanation or rationale
- administrative or non-clinical updates

---

### 5. CLARITY REQUIREMENT

Both the prior and current recommendations must be clear and specific enough to support comparison.

Vague language (e.g., “controversial”, “case-by-case”, “may be considered” without context) weakens the comparison.

---

## SCORING GUIDELINES

### GOOD (1.0)

Assign GOOD if and only if ALL of the following are met:

- same explicit clinical decision unit
- no population or stratification mismatch
- a clear, substantive replacement or modification of guidance
- both recommendations are clear and specific

Typical pattern:
- prior A → current B (A is explicitly replaced, restricted, or superseded)

---

### OK (0.5)

Assign OK if ALL of the following are met:

- same explicit clinical decision unit
- no new disease subtype or decision axis introduced
- BUT:
  - the change is modest, optional, or asymmetric
  - OR the current guideline largely affirms the prior recommendation and adds limited caveats or special situations
  - OR one of the recommendations lacks sufficient specificity

Typical pattern:
- prior A remains valid
- current adds special cases, optional adjustments, or clarifications

---

### BAD (0.0)

Assign BAD if ANY of the following apply:

- different disease, population, stage, or decision unit
- new stratification, subtype, or decision axis introduced only in the current guideline
- refinement occurs along an axis not present in the prior guideline
- no meaningful change in guidance
- one or both recommendations are too vague to compare

---

## IMPORTANT NOTE

- Be conservative and annotation-faithful.
- Do NOT reward clinical plausibility, inferred hierarchy, or medical reasoning beyond what is explicitly stated in the text.
- Only explicit, text-level alignment qualifies as a head-to-head comparison.

## IMPORTANT CLARIFICATION ON SUBSTANTIVE CHANGE:

- A change that only removes or alters relative preference, recommendation strength, or wording (e.g., “may be preferred” vs “no stated preference”), WITHOUT introducing or excluding any clinical action, SHOULD NOT be considered a substantive change.

Such changes should be scored as OK (0.5), not GOOD (1).

"""
    class VerifyScore(BaseModel):
        score: float = Field(
            ...,
            description="0, 0.5, or 1 based on the given scoring rules.",
        )
        rationale: str = Field(
            ...,
            description="Brief justification for the score.",
        )

    if not state.get("extracted_diffs"):
        return {}

    model = AzureChatOpenAI(
        azure_deployment=getattr(config, "AZURE_GPT5_DEPLOYMENT", config.AZURE_DEPLOYMENT),
        # reasoning={
        #     'effort': 'high'
        # },
        api_key=config.AZURE_OPENAI_API_KEY,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_version=config.AZURE_OPENAI_API_VERSION,
        temperature=1,
    )

    verifier = model.with_structured_output(VerifyScore)
    verify_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a very strict verifier. You are evaluating whether the prior and current clinical guideline recommendations shown below truly exhibit a head‑to‑head change. Your goal is to determine if the current recommendation replaces or updates the prior one for the **same clinical scenario** based on the given scoring rules. You MUST follow the given scoring rules.",
        ),
        (
            "human",
            prompt_template
            + """
==============Extracted Change==============
Clinical focus:\n\n{clinical_focus}\n\n
Current recommendation:\n\n{current_recommendation}\n\n
Current recommendation (raw text):\n\n{current_recommendation_raw_text}\n\n
Prior recommendation:\n\n{prior_recommendation}\n\n
Prior recommendation (raw text):\n\n{prior_recommendation_raw_text}\n\n
""",
        ),
    ])

    filtered_entries = []
    for diff_entry in state["extracted_diffs"]:
        kept_diffs = []
        for diff in diff_entry["difference"]:
            resp = verifier.invoke(
                verify_prompt.format_messages(
                    clinical_focus=diff.clinical_focus,
                    current_recommendation=diff.current_recommendation,
                    current_recommendation_raw_text=diff.current_recommendation_raw_text,
                    prior_recommendation=diff.prior_recommendation,
                    prior_recommendation_raw_text=diff.prior_recommendation_raw_text,
                    change_summary=diff.change_summary,
                )
            )
            diff_with_meta = diff.model_dump()
            diff_with_meta["score"] = resp.score
            diff_with_meta["rationale"] = resp.rationale

            if resp.score > 0.5:
                kept_diffs.append(diff_with_meta)
            
            # print(state["current_pmid"])
            # print(state["prior_pmcids"])
            # print(diff.current_recommendation)
            # print(diff.prior_recommendation)
            # print(resp.score)
            # print(resp.rationale)

        # if kept_diffs:
        filtered_entry = dict(diff_entry)
        filtered_entry["difference"] = kept_diffs
        filtered_entries.append(filtered_entry)

        

    return {"extracted_diffs": filtered_entries}

if __name__ == "__main__":
    graph = StateGraph(GuidelineState)
    graph.add_node("load_json", load_json)
    graph.add_node("validate_json", validate_json)
    graph.add_node("resolve_pmids", resolve_pmids)
    graph.add_node("load_pmc_xml", load_pmc_xml)
    graph.add_node("extract_recommendations", extract_recommendations)
    graph.add_node("persist_outputs", persist_outputs)
    graph.add_node("verify_difference", verify_difference)

    graph.set_entry_point("load_json")

    graph.add_edge("load_json", "validate_json")
    graph.add_conditional_edges("validate_json", validation_router)
    graph.add_edge("resolve_pmids", "load_pmc_xml")
    graph.add_conditional_edges("load_pmc_xml", validate_pmid)
    graph.add_edge("extract_recommendations", "verify_difference")
    graph.add_edge("verify_difference", "persist_outputs")
    graph.add_edge("persist_outputs", END)

    app = graph.compile()
    restore = False
    for task_file in [
        os.path.join(project_root, "noncomm_guideline_trajectory_2026_relaxed_augmented_year_calibrated", "*", "*.json"),
        os.path.join(project_root, "comm_guideline_trajectory_2026_relaxed_augmented_year_calibrated", "*", "*.json"),
        os.path.join(project_root, "other_guideline_trajectory_2026_relaxed_augmented_year_calibrated", "*", "*.json"),
    ]:
        json_files = sorted(glob.glob(
            task_file,
            recursive=True
        ), reverse=True)
        
        for file in tqdm(json_files):
            file_name = file.split("/")[-1]
            if file_name == "PMC9706842.json":
                restore = True
            
            if restore:
                try:
                    state = GuidelineState({"json_path": file})
                    result = app.invoke(state)

                except Exception as e:
                    print(f"[ERROR] {file}: {e}")
                    continue
