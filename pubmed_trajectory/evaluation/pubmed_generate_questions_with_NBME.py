import os
import re
import json
import glob
import time
import random
import typing as T
import sys
from dataclasses import dataclass, field

import requests
from pydantic import BaseModel, Field

# LangGraph / LangChain
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import Runnable
from langchain_openai import AzureChatOpenAI
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from utils import config  # your config file

def build_azure_llm():
    return AzureChatOpenAI(
        azure_deployment=config.AZURE_DEPLOYMENT,
        api_key=config.AZURE_OPENAI_API_KEY,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_version=config.AZURE_OPENAI_API_VERSION,
        temperature=1 if "gpt-5" in str(config.AZURE_DEPLOYMENT).lower() else 0,
    )
###############################################################################
# Config
###############################################################################

PMC_FILE_RE = re.compile(r"^(PMC\d+)_+(PMC\d+)_+extracted_diffs\.json$", re.IGNORECASE)

NCBI_PMC_IDCONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
PUBMED_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

DEFAULT_USER_AGENT = os.getenv("NCBI_USER_AGENT", "guideline-qa-pipeline/1.0 (contact: youremail@example.com)")
REQUEST_TIMEOUT = 30

# NCBI recommends not hammering APIs; keep it polite
BASE_SLEEP_SEC = 0.12
JITTER_SEC = 0.15


###############################################################################
# Helpers: NCBI API calls
###############################################################################

def _sleep_polite():
    time.sleep(BASE_SLEEP_SEC + random.random() * JITTER_SEC)


def pmc_to_pmid(
    pmcid: str,
    session: requests.Session | None = None,
    max_retries: int = 5,
) -> str | None:
    url = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
    params = {"ids": pmcid, "format": "json"}
    headers = {"User-Agent": DEFAULT_USER_AGENT}

    for attempt in range(max_retries):
        try:
            if session is None:
                r = requests.get(url, params=params, headers=headers, timeout=10)
            else:
                r = session.get(url, params=params, headers=headers, timeout=10)

            if r.status_code == 429:
                raise requests.HTTPError("429", response=r)

            r.raise_for_status()
            data = r.json()

            records = data.get("records", [])
            if records and "pmid" in records[0]:
                return str(records[0]["pmid"])
            return None

        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                sleep_s = (2 ** attempt) + random.random()
                print(f"[pmc_to_pmid] 429 for {pmcid}, retry in {sleep_s:.2f}s")
                time.sleep(sleep_s)
            else:
                raise

    raise RuntimeError(f"PMC→PMID failed after {max_retries} retries for {pmcid}")

def pubmed_year(pmid: str, session: requests.Session | None = None) -> int | None:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "json",
    }
    headers = {"User-Agent": DEFAULT_USER_AGENT}

    if session is None:
        r = requests.get(url, params=params, headers=headers, timeout=10)
    else:
        r = session.get(url, params=params, headers=headers, timeout=10)

    r.raise_for_status()
    data = r.json()

    result = data.get("result", {})
    obj = result.get(str(pmid), {})

    # Try common fields in order
    pubdate = (
        obj.get("pubdate")
        or obj.get("sortpubdate")
        or obj.get("epubdate")
        or ""
    )

    m = re.search(r"(19|20)\d{2}", pubdate)
    if m:
        return int(m.group(0))
    print('!!!')
    return None


###############################################################################
# LLM output schema (flexible for question types later)
###############################################################################

class QAItem(BaseModel):
    question_type: str = Field(default="binary_guideline_change")
    Question: str = Field(description="A hard but clear Multiple-choice question.")
    Choice_A: str = Field(description="Option A: corresponds to the CURRENT guideline.")
    Choice_B: str = Field(description="Option B: corresponds to the PRIOR guideline.")
    Choice_C: str = Field(description="Option C: Option C: corresponds to neither the current nor the prior guideline; serves as a distractor option to introduce interference.")
    Choice_D: str = Field(description="Option D: Option C: corresponds to neither the current nor the prior guideline; serves as a distractor option to introduce interference.")
    Correct_Answer: str = Field(description="A single option from 'A', 'B', 'C', or 'D'.")
    Explanation: str = Field(description="Short explanation of why that option is correct.")

###############################################################################
# Graph state
###############################################################################

@dataclass
class DiffRecord:
    pmc_current: str
    pmc_prior: str
    diff: dict
    src_file: str

@dataclass
class OutputRecord:
    idx: int
    PMID_current: str
    Year_current: int
    PMID_prior: str
    Year_prior: int
    Question: str
    Answer: str
    # keep room for future expansion
    question_type: str = "guideline_change"
    clinical_focus: T.Optional[str] = None
    change_summary: T.Optional[str] = None

@dataclass
class PipelineState:
    results_dir: str
    out_path: str
    question_type: str = "guideline_change"  # flexible hook
    records: list[DiffRecord] = field(default_factory=list)
    outputs: list[OutputRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


###############################################################################
# Node 1: Load diffs from files
###############################################################################

def node_load_diffs(state: PipelineState) -> PipelineState:
    files = glob.glob(os.path.join(state.results_dir, "*_extracted_diffs.json"))
    for fp in sorted(files):
        name = os.path.basename(fp)
        m = PMC_FILE_RE.match(name)
        if not m:
            # skip files not matching your pattern
            continue
        pmc_current, pmc_prior = m.group(1), m.group(2)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                diffs = json.load(f)
            if not isinstance(diffs, list):
                raise ValueError("JSON root is not a list")
            for d in diffs:
                if isinstance(d, dict):
                    state.records.append(DiffRecord(
                        pmc_current=pmc_current,
                        pmc_prior=pmc_prior,
                        diff=d,
                        src_file=name,
                    ))
        except Exception as e:
            state.errors.append(f"[load] {name}: {repr(e)}")
    return state


###############################################################################
# Node 2: Enrich each record with PMID+Year, then ask LLM to make QA
###############################################################################

def build_prompt(diff: dict, question_type: str, organization:str, current_year, prior_year):
    clinical_focus = diff.get("clinical_focus", "").strip()
    current_rec = diff.get("current_recommendation", "").strip()
    prior_rec = diff.get("prior_recommendation", "").strip()
    current_raw = diff.get("current_recommendation_raw_text", "").strip()
    prior_raw = diff.get("prior_recommendation_raw_text", "").strip()
    change_summary = diff.get("change_summary", "").strip()

    system = (
"You are an expert medical exam writer trained in NBME item-writing guidelines.\n"
"You are generating evaluation questions about CHANGES between clinical guidelines.\n\n"
"==============================\n"
"TASK\n"
"==============================\n"
"Create ONE high-quality multiple-choice question that evaluates knowledge of a CHANGE between two clinical guidelines.\n\n"
"Answer option requirements:\n"
"- Choice A MUST represent the CURRENT guideline recommendation.\n"
"- Choice B MUST represent the PRIOR guideline recommendation.\n"
"- Choices C and D must represent medically plausible alternatives that are NOT the current or prior recommendation.\n\n"
"Difficulty requirements:\n"
"- The question must be HARD and CLEAR.\n"
"- It must require understanding what changed between guidelines.\n"
"- It must describe a realistic clinical scenario.\n"
"- The correct answer should not be obvious without knowing the guideline update.\n\n"
"Guideline citation requirement:\n"
"- The question must explicitly cite the guideline source and year.\n"
"Example: 'According to the Society for Immunotherapy of Cancer guideline issued in 2023...'\n\n"
"Important restriction:\n"
"- Do NOT mention the words 'current' or 'prior' in the question text.\n"
"- Those labels are only used internally for answer construction.\n\n"
"==============================\n"
"NBME ITEM WRITING RULES\n"
"==============================\n"
"1. Test APPLICATION of knowledge, not recall of isolated facts.\n\n"
"2. Use a clinical vignette structured as:\n"
"   - patient demographics\n"
"   - chief complaint\n"
"   - relevant history\n"
"   - physical examination\n"
"   - laboratory or imaging findings\n\n"
"3. Write a CLOSED and FOCUSED lead-in question.\n\n"
"Acceptable lead-ins include:\n"
"- Which of the following is the most appropriate next step in management?\n"
"- Which of the following is recommended according to the guideline?\n"
"- Which of the following interventions should be initiated?\n\n"

"Avoid vague lead-ins such as:\n"
"- Which statement is true?\n"
"- What is associated with this condition?\n\n"

"4. Create EXACTLY FOUR answer options (A–D).\n\n"

"Rules for options:\n"
"- Only ONE option is the best answer.\n"
"- All options must be HOMOGENEOUS (same category such as treatments, tests, or diagnoses).\n"
"- Distractors must be medically plausible.\n"
"- Options must have similar length and grammatical structure.\n\n"

"5. Avoid technical flaws:\n"
"- Do NOT use 'all of the above' or 'none of the above'.\n"
"- Avoid vague terms such as 'often', 'usually', 'frequently'.\n"
"- Avoid absolute terms such as 'always' or 'never'.\n"
"- Avoid grammatical cues that reveal the correct answer.\n"
"- Avoid repeating distinctive words from the vignette in only one option.\n\n"

"6. Apply the COVER-THE-OPTIONS rule:\n"
"A knowledgeable reader should be able to infer the correct answer after reading the vignette and lead-in before seeing the options.\n\n"

"7. Ensure the question tests clinical reasoning and guideline application.\n\n"

# "8. The question stem should not reflect the preferecens of choices.\n\n"
"Important:\nDo not encode the guideline change into the patient scenario unless the guideline explicitly conditions the recommendation on those features.\n\n"
)

#     system = (
#         "You are generating evaluation questions about CHANGES between clinical guidelines.\n\n"
#         "Task:\n"
#         "- Create ONE Multiple-choice question.\n"
#         "- Choice A MUST represent the CURRENT guideline.\n"
#         "- Choice B MUST represent the PRIOR guideline.\n"
#         "- Choices C and D must correspond to neither the current nor the prior guideline recommendation and should serve as medically plausible distractor options.\n"
#         "- The question must be HARD and CLEAR:\n"
#         "  * It must require comparing current vs prior guidance.\n"
#         "  * It must describe a specific clinical scenario.\n"
#         "  * It must not be answerable without knowing what changed.\n"
#         "- The question must clearly indicate which guideline it is based on (e.g., “According to the Society for Immunotherapy of Cancer guideline issued in 2023”).\n\n"
#         "Output fields:\n"
#         "- Question\n"
#         "- Choice_A (current guideline)\n"
#         "- Choice_B (prior guideline)\n"
#         "- Choice_C (Distractor option)\n"
#         "- Choice_D (Distractor option)"
#         "- Correct_Answer: 'A' or 'B' or 'C' or 'D'\n"
#         "- Explanation\n"
#         "Do not mention 'current' or 'prior' in the question text.\n"
#         """
# ==============================
# NBME ITEM WRITING RULES
# ==============================

# 1. Test APPLICATION of knowledge, not recall of isolated facts.

# 2. Use a clinical vignette structured as:
#    - patient demographics
#    - chief complaint
#    - relevant history
#    - physical examination
#    - laboratory or imaging findings

# 3. Write a CLOSED and FOCUSED lead-in question.

# Examples of acceptable lead-ins:
# - Which of the following is the most likely diagnosis?
# - Which of the following is the most appropriate next step in management?
# - Which of the following best explains the mechanism of this condition?

# Avoid vague questions such as:
# - Which statement is true?
# - What is associated with this condition?

# 4. Create EXACTLY FOUR answer options (A–D).

# Rules for options:
# - Only ONE option is the best answer.
# - All options must be HOMOGENEOUS (same category).
# - All distractors must be plausible.
# - Options must be similar length and structure.

# 5. Avoid technical flaws:
# - no "all of the above"
# - no "none of the above"
# - avoid vague terms like "often", "usually"
# - avoid absolute terms like "always", "never"
# - avoid grammatical cues
# - avoid repeating words from the vignette in the correct answer

# 6. Apply the COVER-THE-OPTIONS rule:
# A knowledgeable reader should be able to guess the answer after reading the vignette and lead-in without seeing the options.

# 7. Ensure the question tests clinical reasoning.
#         """
#     )

    user = (
        f"Clinical focus:\n{clinical_focus}\n\n"
        f"Prior recommendation (summary):\n{prior_rec}\n\n"
        f"Prior recommendation (raw text):\n{prior_raw}\n\n"
        f"Current recommendation (summary):\n{current_rec}\n\n"
        f"Current recommendation (raw text):\n{current_raw}\n\n"
        f"Change summary:\n{change_summary}\n\n"
        f"Guideline Organization: {organization}\n"
        f"Year of Current Guideline Issued: {current_year}\n"
        f"Year of Prior Guideline Issued: {prior_year}\n"
        "Generate the Multiple-choice question now."
    )

    example = """
=======================================
The following is an example of the desired evaluation question.

=======================================The summary of guideline difference=======================================
clinical_focus: 
Adjuvant imatinib duration after complete resection in high‑risk GIST

Prior recommendation (summary):
In patients with resected localized GIST at risk of relapse, administer adjuvant imatinib for at least one year; 1 year is likely insufficient for high-risk patients, and the optimal duration was undetermined pending ongoing trials.

Prior recommendation (raw text): 
With the currently available data, it is recommended to use imatinib as an adjuvant treatment for at least one year. And, most of the experts agree that 1 yr of adjuvant imatinib is not long enough for especially patients at high risk. However, optimal duration of adjuvant imatinib remains yet to be determined. Phase III trials comparing one year versus three years and no treatment versus two years of treatment duration are ongoing.

Current recommendation (summary):
"In patients with high-risk GIST after complete resection, administer adjuvant imatinib for at least 36 months.",

Current recommendation (raw text):
With the currently available data, it is recommended to use imatinib as an adjuvant treatment for at least 36 months for patients with high-risk GISTs [II, C]. However, an optimal duration of adjuvant imatinib treatment has yet to be determined.

Change summary: 
New randomized data (SSGXVIII/AIO) showed superior recurrence-free and overall survival with 36 vs 12 months, shifting the recommendation from a minimum of 1 year (with uncertain optimal duration) to at least 3 years for high-risk patients."

Year of Current Guideline Issued: 2012

Year of Prior Guideline Issued: 2010

Generate the Multiple-choice question now.

=======================================The Generated Question=======================================
A 58-year-old man presents for follow-up 3 weeks after undergoing complete surgical resection of a 9-cm gastric mass. Pathology confirms a gastrointestinal stromal tumor (GIST) with a high mitotic rate (>10 mitoses per 50 high-power fields). Surgical margins are negative. Molecular testing reveals a KIT exon 11 mutation. Staging studies show no evidence of metastatic disease. 

He has recovered well from surgery and has no significant comorbidities. He asks about further treatment to reduce the risk of recurrence.

According to the 2012 Korean Gastrointestinal Stromal Tumor Study Group (KGSG) clinical practice guideline, which of the following is the most appropriate next step in management?

A. Adjuvant imatinib therapy for at least 36 months
B. Adjuvant imatinib therapy for 12 months
C. Observation with serial CT imaging every 6 months
D. Adjuvant imatinib therapy indefinitely

Explanation:
This patient has a high-risk gastrointestinal stromal tumor (GIST) based on tumor size (9 cm) and a high mitotic rate (>10 mitoses per 50 high-power fields). Risk stratification for GIST recurrence after complete resection depends primarily on tumor size, mitotic index, tumor location, and the presence of tumor rupture. Large tumors (>5 cm), especially those with elevated mitotic activity (>5 per 50 HPF), carry a substantial risk of recurrence. Although gastric location is generally more favorable than small bowel tumors, the combination of large size and high mitotic index places this patient in the high-risk category. Current management guidelines recommend adjuvant imatinib therapy for at least 36 months in patients with high-risk GIST after complete resection, as prolonged therapy significantly improves recurrence-free and overall survival compared with shorter durations. A 12-month course reflects prior recommendations and is now considered insufficient for high-risk disease.

Correct: A

Note:
- Option A corresponds to the recommendation in the current guideline.
- Option B corresponds to the recommendation in the prior guideline.
- Options C and D correspond to neither guideline and serve as medically plausible distractor options.
"""

    return system, user, example

def node_generate_qa(state: PipelineState) -> PipelineState:
    files = glob.glob(os.path.join("comm_guideline_trajectory_2026_relaxed_augmented_year_calibrated", "*/*.json")) + glob.glob(os.path.join("noncomm_guideline_trajectory_2026_relaxed_augmented_year_calibrated", "*/*.json")) + glob.glob(os.path.join("other_guideline_trajectory_2026_relaxed_augmented_year_calibrated", "*/*.json"))
    llm = build_azure_llm()
    structured_llm: Runnable = llm.with_structured_output(QAItem)
    with requests.Session() as session:
        for idx, rec in tqdm(enumerate(state.records), desc="Generating QA", unit="diff"):
            try:
                pmid_current = pmc_to_pmid(rec.pmc_current, session=session)
                year_of_current_guidance = None
                organization = None
                for file in files:
                    if rec.pmc_current in file:
                        print(file)
                        with open(file, "r") as f:
                            json_file = json.load(f)
                            year_of_current_guidance = json_file["year_of_current_guidance"]
                            organization = json_file["Organization"]
                
                pmid_prior = pmc_to_pmid(rec.pmc_prior, session=session)
                if not pmid_current or not pmid_prior:
                    raise ValueError(f"PMC→PMID failed for {rec.pmc_current} or {rec.pmc_prior}")
                
                
                for i in range(5):
                    try:
                        year_current = pubmed_year(pmid_current, session=session)
                        year_prior = pubmed_year(pmid_prior, session=session)
                        break
                    except Exception as e:
                        if i == 4:
                            raise e
                        time.sleep(3)

                if not year_of_current_guidance or not organization:
                    print(year_of_current_guidance, organization, year_current, year_prior)
                    raise NotImplementedError

                system_prompt, user_prompt, example_prompt = build_prompt(
                    rec.diff,
                    question_type=state.question_type,
                    organization=organization,
                    current_year=year_current,
                    prior_year=year_prior
                )

                qa: QAItem = structured_llm.invoke([
                    SystemMessage(content=system_prompt),
                    AIMessage(content=example_prompt),
                    HumanMessage(content=user_prompt),
                ])

                out = OutputRecord(
                    idx=idx,
                    PMID_current=pmid_current,
                    Year_current=year_current,
                    PMID_prior=pmid_prior,
                    Year_prior=year_prior,
                    Question=qa.Question.strip(),
                    Answer={
                        "Choice_A": qa.Choice_A,
                        "Choice_B": qa.Choice_B,
                        "Choice_C": qa.Choice_C,
                        "Choice_D": qa.Choice_D,
                        "Correct": qa.Correct_Answer,
                        "Explanation": qa.Explanation,
                    },
                    question_type=qa.question_type,
                    clinical_focus=rec.diff.get("clinical_focus"),
                    change_summary=rec.diff.get("change_summary"),
                )
                state.outputs.append(out)

            except Exception as e:
                print(f"[gen] {rec.src_file} / focus={rec.diff.get('clinical_focus','')}: {repr(e)}")
                state.errors.append(
                    f"[gen] {rec.src_file} / focus={rec.diff.get('clinical_focus','')}: {repr(e)}"
                )

    return state


###############################################################################
# Node 3: Write outputs
###############################################################################

def node_write_outputs(state: PipelineState) -> PipelineState:
    os.makedirs(os.path.dirname(state.out_path) or ".", exist_ok=True)
    with open(state.out_path, "w", encoding="utf-8") as f:
        for out in state.outputs:
            # Keep exactly the fields you requested, but keep flexibility via question_type optionally.
            payload = {
                "idx": out.idx,
                "PMID_current": out.PMID_current,
                "Year_current": out.Year_current,
                "PMID_prior": out.PMID_prior,
                "Year_prior": out.Year_prior,
                "Question": out.Question,
                "Answer": out.Answer,
            }
            f.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n\n")
    return state


###############################################################################
# Build graph
###############################################################################

def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("load_diffs", node_load_diffs)
    g.add_node("generate_qa", node_generate_qa)
    g.add_node("write_outputs", node_write_outputs)

    g.set_entry_point("load_diffs")
    g.add_edge("load_diffs", "generate_qa")
    g.add_edge("generate_qa", "write_outputs")
    g.add_edge("write_outputs", END)
    return g.compile()


###############################################################################
# CLI entry
###############################################################################

def run(results_dir: str, out_path: str, question_type: str = "guideline_change"):
    state = PipelineState(
        results_dir=results_dir,
        out_path=out_path,
        question_type=question_type,
    )
    app = build_graph()
    final = app.invoke(state)
    print(final.keys())

    print(f"Loaded diffs: {len(final['records'])}")
    print(f"Generated QA: {len(final['outputs'])}")
    if final['errors']:
        print("\nErrors:")
        for e in final['errors'][:50]:
            print(" -", e)
        if len(final['errors']) > 50:
            print(f" ... and {len(final['errors']) - 50} more")

    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    # Example:
    #   export OPENAI_API_KEY=...
    #   python build_guideline_qa.py ./results_merged ./qa_outputs.jsonl
    import sys
    if len(sys.argv) < 3:
        print("Usage: python build_guideline_qa.py <results_merged_dir> <out_jsonl_path> [question_type]")
        raise SystemExit(2)
    results_dir = sys.argv[1]
    out_path = sys.argv[2]
    qtype = sys.argv[3] if len(sys.argv) >= 4 else "guideline_change"
    run(results_dir, out_path, qtype)