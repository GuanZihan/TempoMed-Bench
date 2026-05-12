from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from tqdm import tqdm
import glob
import os
import tiktoken
import time
import asyncio
from asyncio import Semaphore
from tqdm.asyncio import tqdm_asyncio
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(root_dir)
sys.path.append(root_dir)
from utils import config


MAX_CONCURRENT_REQUESTS = 6
RETRIES = 5
SLEEP_ON_FAIL = 30
sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
XML_MAX_LENGTH = 400000
PROMPT = """
Based on the given article, identify **all prior clinical guidelines** issued by the **same organization** that address the **same topic** as the current guideline discussed in the article.

A prior guideline is defined as an official guidance document issued by the **same organization** before the year of the current guidance and focuses on the **same topic** as the current guideline.

You are encouraged to identify a comprehensive list of such prior guidelines.

For each identified prior guideline:

- Extract the year of publication
- Extract the PMID (if available)
- Extract the issuing organization
- Extract the title
- Provide the reason why this is a prior guideline

## Process:
0. Determine whether the paper itself is a clinical guideline issued by a professional society (e.g., the American College of Rheumatology).
    - If yes, proceed with the steps below and extract prior guidelines.
    - If no, do not record any prior guidelines and set the prior guidelines as an empty list.
1. Examine the *Introduction*, *Related Works* or *Literature Review* sections to identify documents cited as prior guidelines that **meet the above criteria**;
2. Record the `year`, `PMID`, `Organization`, and `title` of each previous guideline.
3. Explain why each document is identified as a prior guideline that precedes the current one (e.g., explicit citation, chronological ordering, or stated replacement).

## Important Notes:
- If multiple prior guidelines exist, include all of them.
- If a field is not available, return `None` for that field. For example, if the old guideline is from the official website with no PMIDs, you can just use PMID: None
- **DO NOT generate or hallucinate PubMed IDs.**
- If no prior guidelines are identified, return an empty list.
- Do **not** classify a document as a prior guideline if it is merely a translation of the current guideline with no substantive content changes (e.g., Chinese or Japanese versions).
```json
{{
  "Topic": "Systemic Lupus Erythematosus",
  "Title": "2025 American College of Rheumatology (ACR) Guideline for the Treatment of Systemic Lupus Erythematosus",
  "year_of_current_guidance": 2025,
  "Organization": ["American College of Rheumatology"],
  "PMID": 41182321
  "prior_guidelines": [
    {{
      "year": 1999,
      "PMID": 10513791,
      "Organization": "American College of Rheumatology",
      "title": "Guidelines for referral and management of systemic lupus erythematosus in adults. American College of Rheumatology Ad Hoc noncommittee on Systemic Lupus Erythematosus Guidelines"
      "reason": "The paper explicitly mentions that 'Guidelines for management of SLE in adults were lastpublished by the American College of Rheumatology in 1999[8].' and [8] corresponds to PMID 10513791."
    }}
  ]
}}
```
"""

# ---- 1. Define the state ----
class KnowledgeState(dict):
    """Holds prompt, XML path, and outputs."""
    prompt: str
    xml_path: str
    output: str
    xml_text: str
    token_usage: dict


class PriorGuideline(BaseModel):
    year: int = Field(
        ..., 
        description="Year when the prior guideline was published"
    )
    PMID: int = Field(
        ..., 
        description="PubMed ID of the prior guideline; If there is no available PMID, just yield 0"
    )
    Organization: str = Field(
        ..., 
        description="Issuing organization of the prior guideline"
    )

    title: str = Field(
        ...,
        description="Title of the prior guideline; should be the full content inside the tag of <article-title></article-title>"
    )

    reason: str = Field(
        ...,
        description="The rationale for why this guideline preceded the current one."
    )


class Item(BaseModel):
    Topic: str = Field(
        ..., 
        description="Disease or clinical condition"
    )
    Title: str = Field(
        ...,
        description="Title the current guidance"
    )
    year_of_current_guidance: int = Field(
        ..., 
        description="Year the current guidance was published"
    )
    PMID: int = Field(
        ...,
        description="PubMed ID of the current guideline"
    )
    Organization: List[str] = Field(
        ..., 
        description="Issuing organization(s) of the current guidance"
    )
    prior_guidelines: List[PriorGuideline] = Field(
        None, 
        description=(
            "All prior clinical guidelines addressing the same topic. "
            "Return an empty list if none are identified."
        )
    )

# class ItemList(BaseModel):
#     items: List[Item] = Field(..., description="List of guideline trajectories")

# ---- 2. Define node functions ----
def load_xml(state: KnowledgeState) -> KnowledgeState:
    """Loads XML content into the state."""
    with open(state["xml_path"], "r", encoding="utf-8") as f:
        xml_text = f.read()
    state["xml_text"] = xml_text
    return state


async def call_gpt(state: KnowledgeState) -> KnowledgeState:
    """Calls GPT-5 and ensures JSON output matching the target schema."""
    
    model = AzureChatOpenAI(
        azure_deployment=config.AZURE_DEPLOYMENT,
        api_key=config.AZURE_OPENAI_API_KEY,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_version=config.AZURE_OPENAI_API_VERSION,
        temperature=0 if config.AZURE_DEPLOYMENT not in ['gpt-5'] else 1,
    )

    model = model.with_structured_output(Item)

    system_prompt = """You are an expert in evidence synthesis.
Extract a JSON array describing how medical guidance evolved.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{prompt}\n\nHere is the PubMed XML:\n\n{xml_text}")
    ])

    chain = prompt | model
    response = await chain.ainvoke({
        "prompt": state["prompt"],
        "xml_text": state["xml_text"]
    })

    state["output"] = response.model_dump()
    input_prompt_text = (
        system_prompt
        + "\n\n"
        + state["prompt"]
        + "\n\nHere is the PubMed XML:\n\n"
        + state["xml_text"]
    )
    input_tokens = estimate_tokens(input_prompt_text, model_name=config.AZURE_DEPLOYMENT)
    state["token_usage"]["input"] += input_tokens
    output_text = json.dumps(response.model_dump())
    output_tokens = estimate_tokens(output_text, model_name=config.AZURE_DEPLOYMENT)
    state["token_usage"]["output"] += output_tokens
    return state


# ---- 3. Build the LangGraph ----
graph = StateGraph(KnowledgeState)
graph.add_node("load_xml", load_xml)
graph.add_node("call_gpt", call_gpt)

graph.add_edge(START, "load_xml")
graph.add_edge("load_xml", "call_gpt")
graph.add_edge("call_gpt", END)

knowledge_graph = graph.compile()

def estimate_tokens(text, model_name="gpt-5"):
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(text))


async def process_one(xml_path):
    async with sem:
        pubmed_id = xml_path.split("/")[-1].split(".")[0]
        directory_name = xml_path.split("/")[-2]

        with open(xml_path, "r", encoding="utf-8") as f:
            xml = f.read()

        if len(xml) > XML_MAX_LENGTH:
            print(f"⏭️ Skipping {pubmed_id} (too large)")
            return None

        state = KnowledgeState(
            prompt=PROMPT,
            xml_path=xml_path,
            token_usage={"input": 0, "output": 0}
        )

        for i in range(RETRIES):
            try:
                return await knowledge_graph.ainvoke(state)
            except Exception as e:
                # print(e)
                print(f"⚠️ {pubmed_id} failed retry {i+1}/{RETRIES}")
                await asyncio.sleep(SLEEP_ON_FAIL)

        print(f"❌ {pubmed_id} permanently failed")
        return None


async def main():
    xml_files = sorted(
        glob.glob(
            os.path.join(project_root, "pmc_oa_comm_extracted_2026_relaxed", "**", "*.xml"),
            recursive=True,
        ),
        reverse=True,
    )


    # new_xml_files = []
    # for xmlfile in xml_files:
    #     directory_name = xmlfile.split("/")[-2]
    #     pubmed_id = xmlfile.split("/")[-1].split(".")[0]
    #     if os.path.exists(os.path.join("comm_guideline_trajectory_2026_relaxed", directory_name, f"{pubmed_id}.json")):
    #         pass
    #     else:
    #         new_xml_files.append(xmlfile)
    
    # print(new_xml_files)
    # print(len(new_xml_files))
    # input()

    os.makedirs("comm_guideline_trajectory_2026_relaxed", exist_ok=True)

    

    tasks = [asyncio.create_task(process_one(p)) for p in xml_files]

    progress = tqdm(total=len(tasks), desc="Processing XML")

    for future in asyncio.as_completed(tasks):
        result = await future
        progress.update(1)

        if not result:
            continue

        pubmed_id = result["xml_path"].split("/")[-1].split(".")[0]
        directory_name = result["xml_path"].split("/")[-2]

        os.makedirs(os.path.join("comm_guideline_trajectory_2026_relaxed", directory_name), exist_ok=True)

        out = json.dumps(result["output"], indent=2)

        with open(os.path.join("comm_guideline_trajectory_2026_relaxed", directory_name, f"{pubmed_id}.json"), "w") as f:
            f.write(out)

    progress.close()

# ---- 4. Example usage ----
if __name__ == "__main__":
    asyncio.run(main())