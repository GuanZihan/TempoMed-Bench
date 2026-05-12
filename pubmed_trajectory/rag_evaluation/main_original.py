import argparse
import csv
import importlib.util
import json
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import AzureOpenAI
from tooluniverse import ToolUniverse
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "questions_2026_relaxed_4_option_augmented_verified.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "pubmed_trajectory" / "rag_evaluation" / "rag_evaluation_results.json"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "pubmed_trajectory" / "rag_evaluation" / "rag_evaluation_summary.json"
DEFAULT_CSV_PATH = PROJECT_ROOT / "pubmed_trajectory" / "rag_evaluation" / "rag_evaluation_summary.csv"

ANSWER_LABELS = ("A", "B", "C", "D")
GROUP_ORDER = ("current_guideline", "prior_guideline", "irrelevant", "unknown", "invalid")
THREAD_LOCAL = threading.local()


SYSTEM_PROMPT = """You are a medical guideline QA model with access to retrieval tools.
Use the tools to identify the relevant guideline evidence before answering the question.
Answer the multiple-choice question with exactly one final option.
If the answer cannot be determined from available evidence, respond UNKNOWN.
"""


USER_PROMPT_TEMPLATE = """Question:
{question}

A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Return your final answer in this exact format:
##Explanation: <brief evidence-based explanation>
##Answer: <A/B/C/D/UNKNOWN>
"""


def load_shared_config():
    config_path = PROJECT_ROOT / "utils" / "config.py"
    if not config_path.exists():
        raise RuntimeError(
            f"Expected local config at {config_path}. Create an untracked utils/config.py before running."
        )

    spec = importlib.util.spec_from_file_location("project_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config from {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_questions(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            return payload
        raise ValueError(f"Expected a JSON list in {path}")

    # The project uses .jsonl for blank-line separated JSON objects.
    questions = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if block:
            questions.append(json.loads(block))
    return questions


def build_prompt(item: Dict[str, Any]) -> str:
    answer = item["Answer"]
    return USER_PROMPT_TEMPLATE.format(
        question=item["Question"],
        choice_a=answer["Choice_A"],
        choice_b=answer["Choice_B"],
        choice_c=answer["Choice_C"],
        choice_d=answer["Choice_D"],
    )


def extract_prediction(text: Optional[str]) -> str:
    if not text or not text.strip():
        return "INVALID"

    cleaned = text.strip()
    match = re.search(r"##\s*Answer\s*:\s*(A|B|C|D|UNKNOWN)\b", cleaned, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    first_line = cleaned.splitlines()[0].strip().upper()
    if first_line in ANSWER_LABELS or first_line == "UNKNOWN":
        return first_line

    if re.search(r"\b(I\s+DO\s+NOT\s+KNOW|UNKNOWN|UNABLE\s+TO\s+DETERMINE|CANNOT\s+DETERMINE)\b", cleaned, re.IGNORECASE):
        return "UNKNOWN"

    tokens = re.findall(r"\b(A|B|C|D)\b", cleaned.upper())
    unique_tokens = sorted(set(tokens))
    if len(unique_tokens) == 1:
        return unique_tokens[0]

    return "INVALID"


def prediction_group(prediction: str) -> str:
    if prediction == "A":
        return "current_guideline"
    if prediction == "B":
        return "prior_guideline"
    if prediction in {"C", "D"}:
        return "irrelevant"
    if prediction == "UNKNOWN":
        return "unknown"
    return "invalid"


GUIDELINE_TOOLS = [
    "PubMed_Guidelines_Search",
    "PubMed_get_article",
    "PubMed_search_articles",
    "web_search",
    "PMC_search_papers"
]


def initialize_tools(tool_names: Sequence[str]) -> Tuple[ToolUniverse, List[Dict[str, Any]]]:
    tu = ToolUniverse()
    tu.load_tools(include_tools=list(tool_names))
    specs = tu.get_tool_specification_by_names(list(tool_names), format="openai")
    tools = [{"type": "function", "function": spec} for spec in specs]
    return tu, tools


def run_one_question(
    client: AzureOpenAI,
    model: str,
    tu: ToolUniverse,
    tools: List[Dict[str, Any]],
    prompt: str,
    max_steps: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    tool_trace: List[Dict[str, Any]] = []

    for _ in range(max_steps):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0 if 'gpt-4' in model else 1,
        )

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content or "", tool_trace

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            try:
                result = tu.run({"name": name, "arguments": args})
            except Exception as exc:
                result = {"error": str(exc)}

            tool_trace.append({"name": name, "arguments": args, "result_preview": str(result)[:10000]})

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    return "INVALID", tool_trace


def get_thread_runtime(config: Any) -> Tuple[AzureOpenAI, ToolUniverse, List[Dict[str, Any]]]:
    runtime = getattr(THREAD_LOCAL, "runtime", None)
    if runtime is None:
        client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_API_KEY,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_version=config.AZURE_OPENAI_API_VERSION,
        )
        tu, tools = initialize_tools(GUIDELINE_TOOLS)
        runtime = (client, tu, tools)
        THREAD_LOCAL.runtime = runtime
    return runtime


def evaluate_one_item(
    item: Dict[str, Any],
    model: str,
    max_steps: int,
    config: Any,
) -> Dict[str, Any]:
    client, tu, tools = get_thread_runtime(config)
    prompt = build_prompt(item)
    try:
        model_answer, tool_trace = run_one_question(
            client=client,
            model=model,
            tu=tu,
            tools=tools,
            prompt=prompt,
            max_steps=max_steps,
        )
    except Exception as exc:
        model_answer = f"INVALID: {exc}"
        tool_trace = [{"error": str(exc)}]

    pred = extract_prediction(model_answer)
    gold = item["Answer"]["Correct"].strip().upper()
    group = prediction_group(pred)

    return {
        "idx": item.get("idx"),
        "PMID_current": item.get("PMID_current"),
        "PMID_prior": item.get("PMID_prior"),
        "Year_current": item.get("Year_current"),
        "Year_prior": item.get("Year_prior"),
        "question": item.get("Question"),
        "gold": gold,
        "prediction": pred,
        "prediction_group": group,
        "correct": pred == gold,
        "model_answer": model_answer,
        "tool_trace": tool_trace,
    }


def evaluate_questions(
    questions: Sequence[Dict[str, Any]],
    model: str,
    limit: Optional[int],
    max_steps: int,
    max_workers: int,
) -> List[Dict[str, Any]]:
    config = load_shared_config()
    selected = list(questions[:limit] if limit is not None else questions)
    results: List[Optional[Dict[str, Any]]] = [None] * len(selected)

    if max_workers <= 1:
        for idx, item in enumerate(tqdm(selected, desc="RAG evaluation", unit="question")):
            results[idx] = evaluate_one_item(item, model, max_steps, config)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(evaluate_one_item, item, model, max_steps, config): idx
                for idx, item in enumerate(selected)
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="RAG evaluation", unit="question"):
                idx = futures[future]
                results[idx] = future.result()

    return [row for row in results if row is not None]


def summarize_results(results: Sequence[Dict[str, Any]], model: str) -> Dict[str, Any]:
    total = len(results)
    correct = sum(1 for row in results if row.get("correct"))
    group_counts = Counter(row.get("prediction_group", "invalid") for row in results)
    label_counts = Counter(row.get("prediction", "INVALID") for row in results)

    summary = {
        "model": model,
        "total_questions": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "prediction_group_counts": {group: group_counts.get(group, 0) for group in GROUP_ORDER},
        "prediction_group_percentages": {
            group: (group_counts.get(group, 0) / total if total else 0.0) for group in GROUP_ORDER
        },
        "prediction_label_counts": dict(label_counts),
    }
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n================ RAG EVALUATION SUMMARY ================")
    print(f"Model: {summary['model']}")
    print(f"Total questions: {summary['total_questions']}")
    print(f"Correct: {summary['correct']}")
    print(f"Accuracy: {summary['accuracy']:.4f} ({summary['accuracy'] * 100:.2f}%)")
    print("\nPrediction distribution:")
    for group in GROUP_ORDER:
        count = summary["prediction_group_counts"][group]
        pct = summary["prediction_group_percentages"][group] * 100
        print(f"  {group:18s}: {count:6d}  {pct:6.2f}%")
    print("========================================================\n")


def write_outputs(
    results: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
    output_path: Path,
    summary_path: Path,
    csv_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "count", "percentage"])
        writer.writerow(["accuracy", summary["correct"], f"{summary['accuracy']:.6f}"])
        for group in GROUP_ORDER:
            writer.writerow(
                [
                    group,
                    summary["prediction_group_counts"][group],
                    f"{summary['prediction_group_percentages'][group]:.6f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ToolUniverse RAG evaluation on guideline QA data.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=str, default="azure-gpt-4.1", help="Azure OpenAI deployment alias.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Optional small-run limit for debugging.")
    parser.add_argument("--max-steps", type=int, default=40, help="Maximum tool-calling turns per question.")
    parser.add_argument("--max-workers", type=int, default=4, help="Number of parallel question workers. Use 1 for serial execution.")
    args = parser.parse_args()

    questions = load_questions(args.data_path)
    if not questions:
        raise RuntimeError(f"No questions loaded from {args.data_path}")

    results = evaluate_questions(
        questions=questions,
        model=args.model,
        limit=args.limit,
        max_steps=args.max_steps,
        max_workers=args.max_workers,
    )
    summary = summarize_results(results, args.model)
    print_summary(summary)
    write_outputs(results, summary, args.output_path, args.summary_path, args.csv_path)

    print(f"Saved detailed results: {args.output_path}")
    print(f"Saved summary JSON: {args.summary_path}")
    print(f"Saved summary CSV: {args.csv_path}")


if __name__ == "__main__":
    main()
