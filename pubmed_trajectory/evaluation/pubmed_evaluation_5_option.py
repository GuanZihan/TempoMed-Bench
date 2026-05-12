
import argparse
import json
import re
import random
import sys
import itertools
from pathlib import Path
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from openai import BadRequestError

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
sys.path.append(str(PROJECT_ROOT))
from utils.model_utils import get_response_with_vllm
from utils import config
import asyncio

PUBLIC_AZURE_MODEL_ATTRS = {
    "azure-gpt-4.1": ("AZURE_GPT41_DEPLOYMENT", "AZURE_DEPLOYMENT"),
    "azure-gpt-4o": ("AZURE_GPT4O_DEPLOYMENT", "AZURE_DEPLOYMENT"),
    "azure-gpt-5": ("AZURE_GPT5_DEPLOYMENT", "AZURE_DEPLOYMENT"),
}


def resolve_azure_deployment(model_name):
    if model_name in PUBLIC_AZURE_MODEL_ATTRS:
        for attr in PUBLIC_AZURE_MODEL_ATTRS[model_name]:
            value = getattr(config, attr, None)
            if value:
                return value
        raise ValueError(f"No configured deployment found for alias {model_name}.")
    return model_name


def is_public_azure_model(model_name):
    return model_name in PUBLIC_AZURE_MODEL_ATTRS


def azure_temperature(model_name):
    return 1 if model_name == "azure-gpt-5" else 0

# --------------------------
# Load dataset
# --------------------------
ANSWER_LABELS = ("A", "B", "C", "D")
UNKNOWN_LABEL = "E"
DEFAULT_LABELS = ANSWER_LABELS + (UNKNOWN_LABEL,)


def parse_labels(labels_csv):
    labels = tuple(x.strip().upper() for x in labels_csv.split(",") if x.strip())
    if len(labels) != 5 or len(set(labels)) != 5:
        raise ValueError("choice_alt_labels must contain exactly 5 unique labels.")
    return labels


def _select_permutations(all_permutations, shuffle_copies):
    if shuffle_copies <= len(all_permutations):
        return random.sample(all_permutations, shuffle_copies)
    return [random.choice(all_permutations) for _ in range(shuffle_copies)]


def build_counterpart_question(question, year_current, year_prior):
    updated_question, replacements = re.subn(
        rf"(issued in ){re.escape(str(year_current))}(\b)",
        rf"\g<1>{year_prior}\2",
        question,
        count=1,
    )
    if replacements:
        return updated_question

    fallback_pattern = rf"\b{re.escape(str(year_current))}\b"
    return re.sub(fallback_pattern, str(year_prior), question, count=1)


def build_question_variants(data, question_variant_mode):
    variant_items = []

    for item in data:
        current_correct = item["Answer"]["Correct"].strip().upper()

        current_item = dict(item)
        current_item["QuestionVariantType"] = "current_guideline"
        current_item["QuestionTargetYear"] = item["Year_current"]
        current_item["QuestionVariantCorrect"] = current_correct
        current_item["OppositeOriginalOption"] = "B"
        variant_items.append(current_item)

        if question_variant_mode == "with_counterpart":
            counterpart_item = dict(item)
            counterpart_item["Question"] = build_counterpart_question(
                item["Question"],
                item["Year_current"],
                item["Year_prior"],
            )
            counterpart_item["QuestionVariantType"] = "prior_guideline"
            counterpart_item["QuestionTargetYear"] = item["Year_prior"]
            counterpart_item["QuestionVariantCorrect"] = "B"
            counterpart_item["OppositeOriginalOption"] = current_correct
            variant_items.append(counterpart_item)

    return variant_items


def build_choice_mode_data(data, shuffle_copies=4, shuffle_method="shuffle_labels", alt_labels=DEFAULT_LABELS):
    if shuffle_copies <= 0:
        return []

    # Keep the abstain option fixed as E and only shuffle the substantive choices.
    permutations = list(itertools.permutations(range(4)))
    augmented = []

    for item in data:
        answer = item["Answer"]
        original_correct = item.get("QuestionVariantCorrect", answer["Correct"]).strip().upper()
        choices_by_default = (
            answer["Choice_A"],
            answer["Choice_B"],
            answer["Choice_C"],
            answer["Choice_D"],
            answer["Choice_E"],
        )
        correct_idx = ANSWER_LABELS.index(original_correct)
        unknown_text = choices_by_default[4]

        selected_perms = _select_permutations(permutations, shuffle_copies)

        for perm in selected_perms:
            if shuffle_method == "reorder_only":
                option_entries = [(ANSWER_LABELS[i], choices_by_default[i]) for i in perm]
                option_entries.append((UNKNOWN_LABEL, unknown_text))
                gold_label = original_correct
                display_to_original = {ANSWER_LABELS[i]: ANSWER_LABELS[i] for i in perm}
                display_to_original[UNKNOWN_LABEL] = UNKNOWN_LABEL
            elif shuffle_method == "shuffle_labels":
                mapping = [(ANSWER_LABELS[i], choices_by_default[perm[i]]) for i in range(4)]
                option_entries = mapping + [(UNKNOWN_LABEL, unknown_text)]
                gold_label = ANSWER_LABELS[perm.index(correct_idx)]
                display_to_original = {ANSWER_LABELS[i]: ANSWER_LABELS[perm[i]] for i in range(4)}
                display_to_original[UNKNOWN_LABEL] = UNKNOWN_LABEL
            elif shuffle_method == "alternate_labels":
                option_entries = [(alt_labels[i], choices_by_default[i]) for i in range(5)]
                gold_label = alt_labels[correct_idx]
                display_to_original = {alt_labels[i]: DEFAULT_LABELS[i] for i in range(5)}
            else:
                raise ValueError(f"Unsupported shuffle method: {shuffle_method}")

            original_to_display = {original: display for display, original in display_to_original.items()}

            shuffled_item = {
                "Year_prior": item["Year_prior"],
                "Year_current": item["Year_current"],
                "Question": item["Question"],
                "ChoiceModeOptions": option_entries,
                "ChoiceModeGold": gold_label,
                "ChoiceShuffleMethod": shuffle_method,
                "DisplayToOriginalOption": display_to_original,
                "OriginalToDisplayOption": original_to_display,
                "CurrentGuidelineOption": original_to_display["A"],
                "PriorGuidelineOption": original_to_display["B"],
                "InterferenceOptions": [original_to_display["C"], original_to_display["D"]],
                "UnknownOption": original_to_display["E"],
                "QuestionVariantType": item.get("QuestionVariantType", "current_guideline"),
                "QuestionTargetYear": item.get("QuestionTargetYear", item["Year_current"]),
            }
            if "idx" in item:
                shuffled_item["idx"] = item["idx"]
            augmented.append(shuffled_item)

    return augmented


def assign_variant_sub_indices(items):
    variant_counts = {}

    for item in items:
        group_key = ("idx", item["idx"]) if "idx" in item else ("question", item["Question"])
        sub_idx = variant_counts.get(group_key, 0)
        item["variant_sub_idx"] = sub_idx
        variant_counts[group_key] = sub_idx + 1

    return items


def load_data(data_path):
    if data_path.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as f:
            blocks = f.read().strip().split("\n\n")
        return [json.loads(block) for block in blocks if block.strip()]

    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------
# Prompt Template
# --------------------------

QUESTION_ONLY_TEMPLATE = """
Question:
{question}

Provide the best possible answer.
"""

JUDGE_TEMPLATE = """
You are a strict judge who determines whether the student’s answer is sufficient to infer the reference answer. I also provide an example of an incorrect answer, which corresponds to an outdated version.

Question:
{question}

Reference answer:
{reference_answer}

Wrong answer:
{opposite_answer}

Student answer:
{student_answer}

Respond with the following Format:
##isInfer: [Yes/No]
##Reason: [Your Reason Here]

Note that:
- Respond ##isInfer with YES if the student's answer is enough to infer the reference answer, otherwise respond with NO.
- ##Reason records your reasons on why the student answer can or cannot infer the referece answer.
"""

def build_azure_judge():
    return AzureChatOpenAI(
        azure_deployment=resolve_azure_deployment("azure-gpt-4.1"),
        api_key=config.AZURE_OPENAI_API_KEY,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_version=config.AZURE_OPENAI_API_VERSION,
        temperature=0,
    )

FALLBACK_ANSWER = "INVALID"
async def _parallel_azure_query(azure_llm, prompts, system_prompt, max_concurrency=6):
    sem = asyncio.Semaphore(max_concurrency)

    async def one(i, p):
        async with sem:
            try:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=p)
                ]
                resp = await azure_llm.ainvoke(messages)
                out = resp.content if hasattr(resp, "content") else str(resp)
                return i, out
            except BadRequestError as e:
                print(f"[WARN] Prompt {i} rejected, using fallback. Reason: {e}")
                return i, FALLBACK_ANSWER
            except Exception as e:
                print(f"[ERROR] Prompt {i} failed: {e}, using fallback.")
                return i, FALLBACK_ANSWER

    tasks = [asyncio.create_task(one(i, p)) for i, p in enumerate(prompts)]
    results = [None] * len(prompts)

    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Azure parallel"):
        i, res = await fut
        results[i] = res

    return results

def parallel_azure_query(azure_llm, prompts, system_prompt, max_concurrency=6):
    return asyncio.run(_parallel_azure_query(azure_llm, prompts, system_prompt, max_concurrency))

def parse_yes_no(text):
    if not text or not text.strip():
        return "INVALID"
    
    # Normalize
    cleaned = text.strip()

    # 1. Try strict match: ##isInfer: Yes/No (case-insensitive)
    match = re.search(r"##\s*isinfer\s*:\s*(yes|no)", cleaned, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 2. Fallback: line-based detection
    for line in cleaned.splitlines():
        line_clean = line.strip().upper()
        if line_clean.startswith("YES"):
            return "YES"
        if line_clean.startswith("NO"):
            return "NO"
        if "YES" == line_clean:
            return "YES"
        if "NO" == line_clean:
            return "NO"

    # 3. Fallback: loose search
    if re.search(r"\bYES\b", cleaned, re.IGNORECASE):
        return "YES"
    if re.search(r"\bNO\b", cleaned, re.IGNORECASE):
        return "NO"

    return "INVALID"

def build_choice_prompt(question, option_entries):
    label_text = "/".join(label for label, _ in option_entries)
    option_block = "\n".join(f"{label}) {text}" for label, text in option_entries)
    return f"""
Question:
{question}

{option_block}

# You MUST Not generated any fabricated evidence or hallucinated words (e.g., the guideline would likely xxx)! If you do know the answer, simply choose the option of 'I do not know the answer' ({UNKNOWN_LABEL}).
# Please answer with the following format:
##Explanation: [Your Explanations Here]
##Answer: [Your Choice Here ({label_text})]
"""

def run_choice_mode(data, model_name, output_path):
    records = []
    prompts = []

    for item in data:
        option_entries = item["ChoiceModeOptions"]
        prompt = build_choice_prompt(
            question=item["Question"],
            option_entries=option_entries
        )

        prompts.append(prompt)

        choices_dict = {label: text for label, text in option_entries}
        base_record = {
            "year_previous": item["Year_prior"],
            "year_current": item["Year_current"],
            "target_guideline_year": item["QuestionTargetYear"],
            "question_variant": item["QuestionVariantType"],
            "question": item["Question"],
            "choices": choices_dict,
            "option_order": [label for label, _ in option_entries],
            "gold": item["ChoiceModeGold"],
            "shuffle_method": item["ChoiceShuffleMethod"],
            "variant_sub_idx": item["variant_sub_idx"],
            "display_to_original_option": item["DisplayToOriginalOption"],
            "original_to_display_option": item["OriginalToDisplayOption"],
            "current_guideline_option": item["CurrentGuidelineOption"],
            "prior_guideline_option": item["PriorGuidelineOption"],
            "interference_options": item["InterferenceOptions"],
            "unknown_option": item["UnknownOption"],
        }
        if "idx" in item:
            base_record["idx"] = item["idx"]

        records.append(base_record)

    if is_public_azure_model(model_name):
        azure_llm = AzureChatOpenAI(
            azure_deployment=resolve_azure_deployment(model_name),
            api_key=config.AZURE_OPENAI_API_KEY,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_version=config.AZURE_OPENAI_API_VERSION,
            temperature=azure_temperature(model_name),
        )
        # responses = []
        # for p in tqdm(prompts, ascii=' #'):
        #     messages = [
        #         SystemMessage(content="You are a medical QA model. Answer the multiple-choice question."),
        #         HumanMessage(content=p)
        #     ]
        #     resp = azure_llm.invoke(messages)
        #     responses.append(resp.content if hasattr(resp, "content") else str(resp))
        responses = parallel_azure_query(
            azure_llm,
            prompts,
            system_prompt="You are a medical QA model. Answer the multiple-choice question.",
            max_concurrency=6
        )
    else:
        responses = get_response_with_vllm(model_name, prompts)

    results = []
    correct = 0
    invalid = 0

    for r, out in zip(records, responses):
        pred = extract_choice_label(out, r["option_order"])
        pred_original = r["display_to_original_option"].get(pred) if pred != "INVALID" else None

        r["model_answer"] = out
        r["prediction"] = pred
        r["prediction_original_option"] = pred_original
        if pred_original == "A":
            r["prediction_group"] = "current_guideline"
        elif pred_original == "B":
            r["prediction_group"] = "prior_guideline"
        elif pred_original in {"C", "D"}:
            r["prediction_group"] = "interference"
        elif pred_original == "E":
            r["prediction_group"] = "unknown"
        else:
            r["prediction_group"] = "invalid"
        r["correct"] = (pred == r["gold"])

        if pred == "INVALID":
            invalid += 1
        if pred == r["gold"]:
            correct += 1

        results.append(r)

    total = len(results)
    accuracy = correct / total if total else 0

    print("\n================ EVALUATION SUMMARY ================")
    print(f"Model: {model_name}")
    print(f"Total questions: {total}")
    print(f"Correct: {correct}")
    print(f"Invalid outputs: {invalid}")
    print(f"Accuracy: {accuracy:.4f}")
    print("===================================================\n")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {output_path}")

def run_judge_mode(data, model_name, output_path):
    question_only_prompts = []
    judge_records = []

    for item in data:
        question_only_prompts.append(
            QUESTION_ONLY_TEMPLATE.format(
                question=item["Question"]
            )
        )
        judge_records.append({
            "year_previous": item["Year_prior"],
            "year_current": item["Year_current"],
            "target_guideline_year": item.get("QuestionTargetYear", item["Year_current"]),
            "question_variant": item.get("QuestionVariantType", "current_guideline"),
            "question": item["Question"],
            "choice_A": item["Answer"]["Choice_A"],
            "choice_B": item["Answer"]["Choice_B"],
            "choice_C": item["Answer"]["Choice_C"],
            "choice_D": item["Answer"]["Choice_D"],
            "choice_E": item["Answer"]["Choice_E"],
            "gold_answer": item["Answer"][f"Choice_{item.get('QuestionVariantCorrect', item['Answer']['Correct'])}"],
            "opposite_answer": item["Answer"][f"Choice_{item.get('OppositeOriginalOption', 'B')}"],
            **({"variant_sub_idx": item["variant_sub_idx"]} if "variant_sub_idx" in item else {}),
            **({"idx": item["idx"]} if "idx" in item else {})
        })


    if is_public_azure_model(model_name):
        azure_llm = AzureChatOpenAI(
            azure_deployment=resolve_azure_deployment(model_name),
            api_key=config.AZURE_OPENAI_API_KEY,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_version=config.AZURE_OPENAI_API_VERSION,
            temperature=azure_temperature(model_name),
            max_retries=5,
        )
        free_form_answers = parallel_azure_query(
            azure_llm,
            question_only_prompts,
            system_prompt="You are a medical QA model. Answer the multiple-choice question.",
            max_concurrency=6
        )
    else:
        free_form_answers = get_response_with_vllm(
            model_name,
            question_only_prompts,
            system_prompt="You are a medical QA model. Answer the question directly and concisely."
        )

    

    judge_prompts = []
    for r, ans in zip(judge_records, free_form_answers):
        r["model_answer"] = ans
        judge_prompts.append(
            JUDGE_TEMPLATE.format(
                question=r["question"],
                reference_answer=r["gold_answer"],
                opposite_answer=r["opposite_answer"],
                student_answer=ans
            )
        )

    judge_llm = build_azure_judge()

    judge_outputs = parallel_azure_query(
        judge_llm,
        judge_prompts,
        system_prompt="You are a strict evaluation model.",
        max_concurrency=6
    )
    judge_correct = 0
    judge_invalid = 0
    judge_results = []

    for r, decision in zip(judge_records, judge_outputs):
        verdict = parse_yes_no(decision)
        r["judge_response"] = decision
        r["judge_decision"] = verdict
        r["correct"] = verdict == "YES"

        if verdict == "INVALID":
            judge_invalid += 1
        if verdict == "YES":
            judge_correct += 1

        judge_results.append(r)

    total_judge = len(judge_results)
    judge_accuracy = judge_correct / total_judge if total_judge else 0

    print("\n================ JUDGE EVALUATION SUMMARY ================")
    print(f"Generator model: {model_name}")
    print(f"Judge model: Azure GPT ({config.AZURE_DEPLOYMENT})")
    print(f"Total questions: {total_judge}")
    print(f"Judge marked correct: {judge_correct}")
    print(f"Invalid judge outputs: {judge_invalid}")
    print(f"Accuracy: {judge_accuracy:.4f}")
    print("=========================================================\n")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(judge_results, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {output_path}")

def extract_choice_label(text, valid_labels):
    """
    Extract answer from format:
    ##Answer: <LABEL>
    """
    if not text:
        return "INVALID"

    normalized_labels = tuple(label.upper() for label in valid_labels)
    valid_set = set(normalized_labels)
    token_pattern = "|".join(re.escape(label) for label in sorted(valid_set, key=len, reverse=True))

    text = text.strip()

    match = re.search(rf"##\s*Answer\s*:\s*({token_pattern})\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    first_line = text.splitlines()[0].strip().upper()
    if first_line in valid_set:
        return first_line

    tokens = re.findall(rf"\b({token_pattern})\b", text.upper())
    if len(tokens) == 1:
        return tokens[0].upper()

    return "INVALID"

def main():
    parser = argparse.ArgumentParser(description="Evaluate guideline QA models.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="questions_2026_relaxed_5_option_with_example_version6_indexed_variant1.jsonl",
        help="Path to the input questions file (.json or .jsonl)."
    )
    parser.add_argument(
        "--question_variant_mode",
        choices=["current_only", "with_counterpart"],
        default="with_counterpart",
        help="Whether to evaluate only the original current-guideline question or also add the prior-guideline counterpart."
    )
    parser.add_argument(
        "--mode",
        choices=["choices", "judge", "both"],
        default="choices",
        help="Run traditional multiple-choice evaluation, judge evaluation, or both."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=""
    )
    parser.add_argument(
        "--choice_shuffle_copies",
        type=int,
        default=1,
        help="Number of shuffled variants to generate per question in choices mode."
    )
    parser.add_argument(
        "--choice_shuffle_method",
        choices=["reorder_only", "shuffle_labels", "alternate_labels", "all"],
        default="shuffle_labels",
        help="How to generate shuffled variants for choices mode."
    )
    parser.add_argument(
        "--choice_alt_labels",
        type=str,
        default="A,B,C,D,E",
        help="Comma-separated labels for alternate_labels mode."
    )
    parser.add_argument(
        "--choice_output_path",
        type=str,
        default="pubmed_trajectory/evaluation_results.json",
        help="Path to save choices-mode evaluation results JSON."
    )
    parser.add_argument(
        "--judge_output_path",
        type=str,
        default="pubmed_trajectory/evaluation_results_judge.json",
        help="Path to save judge-mode evaluation results JSON."
    )
    args = parser.parse_args()
    data = load_data(args.data_path)
    question_variant_data = build_question_variants(data, args.question_variant_mode)
    judge_data = assign_variant_sub_indices([dict(item) for item in question_variant_data])

    alt_labels = DEFAULT_LABELS
    if args.choice_shuffle_method in ("alternate_labels", "all"):
        alt_labels = parse_labels(args.choice_alt_labels)

    choice_data = []
    if args.choice_shuffle_method == "all":
        choice_data.extend(build_choice_mode_data(question_variant_data, args.choice_shuffle_copies, "reorder_only", alt_labels))
        choice_data.extend(build_choice_mode_data(question_variant_data, args.choice_shuffle_copies, "shuffle_labels", alt_labels))
        choice_data.extend(build_choice_mode_data(question_variant_data, args.choice_shuffle_copies, "alternate_labels", alt_labels))
    else:
        choice_data = build_choice_mode_data(
            question_variant_data,
            args.choice_shuffle_copies,
            args.choice_shuffle_method,
            alt_labels
        )
    choice_data = assign_variant_sub_indices(choice_data)

    if args.mode in ("choices", "both"):
        run_choice_mode(choice_data, args.model_name, args.choice_output_path)

    if args.mode in ("judge", "both"):
        run_judge_mode(judge_data, args.model_name, args.judge_output_path)

if __name__ == "__main__":
    main()
