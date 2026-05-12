from datetime import date
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import torch

model_to_date = {
    "Qwen2.5": date(2024, 12, 19),
    "Llama3-8b": date(2023, 12, 31),
    "Llama3-70b": date(2023, 12, 31),
    'Ministral-2410': date(2024, 10, 30),
    'Ministral-2506': date(2025, 6, 30),
    'medgemma-4b': date(2025, 5, 20),
    'medgemma-27b': date(2025, 5, 20)
}

model_name_to_allias = {
    'Qwen/Qwen2.5-7B-Instruct': 'Qwen2.5',
    'Qwen/Qwen2.5-14B-Instruct': 'Qwen2.5',
    'meta-llama/Llama-3.1-8B-Instruct': 'Llama3-8b',
    'meta-llama/Llama-3.1-70B-Instruct': 'Llama3-70b',
    'mistralai/Ministral-8B-Instruct-2410': 'Ministral-2410',
    'google/medgemma-4b-it': 'medgemma-4b',
    'google/medgemma-27b-it': 'medgemma-27b',
    'google/medgemma-1.5-4b-it': 'medgemma-1.5-4b'
}

def get_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, device="auto")
    model = AutoModelForCausalLM.from_pretrained(model_name, device="auto")
    return model

def get_response_from_model(model, query):
    messages = [
        {"role": "user", "content": query},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=1024)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    return response

def get_response_with_vllm(model_name, prompts, system_prompt=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    llm = LLM(
        model=model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype="bfloat16",
        gpu_memory_utilization=0.92,
        trust_remote_code=True
    )

    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.9,
        max_tokens=4096
    )

    system_prompt = system_prompt or "You are a medical QA model. Answer the multiple-choice question."

    # Apply chat template if needed
    formatted_prompts = []
    for p in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": p}
        ]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        formatted_prompts.append(formatted)

    outputs = llm.generate(formatted_prompts, sampling_params)
    return [o.outputs[0].text.strip() for o in outputs]


if __name__ == "__main__":
    
    prompts = [
        "Explain the difference between RAG and fine-tuning.",
        "Write a short poem about AI in healthcare."
    ]

    outputs = get_response_with_vllm("Qwen/Qwen2.5-7B-Instruct", prompts)
    print(outputs)
