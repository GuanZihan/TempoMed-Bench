models=(
    # "meta-llama/Llama-2-7b-chat-hf"
    # "meta-llama/Llama-2-13b-chat-hf"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "google/gemma-3-4b-it"
    # "google/medgemma-4b-it"
    # "openai/gpt-oss-20b"
    # "Qwen/Qwen3-0.6B"
    # "Qwen/Qwen3-1.7B"
    # "Qwen/Qwen3-4B"
    "azure-gpt-4.1"
    "azure-gpt-4o"
    "azure-gpt-5"
)

for model_name in "${models[@]}"; do
    model_slug="${model_name//\//_}"
    choice_result_json="pubmed_trajectory/results/evaluation_results_5_option_${model_slug}_augmented.json"
    judge_result_json="pubmed_trajectory/results/evaluation_results_judge_5_option_${model_slug}_augmented.json"
    choice_cutoff_prefix="pubmed_trajectory/results/cutoff_5_option_${model_slug}_augmented"
    judge_cutoff_prefix="pubmed_trajectory/results/cutoff_judge_5_option_${model_slug}_augmented"

    python pubmed_trajectory/evaluation/pubmed_evaluation_5_option.py \
        --model_name "$model_name" \
        --mode choices \
        --choice_shuffle_method all \
        --choice_output_path "$choice_result_json" \
        --judge_output_path "$judge_result_json" \
        --data_path "questions_2026_relaxed_4_option_augmented_verified.jsonl" \
        --question_variant_mode with_counterpart

    python pubmed_trajectory/pubmed_visualize_5_option.py \
        --input "$choice_result_json" \
        --output pubmed_trajectory/results/visualize_5_option_${model_slug}_augmented.png

done
