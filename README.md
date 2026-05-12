# TempoMed-Bench

We built TempoMed-Bench, a benchmark for evaluating the temporal awareness of LLMs in the medical domain through evolving guideline knowledge.

At a high level, the pipeline:

1. collects PubMed and PMC source documents,
2. identifies guideline-like papers,
3. builds current-to-prior guideline trajectories,
4. extracts recommendation differences between guideline versions,
5. converts those differences into multiple-choice questions, and
6. evaluates models on those questions.

The main code is organized under:

- `pubmed_trajectory/prepare/`
- `pubmed_trajectory/post-processing/`
- `pubmed_trajectory/difference_generation/`
- `pubmed_trajectory/evaluation/`

Several scripts require a `utils/config.py` (You have to create your own by referring to the example we provide in `utils/config_example.py`) for Azure/OpenAI credentials.

## Demo Run
For reviewers, the simplest way to inspect the provided benchmark outputs is:

```bash
bash evaluate_5_option.sh
```

This uses the checked-in result files and runs the downstream visualization/analysis step.

If you want to rerun the Azure-based 5-option evaluation itself, use:

```bash
bash evaluate_5_option_gpt.sh
```

This requires a local untracked `utils/config.py` with valid Azure/OpenAI credentials.

## Dataset Preparation
The following describes how to replicate our data collection process:

### 1. Download PubMed update files

```bash
python3 pubmed_trajectory/prepare/pubmed_download.py
```

You need to download all the subsets, including `comm`, `noncomm`, and `other`. Also you need to download pubmed `baseline` and `updatefiles`.

### 2. Build the guideline lookup

```bash
python3 pubmed_trajectory/prepare/pubmed_meta_graph.py
```

This expects extracted PubMed XML directories such as:

- `pubmed_baseline_xml_extracted_2026`
- `pubmed_updatefiles_xml_extracted_2026`

and writes `pmid_guideline_mapping.json`.

### 3. Extract guideline-like PMC articles

```bash
python3 pubmed_trajectory/prepare/pubmed_extract.py
```

This stage is mainly configured for the `comm` branch by default. If you also use `noncomm` or `other`, you may need to repeat or adapt the same process.

### 4. Build prior-guideline trajectories

```bash
python3 pubmed_trajectory/prepare/pubmed_extract_guideline_groups.py
```

This produces trajectory JSON files such as:

- `comm_guideline_trajectory_2026_relaxed/...`

### 5. Optional post-processing

Useful cleanup and augmentation scripts live in `pubmed_trajectory/post-processing/`, including:

- `pubmed_augment_trajectory_with_related.py`
- `pubmed_apply_resolved_current_pmids.py`
- `pubmed_calibrate_year_from_title.py`
- `pubmed_filter_nonterminal_trajectories.py`
- `pubmed_check_redundant_prior_pmids.py`

This process finalizes the TempoMed-Traj dataset.

### 6. Extract recommendation differences

```bash
python3 pubmed_trajectory/difference_generation/pubmed_construct_guideline_diffs_with_verifier.py
```

This writes nested outputs under directories such as:

- `results_2026_relaxed_with_post_processing/<current_pmcid>/...`

### 7. Flatten non-empty diff files

Question generation expects a flat directory of diff JSON files rather than nested per-PMCID outputs.

For reviewers, use the following directory as the canonical flattened input for MCQ generation:

- `results_2026_relaxed_with_post_processing_flat`

If you need to regenerate it from the nested post-processed outputs, run:

```bash
python3 pubmed_trajectory/difference_generation/copy_flat_json_files.py \
  --source-dir ./results_2026_relaxed_with_post_processing \
  --output-dir ./results_2026_relaxed_with_post_processing_flat
```

### 8. Generate MCQs from recommendation differences

After flattening the non-empty recommendation-difference files, you can generate MCQs with:

```bash
bash generate_questions_4_option.sh
```

This wrapper currently runs:

```bash
python pubmed_trajectory/evaluation/pubmed_generate_questions_with_NBME.py \
  ./results_2026_relaxed_with_post_processing_flat \
  ./questions_2026_relaxed_4_option_augmented.jsonl
```

The input should be a flat directory containing `*_extracted_diffs.json` files. The output is a JSONL file of MCQs, where each item includes the question stem, answer choices, the correct answer, and metadata linking the question back to the current and prior guideline pair.
