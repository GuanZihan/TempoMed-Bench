# Post-Processing Scripts

This directory contains standalone cleanup and filtering utilities for the augmented guideline trajectory JSON files.

These scripts are intended to be run after the base trajectory extraction / augmentation pipeline.

## Scripts

### `pubmed_augment_trajectory_with_related.py`
Augments existing trajectory JSON files with candidate prior guidelines retrieved from PubMed related-work APIs and verified by an Azure LLM.

Main features:
- scans trajectory JSON files under an input tree
- queries PubMed related papers for each current guideline PMID
- fetches candidate titles, abstracts, affiliations, and years
- uses Azure + LangGraph structured output to decide whether a candidate is a prior guideline
- writes an augmented mirrored output tree
- supports parallel processing and break-and-continue

Example:
```bash
python3 pubmed_trajectory/post-processing/pubmed_augment_trajectory_with_related.py \
  --input-dir ./comm_guideline_trajectory_2026_relaxed \
  --output-dir ./comm_guideline_trajectory_2026_relaxed_augmented \
  --max-workers 4
```

### `pubmed_apply_resolved_current_pmids.py`
Fixes `PMID = 0` values in augmented trajectories.

Main features:
- first applies `augmentation_metadata.current_pmid_resolution.resolved_pmid` to the current guideline when available
- then searches PubMed by title for any remaining `PMID = 0`
- applies to both the current guideline title and prior guideline titles
- only keeps the queried PMID if the PubMed title matches exactly after normalization

Dry run:
```bash
python3 pubmed_trajectory/post-processing/pubmed_apply_resolved_current_pmids.py --dry-run
```

Apply changes:
```bash
python3 pubmed_trajectory/post-processing/pubmed_apply_resolved_current_pmids.py
```

### `pubmed_calibrate_year_from_title.py`
Calibrates guideline years from the year text that appears in titles.

Main features:
- if the current `Title` contains a 4-digit year, updates `year_of_current_guidance`
- if a prior guideline `title` contains a 4-digit year, updates that prior entry's `year`

Dry run:
```bash
python3 pubmed_trajectory/post-processing/pubmed_calibrate_year_from_title.py --dry-run
```

Apply changes:
```bash
python3 pubmed_trajectory/post-processing/pubmed_calibrate_year_from_title.py
```

### `pubmed_check_redundant_prior_pmids.py`
Checks whether any file's `prior_guidelines` contains duplicate nonzero PMIDs.

Main features:
- ignores `0`, `"0"`, empty, and `None`
- prints matching file paths directly to stdout
- writes a JSON report with duplicate PMID indices

Example:
```bash
python3 pubmed_trajectory/post-processing/pubmed_check_redundant_prior_pmids.py
```

### `pubmed_filter_nonterminal_trajectories.py`
Finds trajectories whose current PMID already appears as a prior PMID in a newer trajectory.

Main features:
- scans all trajectory JSON files under an input tree
- flags files whose current guideline is already a prior node in a more recent trajectory
- writes a report JSON
- optionally writes a filtered copy of the tree without those flagged files

Report only:
```bash
python3 pubmed_trajectory/post-processing/pubmed_filter_nonterminal_trajectories.py \
  --input-dir ./comm_guideline_trajectory_2026_relaxed_augmented
```

Write filtered copy:
```bash
python3 pubmed_trajectory/post-processing/pubmed_filter_nonterminal_trajectories.py \
  --input-dir ./comm_guideline_trajectory_2026_relaxed_augmented \
  --output-dir ./comm_guideline_trajectory_2026_relaxed_augmented_filtered
```

## Notes

- Most scripts default to the `comm_guideline_trajectory_2026_relaxed_augmented` tree. Override `--target-dir` or `--input-dir` when running on `noncomm` or `other` trees.
- These scripts modify JSON files in place unless they explicitly produce a copied output tree.
- Use `--dry-run` where available before applying writes.
- All scripts are standalone and can be run independently.
