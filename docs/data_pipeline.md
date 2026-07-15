# NewsSumm++ Data Pipeline

This describes the pipeline implemented in `scripts/build_newssumm_pp.py`,
ported from the author's working Colab notebook
(`more_about_dataset_filtering.ipynb`). It replaces the earlier, simpler
"Phase 1" pipeline (category+date grouping only, no deduplication or
clustering), which is now superseded.

## Input

A raw CSV with at least these columns: `article_text`, `human_summary`,
`headline`, `news_category`, `newspaper_name`, `published_date`.

## Steps

1. **Null/empty removal** (`remove_nulls_and_duplicates`): drop rows with a
   missing or empty `article_text`/`human_summary`.
2. **Exact duplicate removal**: drop rows with an identical `article_text`.
3. **Text normalization** (`clean_text`): lowercase, strip non-alphanumeric
   characters, collapse whitespace.
4. **Near-duplicate filtering** (`filter_near_duplicates`): TF-IDF vectorize
   `clean_article`, drop rows whose cosine similarity to any other row
   exceeds 0.90.
5. **Semantic clustering** (`cluster_documents`): embed `headline + clean_article`
   with SentenceTransformer `all-MiniLM-L6-v2`, cluster with `MiniBatchKMeans`.
6. **Cluster-size filtering** (`filter_cluster_size`): keep only clusters with
   7-10 documents, matching the report's per-cluster document count.
7. **Category mapping** (`map_category`): map the original free-text
   `news_category` values onto the 22 fixed categories in
   `scripts.build_newssumm_pp.FIXED_CATEGORIES`; rows that don't match any
   keyword are dropped.

## Output

A CSV with the original columns plus `clean_article`, `clean_summary`,
`cluster_id`, `mapped_category` -- one row per source article, grouped into
event clusters via `cluster_id`.

## Reproducing the exact numbers in the report

The report's Table I figures (11,557 samples, 1,391 clusters, silhouette
score 0.1443, etc.) come from running this pipeline on the full 348K-row raw
NewsSumm dataset in Google Colab (which has the RAM/GPU this pipeline needs
for large-scale embedding and clustering). This repository's own copy of the
pipeline (`scripts/build_newssumm_pp.py`) is unit-tested on small synthetic
inputs (see `tests/test_build_newssumm_pp.py`) to confirm the *logic* is
correct, but has not been re-run end-to-end on the full 348K-row dataset from
this environment -- there's no GPU/large dataset available here. Running
`python scripts/build_newssumm_pp.py` against the real raw CSV will
reproduce the pipeline; the exact resulting counts may differ slightly from
Table I depending on library versions and embedding model updates.
