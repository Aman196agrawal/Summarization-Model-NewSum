# NewsSumm Phase-1 Data Pipeline

## Dataset Overview

The original dataset was provided in Excel format and converted to CSV.
Each row represents a single news article from an Indian news source.

## Cluster Construction

Articles were grouped using:

```
cluster_id = news_category + published_date
```

Only clusters with two or more articles were retained to ensure multi-document
summarization is meaningful.

## Text Cleaning

- HTML tags removed
- URLs removed
- Extra whitespace normalized

## Final Dataset Format

Each sample in the JSONL file contains:
- `cluster_id` — unique identifier for the news cluster
- `documents` — list of source article texts (one string per article)
- `summary` — human-written reference summary

Example:
```json
{
  "cluster_id": "politics_2023-10-01",
  "documents": ["Article 1 text...", "Article 2 text...", "Article 3 text..."],
  "summary": "Reference summary text."
}
```

## Sentence Segmentation Convention

Before feeding clusters to the model, documents are joined with a `[SEP]`
delimiter token using Python's standard `str.join()` — no external NLP library
is required:

```python
input_text = " [SEP] ".join(sample["documents"])
```

This approach:
- Marks document boundaries so the encoder can learn cross-document structure
- Requires no tokenization dependencies (spaCy, NLTK, etc.)
- Is compatible with LongT5's tokenizer (the `[SEP]` string is encoded as
  separate subword tokens, serving as a soft boundary signal)

The `[SEP]` delimiter convention is used consistently in both:
- `phase4/train_novel_model.py` (training)
- `scripts/generate_novel_test_predictions.py` (inference)

## How Clusters Are Fed to the Model

1. Load a JSONL sample with `NewsSummDataset`
2. Join documents with `" [SEP] ".join(sample["documents"])`
3. Tokenize the joined string with `AutoTokenizer`, truncating to 4096 tokens
4. Pass `input_ids` and `attention_mask` to `SalienceAwareLongT5.forward()`
5. The encoder learns to treat `[SEP]` positions as document boundary markers

## Output Files

- `data/newssumm_phase1.jsonl` — raw clustered data
- `data/splits/train.jsonl` — 90% training split
- `data/splits/val.jsonl` — 10% validation split
- `data/splits/test.jsonl` — held-out test split

