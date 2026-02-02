# NewsSumm Phase-1 Data Pipeline

## Dataset Overview
The original dataset was provided in Excel format and converted to CSV.
Each row represents a single news article.

## Cluster Construction
Articles were grouped using:
cluster_id = news_category + published_date

Only clusters with two or more articles were retained.

## Text Cleaning
- HTML tags removed
- URLs removed
- Extra whitespace normalized

## Final Dataset Format
Each sample contains:
- cluster_id
- list of source documents
- human reference summary

## Output
- newssumm_phase1.jsonl
- newssumm_dataset.py
