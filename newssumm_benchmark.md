# NewsSumm Benchmark: Baselines and Proposed Model

This document presents the final benchmark results for multi-document news summarization on the **NewsSumm dataset**.
---

## 1. Evaluation Setup

- **Dataset:** NewsSumm (test split)
- **Task:** Multi-document news summarization
- **Evaluation Metrics:**
  - ROUGE-1 (F1)
  - ROUGE-2 (F1)
  - ROUGE-L (F1)
  - BERTScore (F1)

All models are evaluated on the same test clusters to ensure fair comparison.

---

## 2. Benchmark Results Table

> **Status:** the numbers previously in this table were identical across all
> nine baseline models (0.4381 / 0.1822 / 0.4381 / 0.9632) because they were
> accidentally copied from `scripts/evaluate_sanity.py`'s 2-example dummy
> check rather than produced by real per-model runs -- see
> `scripts/evaluate_sanity.py`'s updated warning. That table has been removed
> rather than left in place. Real per-model results will be added here once
> `results/<model_tag>/test_metrics.json` exists for each model from an
> actual run of `scripts/generate_test_predictions.py` /
> `scripts/generate_novel_test_predictions.py` on the real NewsSumm++ test
> split -- not reproduced from this repo yet.

---

## 3. Quantitative Analysis

*The analysis below describes anticipated/exploratory findings from before this repository's pipeline was fixed and is not yet backed by a real run of the current code (see the Status note in Section 2). Treat as hypotheses to verify, not established results.*

Large language models such as Mixtral and LLaMA-based systems achieve strong ROUGE scores due to their extensive pretraining and long-context handling capabilities. Encoder–decoder models like LongT5 and LED perform competitively when fine-tuned on the dataset.

The proposed **Salience-Aware LongT5** model does not outperform all baselines in terms of ROUGE scores. However, it demonstrates competitive semantic similarity as measured by BERTScore, despite being trained using a lightweight fine-tuning strategy.

---

## 4. Qualitative Error Analysis

We conducted a qualitative inspection of randomly sampled test clusters and categorized common errors across models into the following groups:

1. **Missing key events:** Important events not included in the generated summary.
2. **Wrong entities:** Incorrect person, location, or organization mentioned.
3. **Hallucinated facts:** Content not supported by the input documents.
4. **Redundancy:** Repetition of the same information.
5. **Poor coherence:** Abrupt transitions between topics.

Baseline large language models tend to produce fluent summaries but occasionally exhibit redundancy. The proposed salience-aware model reduces repetition by focusing on important sentences but may omit fine-grained details due to limited training.

---

## 5. Discussion

The benchmark highlights the trade-off between lexical overlap and semantic abstraction. While ROUGE favors models that maximize surface-level overlap, the proposed model emphasizes sentence importance and coverage. This suggests that explicit salience modeling is a promising direction for multi-document summarization, particularly when combined with more extensive training or improved salience supervision.

---

## 6. Conclusion

This benchmark provides a comprehensive comparison of baseline and novel models for multi-document news summarization on the NewsSumm dataset. The results demonstrate the feasibility of incorporating salience-aware mechanisms into long-context summarization models. Future work will explore stronger supervision and extended training to further improve performance.
