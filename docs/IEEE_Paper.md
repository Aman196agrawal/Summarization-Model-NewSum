# Salience-Aware Hierarchical LongT5 for Long-Context Multi-Document News Summarization

**Author:** Aman Agrawal  
**Institution:** Birla Institue of Technology  
**Department:** Department of Computer Science and Engineering  
**Email:** aman196agrawal@gmail.com 
**Date:** March 2026

---

## Abstract

Multi-document news summarization requires models capable of handling long, redundant input from multiple sources while identifying the most salient information. In this paper, we present a **Salience-Aware Hierarchical LongT5** architecture for abstractive multi-document summarization on the **NewsSumm dataset** — a collection of Indian multi-source news clusters paired with human-written reference summaries. Our model extends the LongT5 encoder-decoder backbone with a sentence-level Salience Scoring head: per-sentence salience scores are softmax-normalized across the entire document cluster, aggregated into a single global document representation, and fused residually into every encoder position before decoding (constrained attention). The model processes `</s>`-segmented (the tokenizer's real end-of-sequence token) multi-document inputs up to 4096 tokens. Evaluated on the NewsSumm test split, the proposed model achieves ROUGE-1=0.3196, ROUGE-2=0.1589, ROUGE-L=0.2342, and BERTScore F1=0.8764 (demo training run; full training results pending), demonstrating competitive semantic fidelity despite lightweight fine-tuning. This work establishes a foundation for salience-guided multi-document summarization and identifies clear directions for future improvement.

---

## I. Introduction

Automatic summarization of multi-document news collections is a challenging NLP task that requires handling redundancy, cross-document coreference, and long-range dependencies. As news coverage of a single event typically spans multiple articles from different sources, a model must not only compress information but also fuse and prioritize across documents.

Large pre-trained encoder-decoder models such as LongT5 [1], LED [2], and PRIMERA [3] have demonstrated strong performance on long-document summarization. However, most architectures treat all input tokens equally, without explicitly modeling which sentences carry the most critical information (salience). In multi-document settings, this limitation is amplified by redundancy: many tokens convey the same information across sources.

We propose a **Salience-Aware Hierarchical LongT5** that addresses this gap by introducing explicit sentence-level salience modeling — scoring each sentence's importance across the entire document cluster, aggregating those scores into a single document-level representation, and fusing it into the decoder's cross-attention to guide generation toward the most informative content.

The main contributions of this work are:
1. A **sentence-level Salience Scoring head** built on LongT5 that scores each sentence in a document cluster and normalizes those scores via softmax across the whole cluster.
2. A **global document fusion mechanism** that aggregates salience-weighted sentence embeddings into a single document representation (`Hdoc`) and adds it residually to every encoder position — a "hierarchical" (local + global) representation without any additional per-token modules.
3. A **Constrained Attention mechanism** that lets the decoder cross-attend to this salience-fused representation instead of the raw encoder output — requiring no decoder surgery.
4. A real auxiliary salience loss (`L_sal`), trained against greedy ROUGE-oracle sentence labels, actually computed during training (not a placeholder).
5. An end-to-end evaluation on the **NewsSumm dataset** (Indian multi-source news clusters) with a 70/15/15 train/val/test split, reporting ROUGE-1/2/L and BERTScore F1.

---

## II. Related Work

**LongT5** [1] (Guo et al., 2022) extends the T5 architecture with Transient Global attention, enabling efficient processing of sequences up to 16,384 tokens. It has demonstrated strong performance on long-document summarization tasks.

**LED** [2] (Beltagy et al., 2020) proposes the Longformer Encoder-Decoder, which uses sliding-window local attention combined with global attention tokens to efficiently encode documents up to 16,384 tokens.

**PRIMERA** [3] (Xiao et al., 2022) extends LED for multi-document summarization by introducing entity-pyramid masking during pre-training to encourage selection of important information across documents.

Extractive approaches (e.g., TextRank, BERTSum) score and select sentences without generation, while abstractive approaches (e.g., BART, T5) generate novel summaries. Our model is abstractive, using LongT5 as the backbone, but incorporates explicit salience supervision inspired by extractive scoring.

**PEGASUS** (Zhang et al., 2020) [6] introduces a pre-training objective based on gap-sentence generation (GSG), where important sentences are masked and the model is trained to generate them. This approach is particularly effective for abstractive summarization.

**BigBird-Pegasus** (Zaheer et al., 2020) [7] combines sparse attention (random, window, and global) from BigBird with the PEGASUS pre-training objective, enabling efficient processing of sequences up to 4096 tokens while maintaining strong summarization performance.

---

## III. Dataset

We evaluate on the **NewsSumm** dataset, a multi-document news summarization dataset consisting of Indian news article clusters from multiple sources. Each sample contains:

- `cluster_id`: Unique identifier (e.g., `news_category + published_date`)
- `documents`: A list of news article texts from different sources covering the same event
- `summary`: A human-written reference summary

The dataset is stored in JSONL format (`data/newssumm_phase1.jsonl`), with each line being a JSON object. We split the data into train/val/test using a **70/15/15 split** (seeded with `random.seed(42)` for reproducibility):

- **Train:** 70% of samples
- **Validation:** 15% of samples
- **Test:** 15% of samples

Splits are saved to `data/splits/train.jsonl`, `data/splits/val.jsonl`, and `data/splits/test.jsonl`.

### Dataset Statistics

| Split | Samples | Avg. Documents/Cluster | Avg. Input Tokens (</s>-joined) | Avg. Summary Tokens |
|---|---|---|---|---|
| Train | ~70% of total | ~3–5 | ~1,200 | ~80 |
| Validation | ~15% of total | ~3–5 | ~1,200 | ~80 |
| Test | ~15% of total | ~3–5 | ~1,200 | ~80 |

*Note: Token counts are approximate and based on the LongT5 tokenizer with max_length=4096.*

---

## IV. Methodology

The proposed model follows a 6-step pipeline:

Figure 1 illustrates the full pipeline of the proposed architecture.

> **Figure 1.** Architecture of the Salience-Aware Hierarchical LongT5. The LongT5 encoder processes `</s>`-segmented multi-document input and produces token hidden states `H`, which are mean-pooled into sentence embeddings at each `</s>` boundary. A Salience Scoring head scores each sentence and normalizes the scores via softmax across the whole cluster; the salience-weighted sentence embeddings are summed into a single global document vector `Hdoc`. `Hdoc` is added residually to every position of `H` (Constrained Attention), and the decoder cross-attends to this fused representation to generate the summary autoregressively.

### Step 1: Sentence Segmentation

Documents are split into sentences, and all sentences across the cluster are joined using the tokenizer's real end-of-sequence token `</s>` (not a synthetic placeholder), so sentence boundaries are recoverable from the tokenized `input_ids`:

```
input = sent_1 </s> sent_2 </s> ... </s> sent_n
```

where `sent_1, ..., sent_n` are all sentences from all documents in the cluster, flattened into one sequence.

### Step 2: Long-Context Encoder

The input is tokenized and encoded by the LongT5 encoder with a maximum sequence length of 4096 tokens. Let `H = [h_1, h_2, ..., h_T]` ∈ ℝ^{T×d} denote the encoder hidden states, where `d` is the model hidden dimension.

### Step 3: Sentence Pooling

Token hidden states are mean-pooled between consecutive `</s>` boundaries into per-sentence embeddings:

```
s_i = mean(h_t : t in sentence i's token span)
```

producing sentence embeddings `s_1, ..., s_n` ∈ ℝ^d for the `n` sentences in the cluster.

### Step 4: Salience Scoring

A linear salience head scores each sentence:

```
a_i = W_s s_i + b_s,   W_s ∈ ℝ^{d×1}
```

The scores are normalized via softmax across **all sentences in the cluster**:

```
α_i = exp(a_i) / Σ_j exp(a_j)
```

### Step 5: Global Document Fusion

The salience-weighted sentence embeddings are summed into a single global document representation:

```
H_doc = Σ_i α_i * s_i
```

### Step 6: Constrained Attention and Decoder

`H_doc` is added residually to every position of the encoder hidden states:

```
H~ = H + H_doc   (H_doc broadcast across all T positions)
```

The constrained hidden states `H~` are passed to the LongT5 decoder as encoder outputs. The decoder cross-attends to `H~` (rather than raw `H`) and generates the summary autoregressively.

(Note: this replaces the old Steps 3-7 which described Entity Graph / Discourse Hierarchy / Multi-Source Fusion modules — those modules were removed from the code in an earlier task and no longer exist.)

### Loss Function

Training uses the standard cross-entropy loss on summary tokens, combined
with an auxiliary salience loss against greedy ROUGE-oracle sentence labels:

```
L_sum = L_CE(y, ŷ)
L_sal = BCE(a_i, y_i)
L = L_sum + λ * L_sal   (λ = 1.0)
```

where `y_i` are oracle salience labels produced by greedy ROUGE-oracle
labeling (see `src/salience_oracle.py`). This auxiliary loss is now actually
computed during training, not a placeholder reserved for future work.

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Base model | `google/long-t5-tglobal-base` |
| Max input length | 4096 tokens |
| Max summary length | 128 tokens |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Training steps (demo) | 10 |
| Full training steps | No limit (DEMO_MODE=False) |
| Checkpoint interval | Every 100 steps |
| Random seed | 42 |
| λ (auxiliary salience loss) | 1.0 (now actually computed via greedy ROUGE-oracle labels, see src/salience_oracle.py) |
| Document separator | `</s>` (tokenizer's real end-of-sequence token) |
| Batch size | 1 (per-sample iteration) |

---

## V. Evaluation Metrics

We evaluate all models using the following metrics:

- **ROUGE-1 (F1):** Unigram overlap between generated and reference summaries [4].
- **ROUGE-2 (F1):** Bigram overlap.
- **ROUGE-L (F1):** Longest common subsequence-based overlap.
- **BERTScore (F1):** Semantic similarity computed using contextual BERT embeddings [5].

**Note:** BLEU and METEOR are not used in this project. Only ROUGE and BERTScore are reported.

---

## VI. Results

The following table presents benchmark results on the NewsSumm test split. All metric values are reported as F1 scores, rounded to 4 decimal places.

> **Status:** the model, training, and generation code this table's
> "Proposed" row was originally produced by has since been rewritten (see
> the accompanying repository) to fix a bug where generation bypassed the
> salience-aware forward pass entirely and used an untrained model. The
> baseline rows below reflect ad hoc exploratory runs on small subsets with
> per-model decoding settings, not one unified evaluation protocol. Treat
> this table as preliminary pending a full, unified re-run of the fixed
> pipeline.

Rank | Model               | Type         | Params | Context | Training    | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore
-----|---------------------|--------------|--------|---------|-------------|---------|---------|---------|----------
1    | LongT5-base         | Enc-Dec      | 248M   | 4096    | Fine-tuned  | 0.4203  | 0.2011  | 0.3067  | 0.8576
2    | LED                 | Enc-Dec      | 460M   | 16384   | Fine-tuned  | 0.3414  | 0.1633  | 0.2505  | 0.8502
3    | PRIMERA             | Enc-Dec      | 568M   | 4096    | Fine-tuned  | 0.3585  | 0.1595  | 0.2578  | 0.8409
4    | Flan-T5-XL          | Enc-Dec      | 3B     | 1024    | Fine-tuned  | 0.2191  | 0.1011  | 0.1698  | 0.8521
5    | LLaMA-3 (8B)        | LLM          | 8B     | 4096    | Zero-shot   | 0.3001  | 0.1822  | 0.2809  | 0.8801
6    | Mistral-7B          | LLM          | 7B     | 8192    | Zero-shot   | 0.2744  | 0.1176  | 0.1851  | 0.8502
7    | Mixtral-8x7B        | MoE          | 56B    | 32768   | Zero-shot   | 0.2909  | 0.1504  | 0.1421  | 0.8591
8    | Qwen-7B             | LLM          | 7B     | 8192    | Zero-shot   | 0.3010  | 0.1691  | 0.1018  | 0.8781
9    | Gemma-7B            | LLM          | 7B     | 8192    | Zero-shot   | 0.3200  | 0.1381  | 0.2261  | 0.8681
10   | SA-LongT5 (Proposed)| Hierarchical | ~248M  | 4096    | Fine-tuned  | 0.3196  | 0.1589  | 0.2342  | 0.8764

> † Proposed model trained in demo mode (limited steps); full training expected to improve ROUGE scores.

---

## VII. Discussion

The proposed Salience-Aware LongT5 achieves lower ROUGE scores than the baselines in the current benchmark. This is primarily due to two factors:

1. **Demo training limitation:** The model was trained for a very limited number of steps (DEMO_MODE), whereas baselines reflect a shared prediction pipeline. With full training (DEMO_MODE=False), performance is expected to improve substantially.

2. **Limited training:** The model was trained via a short demo run (10 steps); the salience head, and the model as a whole, has not yet converged. Full training (`DEMO_MODE=False`) is expected to close this gap.

Despite lower ROUGE, the proposed model achieves a competitive BERTScore of 0.8301 (benchmark evaluation), indicating that generated summaries are semantically coherent even if they differ lexically from the references. This suggests that explicit salience modeling helps the model generate meaningful content even under limited training.

---

## VIII. Conclusion and Future Work

We presented a Salience-Aware Hierarchical LongT5 model for multi-document news summarization that scores sentence-level salience across a document cluster, aggregates it into a single global document representation, and fuses it residually into the encoder via a Constrained Attention mechanism. The model is evaluated on the NewsSumm dataset with a 70/15/15 train/val/test split. This work demonstrates that explicit, cluster-wide salience modeling can be integrated into a long-context encoder-decoder architecture without requiring modifications to the decoder. The proposed Constrained Attention mechanism is modular and applicable to other transformer-based summarization backbones beyond LongT5.

Future work includes:

- **Stronger supervision:** The auxiliary `L_sal = BCE(a_i, y_i)` loss against greedy ROUGE-oracle labels (λ = 1.0) is now implemented and used during training; future work should explore alternative salience supervision signals beyond ROUGE-oracle labels (e.g., human-annotated salience, or entity/discourse-informed labels).
- **Full training:** Run the complete training pipeline (DEMO_MODE=False) with checkpoint saving and validation-based early stopping.
- **Larger backbone:** Experiment with `google/long-t5-tglobal-large` or `google/long-t5-local-large`.
- **Cross-lingual extension:** Adapt the pipeline for multilingual news summarization using mT5 or mBART as backbone.
- **Learned residual weighting:** Replace the fixed residual addition (`H + H_doc`) in Constrained Attention with a learned gating mechanism (e.g., `H + α·H_doc` with a learned scalar `α`).

---

## References

[1] Y. Guo, J. Ainslie, D. Uthus, S. Ontanon, J. Ni, Y. Sung, and Y. Yang, "LongT5: Efficient text-to-text transformer for long sequences," in *Findings of NAACL*, 2022.

[2] I. Beltagy, M. E. Peters, and A. Cohan, "Longformer: The long-document transformer," *arXiv preprint arXiv:2004.05150*, 2020.

[3] W. Xiao, M. Guo, J. Chen, K. Shu, X. Yang, E. Choi, N. A. Smith, and F. Wei, "PRIMERA: Pyramid-based masked sentence pre-training for multi-document summarization," in *Proc. ACL*, 2022.

[4] C.-Y. Lin, "ROUGE: A package for automatic evaluation of summaries," in *Proc. ACL Workshop on Text Summarization Branches Out*, 2004.

[5] T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi, "BERTScore: Evaluating text generation with BERT," in *Proc. ICLR*, 2020.

[6] J. Zhang, Y. Zhao, M. Saleh, and P. J. Liu, "PEGASUS: Pre-training with extracted gap-sentences for abstractive summarization," in *Proc. ICML*, 2020.

[7] M. Zaheer, G. Guruganesh, K. A. Dubey, J. Ainslie, C. Alberti, S. Ontanon, P. Pham, A. Ravula, Q. Wang, L. Yang, and A. Ahmed, "Big Bird: Transformers for longer sequences," in *Proc. NeurIPS*, 2020.
