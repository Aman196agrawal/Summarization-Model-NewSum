# Phase 4: Salience-Aware Hierarchical LongT5 — Novel Model Specification

## Motivation

Multi-document news summarization suffers from redundancy and missing key facts.
Baseline encoder-decoder models treat all tokens equally, leading to over-representation
of repeated information and under-emphasis of salient entities and discourse structure.

We propose a **Salience-Aware Hierarchical LongT5** model that incorporates three
parallel analysis modules on the encoder side — Salience Scoring, Entity Graph, and
Discourse Hierarchy — whose outputs are fused before the decoder performs
constrained cross-attention.

---

## Architecture Overview

```
Input Documents
      │
      ▼
Sentence Segmentation  (join with [SEP] delimiter)
      │
      ▼
Long-Context Encoder (LongT5)
      │
      ├──────────────────────┬─────────────────────────┐
      ▼                      ▼                         ▼
① Salience Scoring    ② Entity Graph           ③ Discourse Hierarchy
  (softmax over          (NER-inspired             (positional MLP
   token dim)             sigmoid head)             scorer)
      │                      │                         │
      └──────────────────────┴─────────────────────────┘
                             │
                             ▼
                   Multi-Source Fusion
                (concat + Linear projection)
                             │
                             ▼
           Decoder with Constrained Cross-Attention
        (encoder states re-weighted by fused scores)
                             │
                             ▼
                     Generated Summary
```

---

## Components

### 1. Sentence Segmentation

Before tokenization, source documents are joined with a `[SEP]` delimiter so that
the model can identify document boundaries at the token level:

```python
input_text = " [SEP] ".join(sample["documents"])
```

### 2. Long-Context Encoder (LongT5)

The backbone is `google/long-t5-tglobal-base`, which supports sequences up to
4096 tokens via Transient Global attention.

Let **H** ∈ ℝ^(B × T × d) be the encoder hidden states, where B is batch size,
T is sequence length, and d is the hidden dimension.

### 3. Module ① — Salience Scoring

A linear head is applied token-wise and normalized with softmax across the token
dimension to produce a proper probability distribution of token importance:

```
salience_logits = W_sal · H      ∈ ℝ^(B × T × 1)
salience_scores = softmax(salience_logits, dim=1)
salience_rep    = H ⊙ salience_scores
```

Softmax (not sigmoid) is used so that the scores form a normalized attention
distribution, consistent with the attention interpretation.

### 4. Module ② — Entity Graph (NER-Inspired Learned Attention)

A token-wise binary classifier learns to identify entity-like tokens without
requiring an external NLP library. Sigmoid is used here to allow independent
scores per token (unlike the salience head which must be normalized):

```
entity_logits = W_ent · H        ∈ ℝ^(B × T × 1)
entity_scores = sigmoid(entity_logits)
entity_rep    = H ⊙ entity_scores
```

### 5. Module ③ — Discourse Hierarchy

A two-layer MLP models positional/structural importance of discourse segments.
Applied token-wise (treating each token position as a discourse unit), it learns
to score segments by their contextual position in the narrative:

```
discourse_scores = ReLU(H · W_1 + b_1) · W_2 + b_2    ∈ ℝ^(B × T × 1)
discourse_rep    = H ⊙ discourse_scores
```

where W_1 ∈ ℝ^(d × d/2), W_2 ∈ ℝ^(d/2 × 1).

### 6. Multi-Source Fusion

The three weighted representations are concatenated along the feature dimension
and projected back to d via a single learned linear layer:

```
concat    = [salience_rep ; entity_rep ; discourse_rep]    ∈ ℝ^(B × T × 3d)
fused_H   = W_fuse · concat + b_fuse                       ∈ ℝ^(B × T × d)
```

This single fusion operation appears **once** in the pipeline, before the decoder.

### 7. Decoder with Constrained Cross-Attention

Rather than modifying the internal attention mechanism of LongT5, constrained
attention is implemented by re-weighting the encoder hidden states before they
are passed to the decoder. The fused scores from all three modules are averaged
and normalized:

```
fused_scores    = (salience_scores + entity_scores + discourse_scores) / 3
fused_weights   = softmax(fused_scores, dim=1)             ∈ ℝ^(B × T × 1)
constrained_H   = fused_H ⊙ fused_weights                 ∈ ℝ^(B × T × d)
```

The decoder then performs its standard cross-attention over `constrained_H`
instead of the raw encoder output. Inside the decoder block, this is described
as **constrained cross-attention over Salience, Entity, and Discourse
representations** — the three modules are *not* re-run inside the decoder.

---

## Training Objective

The total training loss is the standard sequence-to-sequence cross-entropy loss
computed by LongT5 over the constrained encoder representations:

```
L = CrossEntropy(y, Decoder(constrained_H))
```

No auxiliary salience supervision is required because salience, entity, and
discourse modules are trained end-to-end through the summarization objective.

---

## Hyperparameters

| Parameter            | Value                        |
|----------------------|------------------------------|
| Backbone             | google/long-t5-tglobal-base  |
| Max input length     | 4096 tokens                  |
| Max output length    | 128 tokens                   |
| Optimizer            | AdamW                        |
| Learning rate        | 2e-5                         |
| Hidden size (d)      | 768                          |
| Fusion input dim     | 2304 (3 × 768)               |

---

## Justification

- **Salience Scoring** reduces redundancy by down-weighting repeated tokens
- **Entity Graph** focuses the decoder on named entities and key facts
- **Discourse Hierarchy** captures positional discourse structure (lead, body, conclusion)
- **Multi-Source Fusion** enables joint reasoning over all three signals
- **Constrained Attention** guides the LongT5 decoder without architectural surgery
- All modules use only `torch`, `torch.nn`, and `transformers` — no external NLP libraries

