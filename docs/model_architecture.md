# Model Architecture: Salience-Aware Hierarchical LongT5

This document provides a detailed walkthrough of the `SalienceAwareLongT5` architecture
defined in `src/novel_newssumm_model.py`.

---

## 1. Architecture Diagram

```
Input: K source documents {d_1, ..., d_K}
              │
              ▼
  ┌───────────────────────────┐
  │   Sentence Segmentation   │
  │  join with " [SEP] "      │
  └───────────┬───────────────┘
              │  input_text (single string)
              ▼
  ┌───────────────────────────┐
  │  LongT5 Tokenizer         │
  │  max_length = 4096        │
  └───────────┬───────────────┘
              │  input_ids, attention_mask
              ▼
  ┌───────────────────────────┐
  │  Long-Context Encoder     │
  │  (LongT5 TGlobal)         │
  │  H ∈ ℝ^(B × T × 768)     │
  └───┬───────────┬───────────┘
      │           │           │
      ▼           ▼           ▼
  ┌───────┐  ┌────────┐  ┌──────────┐
  │  ①    │  │  ②     │  │  ③       │
  │Salie- │  │Entity  │  │Discourse │
  │nce    │  │Graph   │  │Hierarchy │
  │Scoring│  │(NER)   │  │(MLP)     │
  └───┬───┘  └───┬────┘  └────┬─────┘
      │           │            │
      └─────┬─────┘            │
            └────────┬─────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Multi-Source Fusion │
         │  concat + Linear      │
         │  3d → d               │
         └──────────┬────────────┘
                    │  fused_H ∈ ℝ^(B × T × 768)
                    ▼
         ┌───────────────────────┐
         │  Constrained Weights  │
         │  avg scores → softmax │
         └──────────┬────────────┘
                    │  constrained_H ∈ ℝ^(B × T × 768)
                    ▼
         ┌───────────────────────┐
         │  Decoder with         │
         │  Constrained Cross-   │
         │  Attention over       │
         │  Salience, Entity,    │
         │  and Discourse        │
         │  representations      │
         └──────────┬────────────┘
                    │
                    ▼
            Generated Summary
```

---

## 2. Module-by-Module Description

### 2.1 Sentence Segmentation

**Location:** `phase4/train_novel_model.py`, `scripts/generate_novel_test_predictions.py`

```python
input_text = " [SEP] ".join(sample["documents"])
inputs = tokenizer(input_text, truncation=True, padding="max_length",
                   max_length=4096, return_tensors="pt")
```

Documents are joined using a `[SEP]` delimiter before tokenization. This allows
the encoder to learn document-boundary representations without any external dependency.

---

### 2.2 Long-Context Encoder (LongT5)

**Location:** `src/novel_newssumm_model.py` → `SalienceAwareLongT5.__init__`

```python
self.base_model = LongT5ForConditionalGeneration.from_pretrained(
    "google/long-t5-tglobal-base"
)
```

The LongT5 backbone uses Transient Global (TGlobal) attention, which combines:
- Local sliding-window attention (efficient for nearby tokens)
- Global attention on a small set of "global" tokens (for long-range dependencies)

**Output:** H ∈ ℝ^(B × T × d), where d = 768

---

### 2.3 Module ① — Salience Scoring

**Location:** `src/novel_newssumm_model.py` → `SalienceAwareLongT5.__init__` and `forward`

```python
self.salience_head = nn.Linear(hidden_size, 1)

# In forward():
salience_logits = self.salience_head(hidden_states)       # [B, T, 1]
salience_scores = torch.softmax(salience_logits, dim=1)   # [B, T, 1]
salience_rep    = hidden_states * salience_scores          # [B, T, H]
```

**Mathematical formulation:**
```
salience_logits = W_sal · H          W_sal ∈ ℝ^(d × 1)
salience_scores = softmax(salience_logits, dim=1)
salience_rep    = H ⊙ salience_scores
```

Softmax across the token dimension (dim=1) produces a proper probability
distribution, ensuring the total attention mass is conserved.

---

### 2.4 Module ② — Entity Graph (NER-Inspired Learned Attention)

**Location:** `src/novel_newssumm_model.py` → `EntityGraphModule`

```python
class EntityGraphModule(nn.Module):
    def __init__(self, hidden_size):
        self.entity_head = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        entity_logits = self.entity_head(hidden_states)  # [B, T, 1]
        return torch.sigmoid(entity_logits)               # [B, T, 1]
```

**In forward():**
```python
entity_scores = self.entity_graph(hidden_states)   # [B, T, 1]
entity_rep    = hidden_states * entity_scores       # [B, T, H]
```

**Mathematical formulation:**
```
entity_logits = W_ent · H          W_ent ∈ ℝ^(d × 1)
entity_scores = sigmoid(entity_logits)
entity_rep    = H ⊙ entity_scores
```

Sigmoid (not softmax) is appropriate here because entity scores are independent
binary decisions per token — a token can be an entity regardless of whether
other tokens are entities.

---

### 2.5 Module ③ — Discourse Hierarchy

**Location:** `src/novel_newssumm_model.py` → `DiscourseHierarchyModule`

```python
class DiscourseHierarchyModule(nn.Module):
    def __init__(self, hidden_size):
        self.discourse_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, sentence_reps):
        return self.discourse_scorer(sentence_reps)  # [B, T, 1]
```

**In forward():**
```python
discourse_scores = self.discourse_hierarchy(hidden_states)   # [B, T, 1]
discourse_rep    = hidden_states * discourse_scores           # [B, T, H]
```

**Mathematical formulation:**
```
discourse_scores = W_2 · ReLU(W_1 · H + b_1) + b_2
                   W_1 ∈ ℝ^(d × d/2),  W_2 ∈ ℝ^(d/2 × 1)
discourse_rep    = H ⊙ discourse_scores
```

The two-layer MLP allows the module to learn non-linear position/importance patterns
that a single linear layer could not capture.

---

### 2.6 Multi-Source Fusion

**Location:** `src/novel_newssumm_model.py` → `MultiSourceFusion`

```python
class MultiSourceFusion(nn.Module):
    def __init__(self, hidden_size):
        self.fusion = nn.Linear(hidden_size * 3, hidden_size)

    def forward(self, salience_rep, entity_rep, discourse_rep):
        concatenated = torch.cat(
            [salience_rep, entity_rep, discourse_rep], dim=-1
        )               # [B, T, 3d]
        return self.fusion(concatenated)  # [B, T, d]
```

**Mathematical formulation:**
```
concat  = [salience_rep ; entity_rep ; discourse_rep]    ∈ ℝ^(B × T × 3d)
fused_H = W_fuse · concat + b_fuse                       ∈ ℝ^(B × T × d)
          W_fuse ∈ ℝ^(3d × d)
```

The fusion layer appears **exactly once** in the pipeline, immediately before the
constrained attention step. It is not duplicated inside the decoder.

---

### 2.7 Decoder with Constrained Cross-Attention

**Location:** `src/novel_newssumm_model.py` → `SalienceAwareLongT5.forward`

```python
# Average the three module scores and re-normalize
fused_scores  = (salience_scores + entity_scores + discourse_scores) / 3.0
fused_weights = torch.softmax(fused_scores, dim=1)          # [B, T, 1]
constrained_H = fused_hidden * fused_weights                 # [B, T, d]

# Wrap in BaseModelOutput for the LongT5 decoder
constrained_encoder_outputs = BaseModelOutput(
    last_hidden_state=constrained_H,
    ...
)

# Decoder performs cross-attention over constrained_H
outputs = self.base_model(
    encoder_outputs=constrained_encoder_outputs,
    attention_mask=attention_mask,
    labels=labels,
    return_dict=True
)
```

**Mathematical formulation:**
```
fused_scores   = (salience_scores + entity_scores + discourse_scores) / 3
fused_weights  = softmax(fused_scores, dim=1)
constrained_H  = fused_H ⊙ fused_weights
```

Inside the decoder, the standard LongT5 cross-attention mechanism attends to
`constrained_H`. This is described as **constrained cross-attention over Salience,
Entity, and Discourse representations** — the three modules are not re-run inside
the decoder.

---

## 3. Training Procedure

**Script:** `phase4/train_novel_model.py`

```python
optimizer = AdamW(model.parameters(), lr=2e-5)

for step, sample in enumerate(dataset):
    input_text = " [SEP] ".join(sample["documents"])
    inputs = tokenizer(input_text, truncation=True,
                       padding="max_length", max_length=4096, ...)

    labels = tokenizer(sample["summary"], truncation=True,
                       padding="max_length", max_length=128, ...)["input_ids"]

    output = model(input_ids=..., attention_mask=..., labels=labels)
    loss = output["loss"]
    loss.backward()
    optimizer.step()
```

The training loss is the standard cross-entropy loss from LongT5, computed over
the constrained encoder representations. All three analysis modules (salience,
entity, discourse) are trained end-to-end through this single objective.

---

## 4. Hyperparameters

| Parameter | Value |
|-----------|-------|
| Backbone | google/long-t5-tglobal-base |
| Hidden size (d) | 768 |
| Max input length | 4096 tokens |
| Max output length | 128 tokens |
| Discourse MLP hidden | 384 (d/2) |
| Fusion input dim | 2304 (3 × d) |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Document delimiter | [SEP] |

---

## 5. Return Values

`SalienceAwareLongT5.forward()` returns a dictionary:

| Key | Shape | Description |
|-----|-------|-------------|
| `loss` | scalar | Cross-entropy summarization loss |
| `logits` | [B, L, V] | Decoder logits over vocabulary |
| `salience_scores` | [B, T, 1] | Softmax token salience weights |
| `entity_scores` | [B, T, 1] | Sigmoid entity attention weights |
| `discourse_scores` | [B, T, 1] | MLP discourse importance scores |

---

## 6. Class and File Reference

| Symbol | File | Description |
|--------|------|-------------|
| `SalienceAwareLongT5` | `src/novel_newssumm_model.py` | Main model class |
| `EntityGraphModule` | `src/novel_newssumm_model.py` | NER-inspired entity head |
| `DiscourseHierarchyModule` | `src/novel_newssumm_model.py` | Positional discourse MLP |
| `MultiSourceFusion` | `src/novel_newssumm_model.py` | Fusion linear layer |
| `train()` | `phase4/train_novel_model.py` | Training loop |
| `generate_novel_test_predictions.py` | `scripts/` | Inference script |
| `novel_model_spec.md` | `phase4/` | Architecture specification |
