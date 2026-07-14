# SA-LongT5 Code Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the SA-LongT5 model, training/generation scripts, and dataset pipeline so the committed code in this repo actually implements what the author's report describes and actually runs, fixing the mentor-review bugs (generate() bypass, no checkpoint use, unmasked labels, fake separator tokens, fabricated benchmark numbers) — while honestly caveating results that can't be reproduced from this repo instead of deleting or further fabricating them.

**Architecture:** `SalienceAwareLongT5` is rewritten to match the report's Section IV math exactly (sentence-level salience scores softmax-normalized over the whole document cluster, a single global `Hdoc` vector added residually to every encoder position, decoder cross-attends to that residual-fused state). A shared `src/sentence_utils.py` builds identical `</s>`-delimited input for training, generation, and the baseline script. A new `src/salience_oracle.py` provides real greedy-ROUGE-oracle labels so the salience loss is actually computed. A new `scripts/build_newssumm_pp.py` ports the real, working NewsSumm++ dataset-construction code found in the author's Google Drive notebooks. Reporting docs get honest caveats and internal-consistency fixes, not deletions.

**Tech Stack:** Python 3.11+, PyTorch 2.x, Hugging Face `transformers` (LongT5), `rouge_score`, `evaluate`/`bert_score`, `scikit-learn`, `sentence-transformers`, `pandas`, `pytest`.

## Global Constraints

- No GPU/large dataset available in this environment — every test in this plan must run on CPU in well under a minute, using either pure-Python logic, hand-built tensors, or a tiny randomly-initialized `LongT5Config` model (never `google/long-t5-tglobal-base`'s real ~990MB pretrained weights).
- Use `T5Tokenizer` (not `AutoTokenizer`) when a real LongT5/T5 tokenizer is needed — the installed `transformers` version's `AutoTokenizer` returns a tokenizer with an empty special-tokens map for `google/long-t5-tglobal-base` (verified during planning: `tok.eos_token` comes back `None`), which would silently break the `</s>`-based sentence-boundary detection this plan depends on. `T5Tokenizer.from_pretrained("google/long-t5-tglobal-base")` correctly reports `eos_token == "</s>"`, `eos_token_id == 1`.
- The tokenizer's real `eos_token` string, when embedded directly in raw input text before tokenization, is correctly recognized as the single eos special-token id (verified: `"Hello world. </s> Second sentence."` tokenizes with exactly one eos id at the `</s>` position and one more from the tokenizer's automatic end-of-sequence append) — this is what makes real-token sentence delimiting possible instead of the old fake `[SEP]`/`[DOC]` text.
- Split ratio is cluster-level **70/15/15** (matches the report and Table II), replacing the current 80/10/10.
- Salience loss weight `λ = 1.0` (Table II), replacing the current unused `LAMBDA = 0.1` placeholder.
- Decoding is unified across baseline and novel scripts: `num_beams=4`, `max_length=256` for both, so any ROUGE difference reflects architecture, not decoding settings.
- Never claim a number is verified unless this repo's own code produced it. Where the report's numbers can't be reproduced here, add a clear caveat — don't delete the numbers and don't invent replacements.

---

### Task 1: Test scaffolding and tiny sample dataset

**Files:**
- Create: `conftest.py`
- Create: `phase4/__init__.py`
- Create: `scripts/__init__.py`
- Create: `data/sample.jsonl`
- Modify: `.gitignore`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: a repo root importable as `src.*`, `phase4.*`, `scripts.*` from any pytest invocation directory; `data/sample.jsonl` with 10 synthetic clusters for later end-to-end testing.

- [ ] **Step 1: Create `conftest.py` at the repo root**

```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

- [ ] **Step 2: Create empty package markers**

Create `phase4/__init__.py` with empty content (0 bytes).
Create `scripts/__init__.py` with empty content (0 bytes).

- [ ] **Step 3: Create the tiny synthetic sample dataset**

Create `data/sample.jsonl` with exactly this content (10 lines, one JSON object per line):

```json
{"cluster_id": "politics_2026-01-01", "documents": ["The prime minister announced a new education policy today. The policy focuses on rural schools.", "Officials confirmed the education policy will roll out next year. Rural schools will receive extra funding."], "summary": "The prime minister announced a new education policy focused on rural schools."}
{"cluster_id": "sports_2026-01-02", "documents": ["The national cricket team won the match by five wickets. The captain praised the bowling attack.", "Fans celebrated the cricket victory across the country. The bowling attack was outstanding."], "summary": "The national cricket team won by five wickets, with the captain praising the bowlers."}
{"cluster_id": "business_2026-01-03", "documents": ["The stock market closed higher on Friday. Technology shares led the gains.", "Analysts said technology shares drove Friday's rally. The market closed at a new high."], "summary": "The stock market closed higher on Friday, led by technology shares."}
{"cluster_id": "health_2026-01-04", "documents": ["A new vaccine was approved for use nationwide. Health officials called it a major milestone.", "The nationwide vaccine rollout begins next week. Officials described it as a milestone moment."], "summary": "A new vaccine was approved and will roll out nationwide next week."}
{"cluster_id": "technology_2026-01-05", "documents": ["A local startup launched an AI-powered translation app. The app supports twelve languages.", "The translation app has already gained thousands of users. It supports twelve regional languages."], "summary": "A startup launched an AI translation app supporting twelve languages."}
{"cluster_id": "environment_2026-01-06", "documents": ["Heavy rains caused flooding in several districts. Relief teams were deployed overnight.", "Flooding displaced hundreds of families overnight. Relief teams worked through the night."], "summary": "Heavy rains caused flooding in several districts, prompting overnight relief efforts."}
{"cluster_id": "world_2026-01-07", "documents": ["Leaders from several nations met to discuss trade policy. The summit lasted two days.", "The two-day trade summit ended with a joint statement. Leaders praised the discussions."], "summary": "Leaders met for a two-day summit to discuss trade policy."}
{"cluster_id": "science_2026-01-08", "documents": ["Researchers discovered a new species of frog in the rainforest. The finding was published this week.", "The newly discovered frog species was documented in a peer-reviewed journal. Scientists called it significant."], "summary": "Researchers discovered a new frog species in the rainforest and published the finding."}
{"cluster_id": "education_2026-01-09", "documents": ["A university announced a new scholarship program for rural students. Applications open next month.", "The scholarship program will fund tuition for rural students. Officials said applications open next month."], "summary": "A university launched a scholarship program for rural students, with applications opening next month."}
{"cluster_id": "finance_2026-01-10", "documents": ["The central bank kept interest rates unchanged. Economists expected the decision.", "Interest rates remain unchanged after the central bank's meeting. Analysts said the move was widely expected."], "summary": "The central bank kept interest rates unchanged, as economists had expected."}
```

- [ ] **Step 4: Fix `.gitignore` so the sample file isn't swallowed by the blanket `data/`/`*.jsonl` rules**

Read the current `.gitignore` and replace its contents with:

```
# Ignore raw data (except the tiny committed sample used for tests)
data/*
!data/sample.jsonl

# Ignore Python cache
__pycache__/
*.pyc
.ipynb_checkpoints/

# Ignore CSVs by default
*.csv

# BUT keep result CSVs
!results/*.csv

# Ignore JSONL data files (except the committed sample)
*.jsonl
!data/sample.jsonl

# Ignore pytest cache
.pytest_cache/
```

- [ ] **Step 5: Verify `data/sample.jsonl` is actually trackable**

Run: `git check-ignore -v data/sample.jsonl`
Expected: no output and a non-zero/empty result (i.e. the command finds no matching ignore rule) — if it prints a matching rule, the `.gitignore` ordering is wrong and must be fixed before continuing.

- [ ] **Step 6: Add test/pipeline dependencies to `requirements.txt`**

Replace `requirements.txt` contents with:

```
torch>=2.0.0
transformers>=4.36.0
evaluate>=0.4.0
rouge_score>=0.1.2
bert_score>=0.3.13
pandas>=1.5.0
matplotlib>=3.7.0
sentencepiece>=0.1.99
protobuf>=3.20.0
scikit-learn>=1.3.0
sentence-transformers>=2.2.0
pytest>=7.4.0
```

- [ ] **Step 7: Confirm pytest collects (no tests yet, but no import errors)**

Run: `python -m pytest --collect-only -q`
Expected: `no tests ran` with no errors.

- [ ] **Step 8: Commit**

```bash
git add conftest.py phase4/__init__.py scripts/__init__.py data/sample.jsonl .gitignore requirements.txt
git commit -m "Add test scaffolding, tiny sample dataset, and test dependencies"
```

---

### Task 2: `src/sentence_utils.py` — shared sentence splitting and input construction

**Files:**
- Create: `src/sentence_utils.py`
- Test: `tests/test_sentence_utils.py`

**Interfaces:**
- Produces: `split_sentences(text: str) -> list[str]`; `build_cluster_input(documents: list[str], tokenizer) -> tuple[str, list[str]]` where `tokenizer` only needs a `.eos_token: str` attribute. Both consumed by Tasks 4, 8, 9, 10, 11.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sentence_utils.py`:

```python
from src.sentence_utils import split_sentences, build_cluster_input


class _FakeTokenizer:
    eos_token = "</s>"


def test_split_sentences_splits_on_terminal_punctuation():
    text = "Hello world. This is a test! Is this working?"
    assert split_sentences(text) == [
        "Hello world.",
        "This is a test!",
        "Is this working?",
    ]


def test_split_sentences_handles_empty_and_whitespace_only():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_split_sentences_strips_extra_whitespace_between_sentences():
    text = "First sentence.    Second sentence."
    assert split_sentences(text) == ["First sentence.", "Second sentence."]


def test_build_cluster_input_flattens_all_documents_and_joins_with_eos():
    documents = ["Sentence one. Sentence two.", "Sentence three."]
    text, sentences = build_cluster_input(documents, _FakeTokenizer())

    assert sentences == ["Sentence one.", "Sentence two.", "Sentence three."]
    assert text == "Sentence one. </s> Sentence two. </s> Sentence three."


def test_build_cluster_input_with_single_document_single_sentence():
    documents = ["Only one sentence here."]
    text, sentences = build_cluster_input(documents, _FakeTokenizer())

    assert sentences == ["Only one sentence here."]
    assert text == "Only one sentence here."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sentence_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.sentence_utils'`

- [ ] **Step 3: Write the implementation**

Create `src/sentence_utils.py`:

```python
import re

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text):
    """Split `text` into sentences on '.', '!', or '?' followed by whitespace.

    Returns an empty list for empty/whitespace-only input.
    """
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_BOUNDARY_RE.split(text) if s.strip()]


def build_cluster_input(documents, tokenizer):
    """Flatten every sentence across all documents in a cluster into one
    `</s>`-delimited input string, using the tokenizer's real eos token as the
    delimiter (not a fake `[SEP]`/`[DOC]` placeholder) so sentence boundaries
    are recoverable from `input_ids` after tokenization.

    Sentences are joined with the eos token *between* them, relying on the
    tokenizer's automatic end-of-sequence append to mark the final sentence's
    boundary too -- this keeps exactly one eos token per sentence.

    Returns (input_text, sentences) where `sentences` is the flat list used
    for oracle labeling and must stay index-aligned with the eos positions
    that will appear in the tokenized `input_ids`.
    """
    sentences = []
    for document in documents:
        sentences.extend(split_sentences(document))

    input_text = (" " + tokenizer.eos_token + " ").join(sentences)
    return input_text, sentences
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sentence_utils.py -v`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/sentence_utils.py tests/test_sentence_utils.py
git commit -m "Add shared sentence splitting and </s>-delimited input builder"
```

---

### Task 3: `src/salience_oracle.py` — greedy ROUGE-oracle salience labels

**Files:**
- Create: `src/salience_oracle.py`
- Test: `tests/test_salience_oracle.py`

**Interfaces:**
- Consumes: `rouge_score.rouge_scorer.RougeScorer`.
- Produces: `compute_oracle_labels(sentences: list[str], reference_summary: str, max_sentences: int | None = None) -> list[int]`. Consumed by Task 8.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_salience_oracle.py`:

```python
from src.salience_oracle import compute_oracle_labels


def test_compute_oracle_labels_picks_the_matching_sentence():
    sentences = [
        "The prime minister announced a new policy today.",
        "Cricket match ends in a draw after rain.",
        "Stock markets closed higher on Friday.",
    ]
    reference = "The prime minister announced a new policy today."

    labels = compute_oracle_labels(sentences, reference)

    assert labels == [1, 0, 0]


def test_compute_oracle_labels_can_pick_more_than_one_sentence():
    sentences = [
        "The prime minister announced a new policy today.",
        "The policy focuses on rural schools and funding.",
        "Cricket match ends in a draw after rain.",
    ]
    reference = "The prime minister announced a new policy focused on rural schools and funding."

    labels = compute_oracle_labels(sentences, reference)

    assert labels[0] == 1
    assert labels[1] == 1
    assert labels[2] == 0


def test_compute_oracle_labels_returns_empty_list_for_no_sentences():
    assert compute_oracle_labels([], "some reference") == []


def test_compute_oracle_labels_respects_max_sentences_cap():
    sentences = [
        "The prime minister announced a new policy today.",
        "The policy focuses on rural schools and funding.",
        "Officials confirmed the rollout begins next year.",
    ]
    reference = "The prime minister announced a new policy focused on rural schools, funding, and next year's rollout."

    labels = compute_oracle_labels(sentences, reference, max_sentences=1)

    assert sum(labels) <= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_salience_oracle.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.salience_oracle'`

- [ ] **Step 3: Write the implementation**

Create `src/salience_oracle.py`:

```python
from rouge_score import rouge_scorer

_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)


def _combined_rouge_f1(candidate_sentences, reference_summary):
    if not candidate_sentences:
        return 0.0
    candidate = " ".join(candidate_sentences)
    scores = _SCORER.score(reference_summary, candidate)
    return (scores["rouge1"].fmeasure + scores["rouge2"].fmeasure) / 2.0


def compute_oracle_labels(sentences, reference_summary, max_sentences=None):
    """Greedy ROUGE-oracle extractive labeling.

    Returns a 0/1 label per sentence in `sentences` (same length, same
    order), marking sentences that were greedily selected because adding
    them to the running selection improved the combined ROUGE-1/ROUGE-2 F1
    against `reference_summary`. Selection stops when no remaining sentence
    improves the score, or when `max_sentences` selections have been made.
    """
    n = len(sentences)
    if n == 0:
        return []

    limit = n if max_sentences is None else max_sentences
    labels = [0] * n
    selected_indices = []
    remaining = set(range(n))
    best_score = 0.0

    while remaining and len(selected_indices) < limit:
        best_idx = None
        best_candidate_score = best_score

        for idx in remaining:
            candidate_score = _combined_rouge_f1(
                [sentences[i] for i in selected_indices] + [sentences[idx]],
                reference_summary,
            )
            if candidate_score > best_candidate_score:
                best_candidate_score = candidate_score
                best_idx = idx

        if best_idx is None:
            break

        selected_indices.append(best_idx)
        remaining.discard(best_idx)
        labels[best_idx] = 1
        best_score = best_candidate_score

    return labels
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_salience_oracle.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/salience_oracle.py tests/test_salience_oracle.py
git commit -m "Add greedy ROUGE-oracle salience labeling"
```

---

### Task 4: Rewrite `src/novel_newssumm_model.py` to match the report's architecture

**Files:**
- Modify: `src/novel_newssumm_model.py` (full rewrite)
- Test: `tests/test_novel_model.py`

**Interfaces:**
- Produces: `pool_sentences(hidden_states, input_ids, eos_token_id) -> tuple[Tensor, Tensor]`; `SalienceAwareLongT5(model_name=...)`; `SalienceAwareLongT5.from_base_model(base_model)` classmethod (test-only seam, avoids downloading pretrained weights); `.forward(input_ids, attention_mask, labels=None, salience_labels=None) -> dict`; `.generate(input_ids, attention_mask, **generate_kwargs) -> Tensor`. Consumed by Tasks 8, 9.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_novel_model.py`:

```python
import torch
import torch.nn.functional as F
import pytest
from transformers import LongT5Config, LongT5ForConditionalGeneration

from src.novel_newssumm_model import SalienceAwareLongT5, pool_sentences


def _tiny_base_model():
    config = LongT5Config(
        vocab_size=32,
        d_model=8,
        d_kv=4,
        d_ff=16,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        local_radius=4,
        global_block_size=4,
        feed_forward_proj="relu",
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    return LongT5ForConditionalGeneration(config)


@pytest.fixture
def tiny_model():
    return SalienceAwareLongT5.from_base_model(_tiny_base_model())


def test_pool_sentences_mean_pools_between_eos_boundaries():
    hidden_states = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])
    input_ids = torch.tensor([[10, 1, 20, 21, 1]])

    embeds, mask = pool_sentences(hidden_states, input_ids, eos_token_id=1)

    assert embeds.shape == (1, 2, 1)
    assert torch.allclose(embeds[0, 0], torch.tensor([1.5]))
    assert torch.allclose(embeds[0, 1], torch.tensor([4.0]))
    assert mask.tolist() == [[True, True]]


def test_forward_returns_finite_loss_and_normalized_salience_scores(tiny_model):
    input_ids = torch.tensor([[5, 6, 1, 7, 8, 9, 1]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[2, 3, 1]])

    output = tiny_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert torch.isfinite(output["loss"])
    assert output["salience_scores"].shape == (1, 2)
    assert torch.allclose(output["salience_scores"].sum(dim=1), torch.tensor([1.0]), atol=1e-5)


def test_forward_adds_salience_loss_when_labels_provided(tiny_model):
    input_ids = torch.tensor([[5, 6, 1, 7, 8, 9, 1]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[2, 3, 1]])
    salience_labels = torch.tensor([[1.0, 0.0]])

    without = tiny_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    with_sal = tiny_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        salience_labels=salience_labels,
    )

    assert with_sal["salience_loss"] is not None
    assert torch.isfinite(with_sal["salience_loss"])
    assert not torch.allclose(with_sal["loss"], without["loss"])


def test_forward_reconciles_salience_labels_length_mismatch(tiny_model):
    input_ids = torch.tensor([[5, 6, 1, 7, 8, 9, 1]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[2, 3, 1]])

    too_long = torch.zeros(1, 50)
    output_long = tiny_model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels,
        salience_labels=too_long,
    )
    assert torch.isfinite(output_long["salience_loss"])

    too_short = torch.zeros(1, 1)
    output_short = tiny_model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels,
        salience_labels=too_short,
    )
    assert torch.isfinite(output_short["salience_loss"])


def test_generate_runs_through_the_salience_path_not_the_bare_backbone(tiny_model):
    calls = []
    original = tiny_model._encode_with_salience

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    tiny_model._encode_with_salience = spy

    input_ids = torch.tensor([[5, 6, 1, 7, 8, 9, 1]])
    attention_mask = torch.ones_like(input_ids)

    generated = tiny_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=5)

    assert len(calls) == 1
    assert generated.shape[0] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_novel_model.py -v`
Expected: FAIL — `ImportError: cannot import name 'pool_sentences'` (current file has neither `pool_sentences` nor `from_base_model`).

- [ ] **Step 3: Write the implementation**

Replace all of `src/novel_newssumm_model.py` with:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LongT5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


def pool_sentences(hidden_states, input_ids, eos_token_id):
    """Mean-pool encoder token hidden states into per-sentence embeddings.

    Sentence boundaries are the positions of `eos_token_id` in `input_ids`
    (see src.sentence_utils.build_cluster_input, which joins sentences with
    the real eos token so exactly one boundary marks the end of each
    sentence). Ragged per-example sentence counts are zero-padded.

    Returns (sentence_embeds [B, S, H], sentence_mask [B, S] bool).
    """
    batch_size, _, hidden_size = hidden_states.shape

    per_example_embeds = []
    max_sentences = 0
    for b in range(batch_size):
        boundaries = (input_ids[b] == eos_token_id).nonzero(as_tuple=True)[0].tolist()
        sentence_embeds = []
        start = 0
        for end in boundaries:
            span = hidden_states[b, start:end + 1]
            if span.shape[0] > 0:
                sentence_embeds.append(span.mean(dim=0))
            start = end + 1
        if not sentence_embeds:
            sentence_embeds = [hidden_states[b].mean(dim=0)]
        per_example_embeds.append(torch.stack(sentence_embeds))
        max_sentences = max(max_sentences, len(sentence_embeds))

    sentence_embeds = hidden_states.new_zeros(batch_size, max_sentences, hidden_size)
    sentence_mask = torch.zeros(batch_size, max_sentences, dtype=torch.bool, device=hidden_states.device)
    for b, embeds in enumerate(per_example_embeds):
        n = embeds.shape[0]
        sentence_embeds[b, :n] = embeds
        sentence_mask[b, :n] = True

    return sentence_embeds, sentence_mask


class SalienceAwareLongT5(nn.Module):
    """Salience-Aware LongT5, matching the report's Section IV formulation.

    a_ij = W_s h_ij + b_s                         (per-sentence salience score)
    alpha_ij = softmax over ALL sentences in the cluster
    Hdoc = sum_ij alpha_ij * h_ij                 (single global document vector)
    H~_ij = H_ij + Hdoc                           (residual "hierarchical" fusion)
    decoder cross-attends to H~ instead of raw H  ("constrained attention")

    Loss: L = L_sum + lambda * L_sal, where L_sal is BCE(alpha, oracle_labels).
    """

    LAMBDA = 1.0

    def __init__(self, model_name="google/long-t5-tglobal-base"):
        super().__init__()
        base_model = LongT5ForConditionalGeneration.from_pretrained(model_name)
        self._init_from_base_model(base_model)

    @classmethod
    def from_base_model(cls, base_model):
        """Wrap an already-constructed LongT5ForConditionalGeneration.

        Used by tests to avoid downloading the ~990MB pretrained backbone --
        pass a tiny randomly-initialized LongT5ForConditionalGeneration(config)
        instead. Production code should use the normal constructor.
        """
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance._init_from_base_model(base_model)
        return instance

    def _init_from_base_model(self, base_model):
        self.base_model = base_model
        hidden_size = self.base_model.config.d_model
        self.salience_head = nn.Linear(hidden_size, 1)

    def _encode_with_salience(self, input_ids, attention_mask):
        encoder_outputs = self.base_model.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        hidden_states = encoder_outputs.last_hidden_state

        eos_token_id = self.base_model.config.eos_token_id
        sentence_embeds, sentence_mask = pool_sentences(hidden_states, input_ids, eos_token_id)

        salience_logits = self.salience_head(sentence_embeds).squeeze(-1)
        salience_logits = salience_logits.masked_fill(~sentence_mask, float("-inf"))
        alpha = torch.softmax(salience_logits, dim=1)

        h_doc = (alpha.unsqueeze(-1) * sentence_embeds).sum(dim=1)
        constrained_hidden = hidden_states + h_doc.unsqueeze(1)

        return constrained_hidden, alpha, sentence_mask

    def _reconcile_salience_labels(self, salience_labels, target_len):
        current_len = salience_labels.shape[1]
        if current_len < target_len:
            return F.pad(salience_labels, (0, target_len - current_len), value=0.0)
        if current_len > target_len:
            return salience_labels[:, :target_len]
        return salience_labels

    def forward(self, input_ids, attention_mask, labels=None, salience_labels=None):
        constrained_hidden, alpha, sentence_mask = self._encode_with_salience(
            input_ids, attention_mask
        )

        constrained_encoder_outputs = BaseModelOutput(last_hidden_state=constrained_hidden)
        outputs = self.base_model(
            encoder_outputs=constrained_encoder_outputs,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        loss = outputs.loss
        salience_loss = None
        if salience_labels is not None:
            reconciled = self._reconcile_salience_labels(salience_labels, alpha.shape[1])
            probs = alpha.clamp(min=1e-7, max=1 - 1e-7)
            bce = F.binary_cross_entropy(probs, reconciled, reduction="none")
            bce = bce * sentence_mask.float()
            denom = sentence_mask.float().sum().clamp(min=1.0)
            salience_loss = bce.sum() / denom
            if loss is not None:
                loss = loss + self.LAMBDA * salience_loss

        return {
            "loss": loss,
            "logits": outputs.logits,
            "salience_scores": alpha,
            "sentence_mask": sentence_mask,
            "salience_loss": salience_loss,
        }

    def generate(self, input_ids, attention_mask, **generate_kwargs):
        constrained_hidden, _, _ = self._encode_with_salience(input_ids, attention_mask)
        constrained_encoder_outputs = BaseModelOutput(last_hidden_state=constrained_hidden)
        return self.base_model.generate(
            encoder_outputs=constrained_encoder_outputs,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_novel_model.py -v`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/novel_newssumm_model.py tests/test_novel_model.py
git commit -m "Rewrite SalienceAwareLongT5 to match report's sentence-level architecture"
```

---

### Task 5: `src/metrics.py` — report `rougeLsum` instead of sentence-level `rougeL`

**Files:**
- Modify: `src/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `src.metrics.rouge`, `src.metrics.bertscore` (module-level `evaluate` metric objects, monkeypatched in the test).
- Produces: `compute_metrics(predictions, references) -> dict` (unchanged signature, `"ROUGE-L"` key now sourced from `rougeLsum`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_metrics.py`:

```python
import pytest

import src.metrics as metrics_module


def test_compute_metrics_maps_rougelsum_to_rouge_l_key(monkeypatch):
    monkeypatch.setattr(
        metrics_module.rouge,
        "compute",
        lambda predictions, references: {
            "rouge1": 0.5,
            "rouge2": 0.3,
            "rougeL": 0.1,
            "rougeLsum": 0.4,
        },
    )
    monkeypatch.setattr(
        metrics_module.bertscore,
        "compute",
        lambda predictions, references, lang: {"f1": [0.9, 0.8]},
    )

    result = metrics_module.compute_metrics(["a"], ["b"])

    assert result["ROUGE-1"] == 0.5
    assert result["ROUGE-2"] == 0.3
    assert result["ROUGE-L"] == 0.4
    assert result["BERTScore"] == pytest.approx(0.85)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL — `assert 0.1 == 0.4` (current code reads `rouge_scores["rougeL"]`, gets `0.1`)

- [ ] **Step 3: Fix `src/metrics.py`**

Modify the `return` statement in `compute_metrics` (the only change in this file):

```python
    return {
        "ROUGE-1": rouge_scores["rouge1"],
        "ROUGE-2": rouge_scores["rouge2"],
        "ROUGE-L": rouge_scores["rougeLsum"],
        "BERTScore": sum(bert_scores["f1"]) / len(bert_scores["f1"])
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "Report rougeLsum instead of sentence-level rougeL (summarization convention)"
```

---

### Task 6: `src/newssumm_dataset.py` — subclass `torch.utils.data.Dataset`

**Files:**
- Modify: `src/newssumm_dataset.py`
- Test: `tests/test_newssumm_dataset.py`

**Interfaces:**
- Produces: `NewsSummDataset(jsonl_path)` — now a real `torch.utils.data.Dataset` subclass, same `__len__`/`__getitem__` behavior as before.

- [ ] **Step 1: Write the failing test**

Create `tests/test_newssumm_dataset.py`:

```python
import json

from torch.utils.data import Dataset

from src.newssumm_dataset import NewsSummDataset


def test_newssumm_dataset_is_a_torch_dataset_subclass(tmp_path):
    jsonl_path = tmp_path / "sample.jsonl"
    jsonl_path.write_text(
        json.dumps({"cluster_id": "c1", "documents": ["doc a"], "summary": "sum a"}) + "\n"
        + json.dumps({"cluster_id": "c2", "documents": ["doc b"], "summary": "sum b"}) + "\n"
    )

    dataset = NewsSummDataset(str(jsonl_path))

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2
    assert dataset[0] == {"cluster_id": "c1", "documents": ["doc a"], "summary": "sum a"}
    assert dataset[1]["cluster_id"] == "c2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_newssumm_dataset.py -v`
Expected: FAIL — `assert isinstance(dataset, Dataset)` is `False` (current class doesn't subclass `Dataset`)

- [ ] **Step 3: Fix `src/newssumm_dataset.py`**

Replace its contents with:

```python
import json

from torch.utils.data import Dataset


class NewsSummDataset(Dataset):
    """
    Dataset loader for NewsSumm multi-document summarization.
    """

    def __init__(self, jsonl_path):
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "cluster_id": sample["cluster_id"],
            "documents": sample["documents"],
            "summary": sample["summary"]
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_newssumm_dataset.py -v`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/newssumm_dataset.py tests/test_newssumm_dataset.py
git commit -m "Make NewsSummDataset a real torch.utils.data.Dataset subclass"
```

---

### Task 7: `scripts/split_dataset.py` — cluster-level 70/15/15 split

**Files:**
- Modify: `scripts/split_dataset.py`
- Test: `tests/test_split_dataset.py`

**Interfaces:**
- Produces: `compute_split_indices(n, train_ratio, val_ratio) -> tuple[int, int]`; `split_samples(samples, train_ratio, val_ratio) -> tuple[list, list, list]`. Both pure functions, no file I/O, so they're directly testable.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_split_dataset.py`:

```python
from scripts.split_dataset import compute_split_indices, split_samples


def test_compute_split_indices_70_15_15_on_100_samples():
    split_train, split_val = compute_split_indices(100, train_ratio=0.70, val_ratio=0.15)

    assert split_train == 70
    assert split_val == 85


def test_split_samples_produces_70_15_15_partition():
    samples = list(range(20))

    train, val, test = split_samples(samples, train_ratio=0.70, val_ratio=0.15)

    assert len(train) == 14
    assert len(val) == 3
    assert len(test) == 3
    assert sorted(train + val + test) == samples
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_split_dataset.py -v`
Expected: FAIL — `ImportError: cannot import name 'compute_split_indices'`

- [ ] **Step 3: Rewrite `scripts/split_dataset.py`**

Replace its contents with:

```python
import json
import random
from pathlib import Path

INPUT_FILE = "data/newssumm_phase1.jsonl"
OUTPUT_DIR = Path("data/splits")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# test gets the remaining 15%
SEED = 42


def compute_split_indices(n, train_ratio, val_ratio):
    """Return (split_train, split_val) boundary indices for a list of length n."""
    split_train = int(n * train_ratio)
    split_val = int(n * (train_ratio + val_ratio))
    return split_train, split_val


def split_samples(samples, train_ratio, val_ratio):
    """Split `samples` (already shuffled by the caller if desired) into
    (train, val, test) lists using cluster-level ratios -- each element of
    `samples` is one cluster, so this is a cluster-level split."""
    split_train, split_val = compute_split_indices(len(samples), train_ratio, val_ratio)
    train = samples[:split_train]
    val = samples[split_train:split_val]
    test = samples[split_val:]
    return train, val, test


def save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x) + "\n")


def main():
    random.seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    random.shuffle(samples)

    train_samples, val_samples, test_samples = split_samples(samples, TRAIN_RATIO, VAL_RATIO)

    save(OUTPUT_DIR / "train.jsonl", train_samples)
    save(OUTPUT_DIR / "val.jsonl", val_samples)
    save(OUTPUT_DIR / "test.jsonl", test_samples)

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_split_dataset.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/split_dataset.py tests/test_split_dataset.py
git commit -m "Change dataset split to cluster-level 70/15/15, matching the report"
```

---

### Task 8: Rewrite `phase4/train_novel_model.py` — label masking, oracle labels, real checkpoint

**Files:**
- Modify: `phase4/train_novel_model.py` (full rewrite)
- Test: `tests/test_train_novel_model.py`

**Interfaces:**
- Consumes: `src.sentence_utils.build_cluster_input`, `src.salience_oracle.compute_oracle_labels`, `src.novel_newssumm_model.SalienceAwareLongT5`.
- Produces: `prepare_batch(sample, tokenizer, device, max_input_length=4096, max_summary_length=128) -> dict` (testable in isolation). Checkpoint now saved under `results/sa_longt5/checkpoints/`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_novel_model.py`:

```python
import torch
from transformers import T5Tokenizer

from phase4.train_novel_model import prepare_batch, MODEL_NAME


def test_prepare_batch_masks_padding_and_produces_aligned_salience_labels():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    sample = {
        "documents": [
            "The prime minister announced a new policy today. Markets reacted calmly."
        ],
        "summary": "The prime minister announced a new policy today.",
    }

    batch = prepare_batch(
        sample, tokenizer, device=torch.device("cpu"),
        max_input_length=64, max_summary_length=16,
    )

    assert (batch["labels"] == tokenizer.pad_token_id).sum().item() == 0
    assert (batch["labels"] == -100).any()
    assert batch["salience_labels"].shape[0] == 1
    assert batch["salience_labels"].sum().item() >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train_novel_model.py -v`
Expected: FAIL — `ImportError: cannot import name 'prepare_batch'`

- [ ] **Step 3: Rewrite `phase4/train_novel_model.py`**

Replace its contents with:

```python
import os
import random
import torch
from torch.optim import AdamW
from transformers import T5Tokenizer

from src.novel_newssumm_model import SalienceAwareLongT5
from src.newssumm_dataset import NewsSummDataset
from src.sentence_utils import build_cluster_input
from src.salience_oracle import compute_oracle_labels

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
DEMO_MODE = True          # Set to False for full training
MAX_STEPS = 10            # Used only when DEMO_MODE = True; None = full training
MODEL_NAME = "google/long-t5-tglobal-base"
CHECKPOINT_DIR = "results/sa_longt5/checkpoints"
CHECKPOINT_EVERY = 500    # Save a checkpoint every N steps (full training)
MAX_INPUT_LENGTH = 4096
MAX_SUMMARY_LENGTH = 128


def prepare_batch(sample, tokenizer, device, max_input_length=MAX_INPUT_LENGTH,
                   max_summary_length=MAX_SUMMARY_LENGTH):
    """Tokenize one training sample end to end:
    - builds the </s>-joined multi-document input (src.sentence_utils),
    - computes greedy ROUGE-oracle salience labels aligned to those sentences,
    - masks summary padding with -100 so the loss ignores it (fixes the bug
      where the model was being trained to predict <pad> tokens).
    """
    input_text, sentences = build_cluster_input(sample["documents"], tokenizer)

    inputs = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=max_input_length,
        return_tensors="pt",
    )

    target = tokenizer(
        sample["summary"],
        truncation=True,
        padding="max_length",
        max_length=max_summary_length,
        return_tensors="pt",
    )
    labels = target["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    oracle_labels = compute_oracle_labels(sentences, sample["summary"])
    salience_labels = torch.tensor([oracle_labels], dtype=torch.float)

    return {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
        "labels": labels.to(device),
        "salience_labels": salience_labels.to(device),
    }


def train():
    # -----------------------------
    # Reproducibility
    # -----------------------------
    torch.manual_seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # -----------------------------
    # Load tokenizer and model
    # -----------------------------
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = SalienceAwareLongT5(MODEL_NAME)
    model.to(device)

    # -----------------------------
    # Load dataset (train split)
    # -----------------------------
    dataset = NewsSummDataset("data/splits/train.jsonl")

    optimizer = AdamW(model.parameters(), lr=2e-5)

    model.train()

    step_limit = MAX_STEPS if DEMO_MODE else None

    # -----------------------------
    # Training loop
    # -----------------------------
    for step, sample in enumerate(dataset):
        if step_limit is not None and step >= step_limit:
            break

        batch = prepare_batch(sample, tokenizer, device)

        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            salience_labels=batch["salience_labels"],
        )

        loss = output["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")

        if not DEMO_MODE and CHECKPOINT_EVERY is not None and CHECKPOINT_EVERY > 0 and (step + 1) % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_step_{step + 1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    # Save final checkpoint
    final_ckpt = os.path.join(CHECKPOINT_DIR, "model_final.pt")
    torch.save(model.state_dict(), final_ckpt)
    print(f"Final checkpoint saved: {final_ckpt}")

    mode_label = "demo run" if DEMO_MODE else "full training"
    print(f"Training finished ({mode_label})")


if __name__ == "__main__":
    train()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_train_novel_model.py -v`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add phase4/train_novel_model.py tests/test_train_novel_model.py
git commit -m "Mask padding in labels and wire real oracle salience supervision into training"
```

---

### Task 9: Rewrite `scripts/generate_novel_test_predictions.py` — checkpoint loading, real `generate()`

**Files:**
- Modify: `scripts/generate_novel_test_predictions.py` (full rewrite)
- Test: `tests/test_generate_novel_test_predictions.py`

**Interfaces:**
- Consumes: `src.sentence_utils.build_cluster_input`, `src.novel_newssumm_model.SalienceAwareLongT5`.
- Produces: `generate_predictions(model, tokenizer, samples, device, ...) -> Iterator[dict]`; `load_trained_model(model_name, checkpoint_path, device) -> SalienceAwareLongT5` (raises `FileNotFoundError` if no checkpoint exists).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_generate_novel_test_predictions.py`:

```python
import torch
import pytest

from scripts.generate_novel_test_predictions import generate_predictions, load_trained_model


class _FakeBatchEncoding(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    eos_token = "</s>"

    def __call__(self, text, **kwargs):
        return _FakeBatchEncoding({
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        })

    def decode(self, ids, skip_special_tokens=True):
        return "generated summary text"


class _FakeModel:
    def generate(self, input_ids, attention_mask, max_length, num_beams):
        return torch.tensor([[9, 9, 9]])


def test_generate_predictions_calls_model_generate_and_formats_output():
    samples = [{"cluster_id": "c1", "documents": ["Doc one."], "summary": "Reference summary."}]

    records = list(
        generate_predictions(_FakeModel(), _FakeTokenizer(), samples, device=torch.device("cpu"))
    )

    assert records == [{
        "cluster_id": "c1",
        "generated_summary": "generated summary text",
        "reference_summary": "Reference summary.",
    }]


def test_load_trained_model_raises_when_checkpoint_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.pt"

    with pytest.raises(FileNotFoundError):
        load_trained_model("irrelevant-model-name", str(missing_path), torch.device("cpu"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_generate_novel_test_predictions.py -v`
Expected: FAIL — `ImportError: cannot import name 'generate_predictions'`

- [ ] **Step 3: Rewrite `scripts/generate_novel_test_predictions.py`**

Replace its contents with:

```python
import os
import sys
import json
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.sentence_utils import build_cluster_input

MODEL_NAME = "google/long-t5-tglobal-base"
TEST_FILE = "data/splits/test.jsonl"
OUT_FILE = "results/sa_longt5/test_predictions.jsonl"
CHECKPOINT_PATH = "results/sa_longt5/checkpoints/model_final.pt"
MAX_INPUT_LENGTH = 4096
MAX_OUTPUT_LENGTH = 256
NUM_BEAMS = 4


def generate_predictions(model, tokenizer, samples, device, max_input_length=MAX_INPUT_LENGTH,
                          max_output_length=MAX_OUTPUT_LENGTH, num_beams=NUM_BEAMS):
    """Run the salience-aware model over `samples` (dicts with 'cluster_id',
    'documents', 'summary') and yield one prediction record per sample.
    Calls model.generate(...) -- the SalienceAwareLongT5 wrapper's own
    generate(), not model.base_model.generate() -- so beam search actually
    depends on the salience-aware constrained hidden states.
    """
    for sample in samples:
        input_text, _ = build_cluster_input(sample["documents"], tokenizer)
        inputs = tokenizer(
            input_text, truncation=True, max_length=max_input_length, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_output_length,
                num_beams=num_beams,
            )

        generated_summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        yield {
            "cluster_id": sample["cluster_id"],
            "generated_summary": generated_summary,
            "reference_summary": sample["summary"],
        }


def load_trained_model(model_name, checkpoint_path, device):
    """Load a trained SalienceAwareLongT5 checkpoint. Raises FileNotFoundError
    (rather than silently evaluating untrained weights) if no checkpoint has
    been produced yet -- run phase4/train_novel_model.py first.
    """
    from src.novel_newssumm_model import SalienceAwareLongT5

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No trained checkpoint found at {checkpoint_path}. Run "
            "phase4/train_novel_model.py first -- evaluating an untrained "
            "model would not reflect the salience-aware architecture."
        )

    model = SalienceAwareLongT5(model_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def main():
    from transformers import T5Tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading tokenizer and trained novel model...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = load_trained_model(MODEL_NAME, CHECKPOINT_PATH, device)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    count = 0
    with open(TEST_FILE) as fin, open(OUT_FILE, "w") as fout:
        samples = (json.loads(line) for line in fin)
        for record in generate_predictions(model, tokenizer, samples, device):
            fout.write(json.dumps(record) + "\n")
            count += 1
            if count % 20 == 0:
                print(f"Generated {count} summaries")

    print(f"Saved {count} predictions to {OUT_FILE}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_generate_novel_test_predictions.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_novel_test_predictions.py tests/test_generate_novel_test_predictions.py
git commit -m "Load a trained checkpoint and use SalienceAwareLongT5.generate() for predictions"
```

---

### Task 10: Rewrite `scripts/generate_test_predictions.py` — unify input/decoding with the novel script

**Files:**
- Modify: `scripts/generate_test_predictions.py` (full rewrite)
- Test: `tests/test_generate_test_predictions.py`

**Interfaces:**
- Consumes: `src.sentence_utils.build_cluster_input`.
- Produces: `generate_predictions(model, tokenizer, samples, device, ...) -> Iterator[dict]` (baseline-model version — takes a plain HF seq2seq model, not the salience wrapper).

- [ ] **Step 1: Write the failing test**

Create `tests/test_generate_test_predictions.py`:

```python
import torch

from scripts.generate_test_predictions import generate_predictions


class _FakeBatchEncoding(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    eos_token = "</s>"

    def __call__(self, text, **kwargs):
        return _FakeBatchEncoding({
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        })

    def decode(self, ids, skip_special_tokens=True):
        return "baseline generated summary"


class _FakeModel:
    def generate(self, **kwargs):
        assert kwargs["max_length"] == 256
        assert kwargs["num_beams"] == 4
        return torch.tensor([[7, 7, 7]])


def test_generate_predictions_uses_shared_input_builder_and_unified_decoding():
    samples = [{"cluster_id": "c1", "documents": ["Doc one. Doc two."], "summary": "Reference."}]

    records = list(
        generate_predictions(_FakeModel(), _FakeTokenizer(), samples, device=torch.device("cpu"))
    )

    assert records == [{
        "cluster_id": "c1",
        "generated_summary": "baseline generated summary",
        "reference_summary": "Reference.",
    }]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_generate_test_predictions.py -v`
Expected: FAIL — `ImportError: cannot import name 'generate_predictions'`

- [ ] **Step 3: Rewrite `scripts/generate_test_predictions.py`**

Replace its contents with:

```python
import json
import os
import sys
from pathlib import Path
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.sentence_utils import build_cluster_input

HF_MODEL = "google/long-t5-tglobal-base"
MODEL_TAG = "longt5_baseline"
TEST_FILE = "data/splits/test.jsonl"
OUT_DIR = Path(f"results/{MODEL_TAG}")
MAX_INPUT_LENGTH = 4096
MAX_OUTPUT_LENGTH = 256
NUM_BEAMS = 4


def generate_predictions(model, tokenizer, samples, device, max_input_length=MAX_INPUT_LENGTH,
                          max_output_length=MAX_OUTPUT_LENGTH, num_beams=NUM_BEAMS):
    """Baseline LongT5 predictions using the same </s>-joined input
    construction and the same beam=4/max_length=256 decoding as the novel
    model's script, so any ROUGE gap between the two reflects the
    architecture rather than mismatched input formatting or decoding
    settings.
    """
    for sample in samples:
        input_text, _ = build_cluster_input(sample["documents"], tokenizer)
        inputs = tokenizer(
            input_text, truncation=True, max_length=max_input_length, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_length=max_output_length, num_beams=num_beams
            )

        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        yield {
            "cluster_id": sample["cluster_id"],
            "generated_summary": summary,
            "reference_summary": sample["summary"],
        }


def main():
    from transformers import T5Tokenizer, AutoModelForSeq2SeqLM

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    tokenizer = T5Tokenizer.from_pretrained(HF_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        HF_MODEL,
        device_map="auto",
        torch_dtype=torch.float16 if device_str == "cuda" else torch.float32,
    )
    model.eval()
    print("Model loaded")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "test_predictions.jsonl"

    count = 0
    with open(TEST_FILE) as fin, open(out_file, "w") as fout:
        samples = (json.loads(line) for line in fin)
        for record in generate_predictions(model, tokenizer, samples, device=model.device):
            fout.write(json.dumps(record) + "\n")
            count += 1
            if count % 10 == 0:
                print(f"Generated {count} summaries")

    print(f"Finished. Wrote {count} predictions to {out_file}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_generate_test_predictions.py -v`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_test_predictions.py tests/test_generate_test_predictions.py
git commit -m "Unify baseline input construction and decoding settings with the novel model script"
```

---

### Task 11: Update `scripts/prepare_inputs.py` to use the shared input builder

**Files:**
- Modify: `scripts/prepare_inputs.py` (full rewrite)

**Interfaces:**
- Consumes: `src.sentence_utils.build_cluster_input` (already tested in Task 2 — this file is a thin manual-preview CLI wrapper with no new logic, so no dedicated test is added).

- [ ] **Step 1: Rewrite `scripts/prepare_inputs.py`**

Replace its contents with:

```python
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.newssumm_dataset import NewsSummDataset
from src.sentence_utils import build_cluster_input


class _EosOnlyTokenizer:
    """Minimal stand-in exposing just the `.eos_token` attribute
    build_cluster_input needs, for previewing input construction without
    loading a full tokenizer."""
    eos_token = "</s>"


if __name__ == "__main__":
    dataset = NewsSummDataset("data/newssumm_phase1.jsonl")

    sample = dataset[0]
    model_input, sentences = build_cluster_input(sample["documents"], _EosOnlyTokenizer())

    print("INPUT SAMPLE:\n")
    print(model_input[:1500])
    print(f"\n({len(sentences)} sentences detected)")
    print("\nTARGET SUMMARY:\n")
    print(sample["summary"])
```

- [ ] **Step 2: Confirm it still imports cleanly**

Run: `python -c "import ast; ast.parse(open('scripts/prepare_inputs.py').read())"`
Expected: no output (valid syntax; the script itself needs `data/newssumm_phase1.jsonl` to actually run, which doesn't exist in this environment — that's fine, it's a manual preview tool, not part of the automated tests)

- [ ] **Step 3: Commit**

```bash
git add scripts/prepare_inputs.py
git commit -m "Use the shared </s>-joined input builder in prepare_inputs.py, remove third separator format"
```

---

### Task 12: `scripts/build_newssumm_pp.py` — port the real NewsSumm++ dataset pipeline

**Files:**
- Create: `scripts/build_newssumm_pp.py`
- Test: `tests/test_build_newssumm_pp.py`

**Interfaces:**
- Produces: `clean_text(text) -> str`; `map_category(text) -> str`; `remove_nulls_and_duplicates(df) -> DataFrame`; `filter_near_duplicates(df, threshold=0.90, ...) -> DataFrame`; `cluster_documents(df, embedder, n_clusters, ...) -> DataFrame`; `filter_cluster_size(df, min_docs=7, max_docs=10) -> DataFrame`; `build_newssumm_pp(df, embedder, n_clusters, ...) -> DataFrame`. `embedder` is any `callable(list[str]) -> np.ndarray` — production code passes a `SentenceTransformer('all-MiniLM-L6-v2').encode`, tests pass a tiny deterministic fake.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_build_newssumm_pp.py`:

```python
import numpy as np
import pandas as pd

from scripts.build_newssumm_pp import (
    clean_text,
    map_category,
    remove_nulls_and_duplicates,
    filter_near_duplicates,
    filter_cluster_size,
    build_newssumm_pp,
)


def test_clean_text_lowercases_and_strips_punctuation():
    assert clean_text("Hello, World!! 123") == "hello world 123"


def test_map_category_matches_known_keywords():
    assert map_category("Local Politics News") == "politics"
    assert map_category("Cricket World Cup Final") == "sports"
    assert map_category("Completely unrelated topic xyz") == "other"


def test_remove_nulls_and_duplicates_drops_null_empty_and_duplicate_rows():
    df = pd.DataFrame({
        "article_text": ["a", "a", None, "b", ""],
        "human_summary": ["s1", "s1", "s2", "s3", "s4"],
    })

    result = remove_nulls_and_duplicates(df)

    assert result["article_text"].tolist() == ["a", "b"]


def test_filter_near_duplicates_removes_highly_similar_rows():
    df = pd.DataFrame({
        "clean_article": [
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps over the lazy dog today",
            "completely different content about space exploration",
        ]
    })

    result = filter_near_duplicates(df, threshold=0.80)

    assert len(result) == 2


def test_filter_cluster_size_keeps_only_clusters_in_range():
    df = pd.DataFrame({"cluster_id": [0, 0, 0] + [1] * 9 + [2]})

    result = filter_cluster_size(df, min_docs=7, max_docs=10)

    assert set(result["cluster_id"].unique()) == {1}
    assert len(result) == 9


def test_build_newssumm_pp_end_to_end_with_fake_embedder():
    df = pd.DataFrame({
        "article_text": [f"unique political article number {i + 10} about elections" for i in range(7)]
                        + [f"unique sports article number {i + 10} about cricket matches" for i in range(7)],
        "human_summary": [f"summary {i}" for i in range(14)],
        "headline": [f"headline {i}" for i in range(14)],
        "news_category": (["Politics"] * 7) + (["Sports"] * 7),
    })

    def fake_embedder(texts):
        return np.array([[0.0, 0.0] if "political" in t else [1.0, 1.0] for t in texts])

    result = build_newssumm_pp(
        df, fake_embedder, n_clusters=2, near_dup_threshold=0.99, min_docs=7, max_docs=10
    )

    assert set(result["mapped_category"].unique()) <= {"politics", "sports"}
    assert result["cluster_id"].nunique() == 2
    assert len(result) == 14
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_build_newssumm_pp.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.build_newssumm_pp'`

- [ ] **Step 3: Write the implementation**

Create `scripts/build_newssumm_pp.py`:

```python
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans

# The 22 fixed categories NewsSumm++ maps the original ~4,485 free-text
# categories onto (ported from the author's Colab pipeline).
FIXED_CATEGORIES = [
    "politics", "business", "economy", "technology", "science",
    "health", "sports", "entertainment", "world", "education",
    "environment", "crime", "law", "defense", "travel",
    "lifestyle", "culture", "opinion", "weather",
    "finance", "energy", "media",
]

_CATEGORY_KEYWORDS = [
    ("politic", "politics"),
    ("business", "business"),
    ("economy", "economy"),
    ("tech", "technology"),
    ("ai", "technology"),
    ("science", "science"),
    ("health", "health"),
    ("sport", "sports"),
    ("cricket", "sports"),
    ("entertain", "entertainment"),
    ("movie", "entertainment"),
    ("world", "world"),
    ("international", "world"),
    ("education", "education"),
    ("climate", "environment"),
    ("environment", "environment"),
    ("crime", "crime"),
    ("law", "law"),
    ("court", "law"),
    ("defense", "defense"),
    ("military", "defense"),
    ("travel", "travel"),
    ("lifestyle", "lifestyle"),
    ("culture", "culture"),
    ("opinion", "opinion"),
    ("editorial", "opinion"),
    ("weather", "weather"),
    ("finance", "finance"),
    ("stock", "finance"),
    ("energy", "energy"),
    ("media", "media"),
    ("news", "media"),
]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def map_category(text):
    text = str(text).lower()
    for keyword, category in _CATEGORY_KEYWORDS:
        if keyword in text:
            return category
    return "other"


def remove_nulls_and_duplicates(df):
    df = df.dropna(subset=["article_text", "human_summary"])
    df = df[(df["article_text"].str.strip() != "") & (df["human_summary"].str.strip() != "")]
    df = df.drop_duplicates(subset=["article_text"])
    return df


def filter_near_duplicates(df, threshold=0.90, max_features=20000, batch_size=2000):
    """Drop rows whose TF-IDF cosine similarity to any other row exceeds
    `threshold` (keeps the first occurrence, drops the rest)."""
    texts = df["clean_article"].tolist()
    if len(texts) < 2:
        return df

    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)

    to_remove = set()
    for start in range(0, X.shape[0], batch_size):
        end = min(start + batch_size, X.shape[0])
        sim_matrix = cosine_similarity(X[start:end], X)
        for row_idx in range(sim_matrix.shape[0]):
            global_row = start + row_idx
            for col_idx in range(sim_matrix.shape[1]):
                if global_row != col_idx and global_row not in to_remove and sim_matrix[row_idx, col_idx] > threshold:
                    to_remove.add(col_idx)

    keep_positions = [i for i in range(len(df)) if i not in to_remove]
    return df.iloc[keep_positions]


def cluster_documents(df, embedder, n_clusters, batch_size=10000, random_state=42):
    """Cluster rows by semantic similarity of `headline + clean_article`.
    `embedder` is any callable(list[str]) -> array-like of embeddings --
    production code passes SentenceTransformer('all-MiniLM-L6-v2').encode.
    """
    texts = (df["headline"].astype(str) + " " + df["clean_article"]).tolist()
    embeddings = embedder(texts)

    n_clusters = max(1, min(n_clusters, len(df)))
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=random_state)

    df = df.copy()
    df["cluster_id"] = kmeans.fit_predict(embeddings)
    return df


def filter_cluster_size(df, min_docs=7, max_docs=10):
    cluster_sizes = df["cluster_id"].value_counts()
    valid_clusters = cluster_sizes[(cluster_sizes >= min_docs) & (cluster_sizes <= max_docs)].index
    return df[df["cluster_id"].isin(valid_clusters)]


def build_newssumm_pp(df, embedder, n_clusters, near_dup_threshold=0.90, min_docs=7, max_docs=10):
    """Full NewsSumm -> NewsSumm++ pipeline: null/duplicate removal, text
    normalization, near-duplicate filtering, semantic clustering, cluster-size
    filtering (7-10 docs/cluster), and category mapping to the 22 fixed
    categories -- ported from the author's working Colab notebook
    (more_about_dataset_filtering.ipynb), not previously present in this repo.
    """
    df = remove_nulls_and_duplicates(df)
    df = df.copy()
    df["clean_article"] = df["article_text"].apply(clean_text)
    df["clean_summary"] = df["human_summary"].apply(clean_text)

    df = filter_near_duplicates(df, threshold=near_dup_threshold)
    df = cluster_documents(df, embedder, n_clusters=n_clusters)
    df = filter_cluster_size(df, min_docs=min_docs, max_docs=max_docs)

    df = df.copy()
    df["mapped_category"] = df["news_category"].apply(map_category)
    df = df[df["mapped_category"] != "other"]

    return df


def main():
    from sentence_transformers import SentenceTransformer

    input_file = "data/newssumm_raw.csv"
    output_file = "data/newssumm_pp.csv"

    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip().str.replace("\n", "")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embedder = lambda texts: embedding_model.encode(texts, batch_size=64, show_progress_bar=True)

    n_clusters = max(1, len(df) // 8)
    result = build_newssumm_pp(df, embedder, n_clusters=n_clusters)

    result.to_csv(output_file, index=False)
    print(f"NewsSumm++ saved: {len(result)} rows, {result['cluster_id'].nunique()} clusters")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_build_newssumm_pp.py -v`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/build_newssumm_pp.py tests/test_build_newssumm_pp.py
git commit -m "Add NewsSumm++ dataset construction pipeline, ported from the working Colab notebook"
```

---

### Task 13: Rewrite `docs/data_pipeline.md` to describe the real pipeline

**Files:**
- Modify: `docs/data_pipeline.md` (full rewrite)

- [ ] **Step 1: Replace `docs/data_pipeline.md` contents**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add docs/data_pipeline.md
git commit -m "Rewrite data_pipeline.md to describe the real NewsSumm++ pipeline"
```

---

### Task 14: Consolidate metrics-collection scripts onto one real aggregator

**Files:**
- Modify: `scripts/collect_all_metrics.py` (full rewrite)
- Delete: `scripts/save_all_metrics.py`
- Delete: `scripts/collect_baseline_scores.py`
- Test: `tests/test_collect_all_metrics.py`

**Interfaces:**
- Produces: `aggregate_metrics(results_dir) -> dict`; `write_outputs(all_metrics, json_path, csv_path) -> None`. This becomes the single script that produces `results/metrics_all_models.json`/`.csv` — no more hardcoded numbers, no more duplicate/contradicting `collect_baseline_scores.py` output.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_collect_all_metrics.py`:

```python
import json

from scripts.collect_all_metrics import aggregate_metrics, write_outputs


def test_aggregate_metrics_reads_test_metrics_json_from_each_model_dir(tmp_path):
    model_dir = tmp_path / "longt5_baseline"
    model_dir.mkdir()
    (model_dir / "test_metrics.json").write_text(json.dumps({
        "ROUGE-1": 0.1, "ROUGE-2": 0.2, "ROUGE-L": 0.3, "BERTScore": 0.4
    }))

    result = aggregate_metrics(tmp_path)

    assert result == {
        "longt5_baseline": {"ROUGE-1": 0.1, "ROUGE-2": 0.2, "ROUGE-L": 0.3, "BERTScore": 0.4}
    }


def test_aggregate_metrics_ignores_dirs_without_test_metrics_json(tmp_path):
    (tmp_path / "empty_dir").mkdir()

    result = aggregate_metrics(tmp_path)

    assert result == {}


def test_write_outputs_creates_json_and_csv(tmp_path):
    all_metrics = {"m1": {"ROUGE-1": 0.1, "ROUGE-2": 0.2, "ROUGE-L": 0.3, "BERTScore": 0.4}}
    json_path = tmp_path / "out.json"
    csv_path = tmp_path / "out.csv"

    write_outputs(all_metrics, json_path, csv_path)

    assert json.loads(json_path.read_text()) == all_metrics
    assert "m1" in csv_path.read_text()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_collect_all_metrics.py -v`
Expected: FAIL — `ImportError: cannot import name 'aggregate_metrics'`

- [ ] **Step 3: Rewrite `scripts/collect_all_metrics.py`**

Replace its contents with:

```python
import json
import csv
from pathlib import Path

RESULTS_DIR = Path("results")
OUTPUT_JSON = RESULTS_DIR / "metrics_all_models.json"
OUTPUT_CSV = RESULTS_DIR / "metrics_all_models.csv"


def aggregate_metrics(results_dir):
    """Scan results_dir/<model_tag>/test_metrics.json for every model
    directory and return {model_tag: metrics_dict}. This is the one place
    that reads real, on-disk metrics -- no hardcoded numbers anywhere.
    """
    results_dir = Path(results_dir)
    all_metrics = {}
    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        metrics_file = model_dir / "test_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                all_metrics[model_dir.name] = json.load(f)
    return all_metrics


def write_outputs(all_metrics, json_path, csv_path):
    json_path = Path(json_path)
    csv_path = Path(csv_path)

    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore"])
        for model, scores in all_metrics.items():
            writer.writerow([
                model,
                scores["ROUGE-1"],
                scores["ROUGE-2"],
                scores["ROUGE-L"],
                scores["BERTScore"],
            ])


def main():
    all_metrics = aggregate_metrics(RESULTS_DIR)
    write_outputs(all_metrics, OUTPUT_JSON, OUTPUT_CSV)
    print("Saved final metrics table:")
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_collect_all_metrics.py -v`
Expected: `3 passed`

- [ ] **Step 5: Delete the redundant/fabricated scripts**

```bash
git rm scripts/save_all_metrics.py scripts/collect_baseline_scores.py
```

- [ ] **Step 6: Commit**

```bash
git add scripts/collect_all_metrics.py tests/test_collect_all_metrics.py
git commit -m "Consolidate metrics collection onto one real aggregator; remove hardcoded/duplicate scripts"
```

---

### Task 15: `scripts/evaluate_sanity.py` — guard against its numbers leaking into real results

**Files:**
- Modify: `scripts/evaluate_sanity.py`

- [ ] **Step 1: Add a clear warning to `scripts/evaluate_sanity.py`**

Replace its contents with:

```python
import sys, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.metrics import compute_metrics


def main():
    # Dummy predictions (simulate model output)
    predictions = [
        "The prime minister addressed the nation regarding COVID.",
        "The economy is expected to recover after reforms."
    ]

    # Ground truth summaries
    references = [
        "PM spoke to the nation about COVID.",
        "Economic recovery is expected after reforms."
    ]

    metrics = compute_metrics(predictions, references)

    print("Sanity Evaluation Metrics (2-example dummy check only -- NOT real")
    print("results; never copy these numbers into results/*/test_metrics.json):")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/evaluate_sanity.py
git commit -m "Warn that evaluate_sanity.py output is a dummy check, not real results"
```

---

### Task 16: Remove fabricated/duplicated results files

**Files:**
- Delete: `results/phase3/`
- Delete: `results/phase4/novel/test_metrics.json`, `results/phase4/novel/test_predictions.jsonl` (if present)
- Delete: `results/longt5/`, `results/primera/`, `results/flan_t5_xl/`, `results/gemma/`, `results/led/`, `results/mistral/`, `results/mixtral/`, `results/qwen/`
- Delete: `results/newssumm_baselines_scores.csv`
- Delete: `results/metrics_all_models.json`, `results/metrics_all_models.csv`
- Delete: `results/plots/` (bar charts rendered from the fabricated numbers)

**Interfaces:**
- None — this is a pure cleanup task. Nothing in the rewritten scripts (Tasks 9, 10, 14) reads from any of these paths; they all write to `results/<model_tag>/` where `model_tag` is `sa_longt5` or `longt5_baseline`.

- [ ] **Step 1: Confirm nothing in the rewritten code still references the old paths**

Run: `grep -rn "results/phase3\|results/phase4/novel\|results/longt5/\|results/primera\|results/flan_t5_xl\|results/gemma\|results/led\|results/mistral\|results/mixtral\|results/qwen\|newssumm_baselines_scores" src/ scripts/ phase4/ --include="*.py"`
Expected: no matches (if any script still references these paths, fix that script before deleting the directory it depends on)

- [ ] **Step 2: Delete the fabricated/duplicated results**

```bash
git rm -r results/phase3 results/longt5 results/primera results/flan_t5_xl results/gemma results/led results/mistral results/mixtral results/qwen results/plots results/newssumm_baselines_scores.csv results/metrics_all_models.json results/metrics_all_models.csv
git rm -r results/phase4/novel 2>/dev/null || true
```

- [ ] **Step 3: Commit**

```bash
git commit -m "Remove fabricated/duplicated results files (identical sanity-dummy numbers, hardcoded metrics, contradicting baselines)"
```

---

### Task 17: Update reporting docs — fix inconsistencies, add honest caveats

**Files:**
- Modify: `README.md`
- Modify: `newssumm_benchmark.md`
- Modify: `docs/IEEE_Paper.md`

**Interfaces:**
- None — documentation only. No code depends on these files.

- [ ] **Step 1: Fix `README.md`**

Apply these specific edits to `README.md`:

1. Repository structure section: update `scripts/split_dataset.py` comment from `# Train/val split (90/10)` to `# Train/val/test split (70/15/15, cluster-level)`; remove the `save_all_metrics.py` and `collect_baseline_scores.py` lines (deleted in Task 14); add `build_newssumm_pp.py` under `scripts/`.

2. "Novel Model" section: replace the "Training loss" bullet with:
   ```
   - **Training loss:** `L = L_summarization + λ * L_salience` (λ = 1.0), with
     `L_salience` a real BCE loss against greedy ROUGE-oracle sentence labels
     (see `src/salience_oracle.py`) -- not a placeholder.
   ```

3. Replace the entire "📊 Benchmark Results" section (the 10-row table and the sentence below it) with:
   ```markdown
   ## 📊 Benchmark Results

   Evaluated on the **NewsSumm++ test split** using ROUGE-1, ROUGE-2,
   ROUGE-L (rougeLsum, F1), and BERTScore (F1).

   > **Status:** this repository's own pipeline (`scripts/generate_test_predictions.py`,
   > `scripts/generate_novel_test_predictions.py`, `scripts/evaluate_test_split.py`,
   > `scripts/collect_all_metrics.py`) is fixed and unit-tested (see `tests/`), but has
   > not been re-run end-to-end on the full dataset in this environment -- there is no
   > GPU/large dataset available here. The author's separate report presents numbers
   > produced in Google Colab; those are **not yet reproduced from this repo** and
   > should be treated as preliminary until `results/<model_tag>/test_metrics.json`
   > exists for each model from a real run of the scripts above.
   >
   > Run the pipeline (see Usage below) to populate `results/metrics_all_models.csv`.
   ```

4. Fix the split-ratio line under "1. Split the Dataset": change
   `Generates data/splits/train.jsonl and data/splits/val.jsonl (90/10 split).`
   to
   `Generates data/splits/{train,val,test}.jsonl (70/15/15 cluster-level split).`

5. Fix the params claim: in the removed benchmark table this was `237M` for
   SA-LongT5, which is impossible against a 248M backbone. Since the table is
   now replaced with the status note above, no further edit is needed here,
   but add this sentence directly under the "Novel Model" architecture bullets:
   ```
   Total parameters = the 248M LongT5-base backbone plus one `nn.Linear(768, 1)`
   salience head (~769 parameters) -- there are no other trainable modules in
   this architecture, so total parameters are ≈248M, not 237M as previously
   (and impossibly) stated.
   ```

- [ ] **Step 2: Fix `newssumm_benchmark.md`**

Replace section "2. Benchmark Results Table" (the identical-0.4381-everywhere
table plus its warning note) with:

```markdown
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
```

- [ ] **Step 3: Fix `docs/IEEE_Paper.md`**

1. In the "Hyperparameters" table, change the `Document separator` row value
   from `` `[SEP]` `` to `` `</s>` (tokenizer's real end-of-sequence token) ``.

2. Change the `λ (auxiliary salience loss)` row value from
   `0.1 (reserved for future use)` to `1.0 (now actually computed via
   greedy ROUGE-oracle labels, see src/salience_oracle.py)`.

3. In "VI. Results", add this paragraph directly above the results table:
   ```markdown
   > **Status:** the model, training, and generation code this table's
   > "Proposed" row was originally produced by has since been rewritten (see
   > the accompanying repository) to fix a bug where generation bypassed the
   > salience-aware forward pass entirely and used an untrained model. The
   > baseline rows below reflect ad hoc exploratory runs on small subsets with
   > per-model decoding settings, not one unified evaluation protocol. Treat
   > this table as preliminary pending a full, unified re-run of the fixed
   > pipeline.
   ```

4. In "III. Dataset", change `an 80/10/10 split` (both occurrences) to
   `a 70/15/15 split`, and update the bullet list to `Train: 70%`,
   `Validation: 15%`, `Test: 15%`.

- [ ] **Step 4: Commit**

```bash
git add README.md newssumm_benchmark.md docs/IEEE_Paper.md
git commit -m "Fix split-ratio/params/separator inconsistencies and add honest status notes to reporting docs"
```

---

### Task 18: End-to-end smoke test on the tiny sample dataset

**Files:**
- Test: `tests/test_end_to_end_smoke.py`

**Interfaces:**
- Consumes: everything built in Tasks 2-10 (`build_cluster_input`, `compute_oracle_labels`, `SalienceAwareLongT5`, `prepare_batch`, `generate_predictions`, `compute_split_indices`/`split_samples`).
- Produces: confidence that the full split -> train (a few steps) -> generate -> evaluate chain runs without crashing, using a tiny LongT5 config (never the real 990MB backbone) and the `data/sample.jsonl` fixture from Task 1.

- [ ] **Step 1: Write the smoke test**

Create `tests/test_end_to_end_smoke.py`:

```python
import json

import torch
from torch.optim import AdamW
from transformers import LongT5Config, LongT5ForConditionalGeneration, T5Tokenizer

from src.novel_newssumm_model import SalienceAwareLongT5
from src.sentence_utils import build_cluster_input
from src.salience_oracle import compute_oracle_labels
from src.metrics import compute_metrics
from scripts.split_dataset import split_samples
from scripts.generate_novel_test_predictions import generate_predictions


def _tiny_base_model(vocab_size):
    config = LongT5Config(
        vocab_size=vocab_size,
        d_model=8,
        d_kv=4,
        d_ff=16,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        local_radius=4,
        global_block_size=4,
        feed_forward_proj="relu",
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    return LongT5ForConditionalGeneration(config)


def test_full_pipeline_runs_on_sample_data_with_a_tiny_model(monkeypatch):
    with open("data/sample.jsonl") as f:
        samples = [json.loads(line) for line in f]

    train, val, test = split_samples(samples, train_ratio=0.70, val_ratio=0.15)
    assert len(train) + len(val) + len(test) == len(samples)
    assert len(test) > 0

    tokenizer = T5Tokenizer.from_pretrained("google/long-t5-tglobal-base")
    model = SalienceAwareLongT5.from_base_model(_tiny_base_model(tokenizer.vocab_size))

    device = torch.device("cpu")
    optimizer = AdamW(model.parameters(), lr=1e-3)
    model.train()

    for sample in train[:2]:
        input_text, sentences = build_cluster_input(sample["documents"], tokenizer)
        inputs = tokenizer(
            input_text, truncation=True, padding="max_length", max_length=64, return_tensors="pt"
        )
        target = tokenizer(
            sample["summary"], truncation=True, padding="max_length", max_length=16, return_tensors="pt"
        )
        labels = target["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        oracle_labels = compute_oracle_labels(sentences, sample["summary"])
        salience_labels = torch.tensor([oracle_labels], dtype=torch.float)

        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
            salience_labels=salience_labels,
        )
        output["loss"].backward()
        optimizer.step()
        optimizer.zero_grad()
        assert torch.isfinite(output["loss"])

    model.eval()
    predictions = list(generate_predictions(
        model, tokenizer, test, device, max_input_length=64, max_output_length=8, num_beams=2
    ))
    assert len(predictions) == len(test)
    for record in predictions:
        assert isinstance(record["generated_summary"], str)

    monkeypatch.setattr(
        "src.metrics.rouge.compute",
        lambda predictions, references: {
            "rouge1": 0.1, "rouge2": 0.05, "rougeL": 0.02, "rougeLsum": 0.08
        },
    )
    monkeypatch.setattr(
        "src.metrics.bertscore.compute",
        lambda predictions, references, lang: {"f1": [0.5] * len(predictions)},
    )
    metrics = compute_metrics(
        [r["generated_summary"] for r in predictions],
        [r["reference_summary"] for r in predictions],
    )
    assert metrics["ROUGE-L"] == 0.08
```

- [ ] **Step 2: Run the smoke test**

Run: `python -m pytest tests/test_end_to_end_smoke.py -v`
Expected: `1 passed` (this exercises the real tokenizer's vocab size against the tiny model, real sentence splitting, real oracle labeling, the real `SalienceAwareLongT5` forward/generate path, and the real `generate_predictions` generator end to end)

- [ ] **Step 3: Run the full test suite one more time**

Run: `python -m pytest -v`
Expected: every test across all previous tasks plus this one passes (roughly 30 tests total)

- [ ] **Step 4: Commit**

```bash
git add tests/test_end_to_end_smoke.py
git commit -m "Add end-to-end smoke test covering split -> train -> generate -> evaluate on sample data"
```

---

## Self-Review Notes

- **Spec coverage:** every numbered item in the design spec (model architecture, salience oracle, label masking, checkpoint loading, unified decoding, dataset pipeline, split ratio, reporting caveats, low-priority cleanups) maps to a task above. The "out of scope" items from the spec (re-running training for new numbers, building efficiency/ablation/redundancy harnesses, revoking the token) are deliberately not tasked here.
- **Type/signature consistency checked:** `build_cluster_input(documents, tokenizer) -> (str, list[str])` is used identically in Tasks 2, 8, 9, 10, 11, 18. `SalienceAwareLongT5.forward(..., salience_labels=None)` and `.generate(...)` signatures introduced in Task 4 are used unchanged in Tasks 8, 9, 18. `generate_predictions(model, tokenizer, samples, device, ...)` has the same shape in Tasks 9, 10, 18 (differing only in which kind of model object is passed).
- **No placeholders:** every step above has complete, verified-runnable code (the tiny-`LongT5Config` model, the literal-`</s>`-in-text tokenization behavior, and `pool_sentences`'s exact output were all hand-verified against real `torch`/`transformers` output before being written into this plan).
