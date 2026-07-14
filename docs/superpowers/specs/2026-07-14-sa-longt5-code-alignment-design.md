# SA-LongT5 Code Alignment — Design

## Background

A mentor code review (see conversation history) found that the committed pipeline in this
repo does not do what it claims: the novel model's `generate()` call bypasses its own custom
`forward()`, no checkpoint is ever loaded, pad tokens are unmasked in the loss, the salience
loss is never computed, `[SEP]`/`[DOC]` are not real tokenizer tokens, and the benchmark
numbers in `README.md` / `newssumm_benchmark.md` / `results/*` are contradictory or
duplicated placeholder values.

The author has a separate IEEE-style report
(`Salience-Aware Hierarchical LongT5 for Long-Context Multi-Document News Summarization.docx`)
that will be presented to professors. Professors will not run the code live but may read it.
The report's Section IV formalizes a specific architecture (sentence-level salience →
softmax over the whole cluster → global weighted sum `Hdoc` → residual fusion `H̃ = H + Hdoc`
→ decode from `H̃`) that is simpler than, and inconsistent with, the entity/discourse/fusion
modules currently in `src/novel_newssumm_model.py`.

Cross-referencing the author's Google Drive notebooks confirmed:
- The **NewsSumm++ dataset construction pipeline** (dedup, near-duplicate filtering,
  SentenceBERT + MiniBatchKMeans clustering, category mapping, silhouette score) is real,
  working code (`more_about_dataset_filtering.ipynb`), not present anywhere in this repo.
- The **SA-LongT5 model** has no working implementation anywhere in the Drive matching
  either the repo's current code or the report's math. The only model code found
  (`Phase 2.ipynb`) is more primitive than the repo's current version and its own
  `generate()` call bypasses the salience head — the same bug the mentor flagged.
- Tables IV (efficiency), V (input-length breakdown), VI (ablation), and the redundancy/
  hallucination percentage claims have no corresponding code anywhere in the Drive.
- A live GitHub personal access token was found hardcoded in `Phase 2.ipynb` — flagged
  separately to the user for revocation; out of scope for this spec.

## Decisions already made with the user

1. **Scope:** code-correctness pass only. No GPU/data available here to produce new
   verified numbers; a tiny synthetic sample dataset will be added so every script runs
   end-to-end for verification.
2. **Source of truth for the model:** the docx report's Section IV math, not the current
   repo code and not the earlier from-scratch redesign discussed before the report surfaced.
3. **Salience supervision:** implement real pseudo-oracle labels via greedy ROUGE-oracle
   sentence extraction, and compute the real `BCE` loss the report specifies.
4. **Unverified results (Table III proposed row, IV, V, VI, redundancy/hallucination %):**
   fix the code to be honest and internally consistent, and add clear caveats in the repo's
   own docs that these figures are pending full in-repo reproduction — do not delete them,
   do not fabricate replacements, and do not silently claim them as verified.

## Architecture (matches report Section IV)

`src/novel_newssumm_model.py` is rewritten. The entity-graph module, discourse-hierarchy
module, and 3-way concatenation fusion currently in the repo are removed — they appear
nowhere in the report's formulas and have no working implementation anywhere in the
author's Drive, so keeping them would just be another code/paper mismatch.

- **Input construction:** documents are split into sentences and joined with the
  tokenizer's real `</s>` token (not literal `[SEP]`/`[DOC]` text). This is a single
  shared function used by training, generation, and the baseline script, so all three
  use identical input formatting.
- **Encoder:** LongT5 encoder produces token hidden states; `</s>` positions are located
  in `input_ids` to mean-pool token states into sentence embeddings `h_ij` (sentence `j`
  of document `i`, flattened across the whole cluster into one sequence for the
  cluster-wide softmax).
- **Salience scoring:** `a_ij = W_s h_ij + b_s` (`nn.Linear(hidden, 1)`), then
  `α_ij = softmax` over **all** sentences in the cluster (matches the report's
  `softmax_{p,x}` normalization, not a per-document or per-token softmax).
- **Multi-source fusion / hierarchical encoding:** `Hdoc = Σ_ij α_ij · h_ij` (single
  global vector per cluster), broadcast and added residually to every token position:
  `H̃ = H + Hdoc`. This residual add is what the report calls "hierarchical" (local +
  global) and is exactly what Table VI's "without hierarchical encoding" ablation arm
  would toggle off.
- **Constrained attention / decoding:** the LongT5 decoder cross-attends to `H̃` instead
  of raw `H` — implemented by passing `H̃` as `encoder_outputs.last_hidden_state`.
- **`generate()` method on `SalienceAwareLongT5`:** runs the identical
  encode → pool → salience → `Hdoc` → `H̃` path as `forward()`, then delegates to
  `self.base_model.generate(encoder_outputs=..., ...)`. This is the fix for the
  mentor's #1 finding — beam search now actually depends on the salience-aware states.
- **Salience loss:** `z_ij` pseudo-labels from greedy ROUGE-oracle sentence extraction
  (new `src/salience_oracle.py`), `L_sal = BCE(z_ij, α_ij)` computed for real,
  `λ = 1.0` (Table II), `L = L_sum + λ·L_sal`.

## Training and generation script fixes

- `phase4/train_novel_model.py`: mask pad-token labels with `-100` before the loss;
  save a real checkpoint at the end of a run.
- `scripts/generate_novel_test_predictions.py`: load a saved checkpoint before eval
  (raise a clear error if none exists — no more silently running untrained weights);
  call the new `model.generate(...)` instead of `model.base_model.generate(...)`.
- `scripts/generate_test_predictions.py` (baseline) and the novel script are unified on
  the same `</s>`-joined input construction and the same decoding settings
  (`num_beams=4`, matching Table II), so any ROUGE gap reflects the architecture, not
  decoding differences.
- `scripts/prepare_inputs.py` uses the same shared input-builder (removes the third,
  differently-formatted separator style).

## Dataset pipeline — new, ports real notebook code

New `scripts/build_newssumm_pp.py`, porting the working pipeline found in
`more_about_dataset_filtering.ipynb`:
1. Drop null article/summary rows.
2. Drop exact-duplicate articles.
3. Text normalization.
4. Near-duplicate filtering via TF-IDF cosine similarity > 0.90.
5. SentenceBERT (`all-MiniLM-L6-v2`) embeddings + `MiniBatchKMeans` clustering.
6. Filter to clusters with 7–10 documents.
7. Category mapping to the 22 fixed categories (exact list ported from the notebook).

`docs/data_pipeline.md` is rewritten to describe this pipeline instead of the old
phase-1-only (category+date grouping) description, which is now stale.

`scripts/split_dataset.py` changes from 80/10/10 to cluster-level **70/15/15**, matching
the report and Table II. Every "90/10"/"80/10/10" mention in `README.md` is corrected to
70/15/15.

## Reporting: honest caveats, not deletion

- `README.md`, `newssumm_benchmark.md`, `docs/IEEE_Paper.md` keep their existing numbers
  but gain a clear, visible note: Table III's proposed-model row, Tables IV/V/VI, and the
  redundancy/hallucination percentages are from external (Colab) experimentation and are
  pending full reproduction from this repo's own pipeline — not a claim that they're
  fabricated, but not a claim that this repo currently reproduces them either.
- Plain inconsistencies are still fixed regardless of the caveat (these are arithmetic/
  consistency bugs, not disputed research results):
  - SA-LongT5 params: 237M is impossible given a 248M backbone — replaced with an
    honestly-computed figure (backbone + actual head sizes from the new, simpler
    architecture).
  - The four different LongT5 baseline numbers across `README.md`,
    `newssumm_benchmark.md`/`metrics_all_models.csv`, `results/phase3/longt5/test_metrics.json`,
    and `newssumm_baselines_scores.csv` are consolidated to one canonical location with a
    note on provenance, rather than silently left contradicting each other.
  - `src/metrics.py` reports `rougeLsum` instead of sentence-level `rougeL` (summarization
    convention).
- `scripts/save_all_metrics.py`'s hardcoded metrics dict is removed; `collect_all_metrics.py`
  remains the one real aggregator that reads from disk.

## Low-priority cleanups (from the mentor review, still applicable)

- `NewsSummDataset` subclasses `torch.utils.data.Dataset`.
- Dead `hidden_states=None, attentions=None` passthrough in `BaseModelOutput` removed.
- `evaluate_sanity.py` gets a one-line comment noting its numbers are a dummy sanity
  check, never real results — guards against the exact mistake that produced the
  identical 0.4381 figures across `results/*/metrics.json`.

## Out of scope

- Actually re-running training/evaluation to produce new verified numbers (no GPU/data
  here).
- Building efficiency-profiling, ablation, or redundancy-scoring harnesses to back
  Tables IV/V/VI/redundancy% — flagged as pending, not built, since no working code for
  these exists anywhere to port from.
- Revoking/rotating the leaked GitHub token — user's action, already flagged separately.
