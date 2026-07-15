# NewsSumm Multi-Document Summarization

A research project for **multi-document news summarization** on the [NewsSumm dataset](https://github.com/Aman196agrawal/Summarization-Model-NewSum), featuring baseline model evaluation and a novel **Salience-Aware LongT5** architecture.

---

## 📌 Project Overview

This project implements and benchmarks multiple summarization models on the NewsSumm dataset — a collection of multi-source Indian news article clusters paired with human-written reference summaries.

**Key contributions:**
- Data pipeline for constructing multi-document clusters from raw news data
- Evaluation of 9 baseline models (encoder-decoder and LLM-based)
- A novel **Salience-Aware Hierarchical LongT5** model that predicts sentence importance to guide summary generation
- Comprehensive benchmark using ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore

---

## 🗂️ Repository Structure

```
Summarization-Model-NewSum/
├── src/
│   ├── newssumm_dataset.py        # Dataset loader (JSONL format)
│   ├── novel_newssumm_model.py    # SalienceAwareLongT5 model definition
│   └── metrics.py                 # ROUGE + BERTScore evaluation utilities
│
├── scripts/
│   ├── build_newssumm_pp.py       # Build NewsSumm++ clustered dataset
│   ├── split_dataset.py           # Train/val/test split (70/15/15, cluster-level)
│   ├── prepare_inputs.py          # Preview </s>-joined model input construction
│   ├── generate_test_predictions.py       # Generate predictions (LongT5 baseline)
│   ├── generate_novel_test_predictions.py # Generate predictions (Proposed model)
│   ├── evaluate_test_split.py     # Evaluate predictions against references
│   ├── evaluate_sanity.py         # Sanity check with dummy data
│   └── collect_all_metrics.py     # Aggregate all model metrics
│
├── phase4/
│   ├── novel_model_spec.md        # Architecture specification & math formulation
│   └── train_novel_model.py       # Training loop for proposed model
│
├── results/                        # created by running the pipeline (see Usage below) --
│                                    # not checked into the repo; results/<model_tag>/test_metrics.json
│                                    # appears here for each model once you've run the scripts
│
├── docs/
│   ├── data_pipeline.md           # Data preprocessing pipeline documentation
│   └── IEEE_Paper.md              # Paper draft
│
├── newssumm_benchmark.md          # Full benchmark results and analysis
└── data/                          # Raw and processed data (gitignored)
```

---

## 🧠 Novel Model: Salience-Aware LongT5

The proposed model extends LongT5 with a **salience prediction head** that scores sentence importance during encoding. This guides the decoder to focus on key information.

**Architecture:**
- **Backbone:** `google/long-t5-tglobal-base` (encoder-decoder)
- **Salience head:** Linear layer → softmax (normalized across all sentences in the cluster), predicting relative per-sentence importance
- **Training loss:** `L = L_summarization + λ * L_salience` (λ = 1.0), with
  `L_salience` a real BCE loss against greedy ROUGE-oracle sentence labels
  (see `src/salience_oracle.py`) -- not a placeholder.

Total parameters = the 248M LongT5-base backbone plus one `nn.Linear(768, 1)`
salience head (~769 parameters) -- there are no other trainable modules in
this architecture, so total parameters are ≈248M, not 237M as previously
(and impossibly) stated.

```python
from src.novel_newssumm_model import SalienceAwareLongT5

model = SalienceAwareLongT5(model_name="google/long-t5-tglobal-base")
```

See [`phase4/novel_model_spec.md`](phase4/novel_model_spec.md) for the full mathematical formulation.

---

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

> See [`newssumm_benchmark.md`](newssumm_benchmark.md) for detailed quantitative and qualitative analysis.

---

## ⚙️ Setup

### Requirements

```bash
pip install torch transformers evaluate rouge_score bert_score pandas matplotlib
```

### Data

Place your raw data file at:
```
data/newssumm_phase1.jsonl
```

Each line should be a JSON object with:
```json
{
  "cluster_id": "...",
  "documents": ["article 1 text", "article 2 text", ...],
  "summary": "reference summary"
}
```

---

## 🚀 Usage

### 1. Split the Dataset
```bash
python scripts/split_dataset.py
```
Generates `data/splits/{train,val,test}.jsonl` (70/15/15 cluster-level split).

### 2. Load the Dataset
```python
from src.newssumm_dataset import NewsSummDataset

dataset = NewsSummDataset("data/splits/train.jsonl")
sample = dataset[0]
print(sample["cluster_id"], sample["documents"], sample["summary"])
```

### 3. Train the Proposed Model
```bash
python phase4/train_novel_model.py
```

### 4. Generate Predictions (Proposed Model)
```bash
python scripts/generate_novel_test_predictions.py
```
Output: `results/sa_longt5/test_predictions.jsonl`
(Requires a trained checkpoint at `results/sa_longt5/checkpoints/model_final.pt` -- run step 3 first.)

### 5. Evaluate
```bash
python scripts/evaluate_test_split.py \
  --predictions results/sa_longt5/test_predictions.jsonl \
  --output results/sa_longt5/test_metrics.json
```

---

## 📁 Data Pipeline

NewsSum (348K)
      ↓
Cleaning
      ↓
Clustering
      ↓
NewsSum++ Full (~248K)
      ↓
Filtering (7–10 docs)
      ↓
NewsSum++ Eval (~12K)
      ↓
Model Benchmarking

---

## 📄 Paper

A draft IEEE-format paper is available at [`docs/IEEE_Paper.md`](docs/IEEE_Paper.md).

---

## 📬 Author

**Aman Agrawal** — [@Aman196agrawal](https://github.com/Aman196agrawal)
