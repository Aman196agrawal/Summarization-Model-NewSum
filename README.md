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
│   ├── split_dataset.py           # Train/val split (90/10)
│   ├── prepare_inputs.py          # Build [DOC]-delimited model inputs
│   ├── generate_test_predictions.py       # Generate predictions (LongT5 baseline)
│   ├── generate_novel_test_predictions.py # Generate predictions (Proposed model)
│   ├── evaluate_test_split.py     # Evaluate predictions against references
│   ├── evaluate_sanity.py         # Sanity check with dummy data
│   ├── collect_baseline_scores.py # Collect baseline metrics into CSV
│   ├── collect_all_metrics.py     # Aggregate all model metrics
│   ├── save_all_metrics.py        # Save final metrics to JSON/CSV
│   └── plot_baseline_scores.py    # Bar chart plots per metric
│
├── phase4/
│   ├── novel_model_spec.md        # Architecture specification & math formulation
│   └── train_novel_model.py       # Training loop for proposed model
│
├── results/
│   ├── metrics_all_models.json    # All baseline metrics (JSON)
│   ├── metrics_all_models.csv     # All baseline metrics (CSV)
│   ├── newssumm_baselines_scores.csv
│   ├── phase3/longt5/test_metrics.json
│   └── phase4/novel/test_metrics.json
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
- **Salience head:** Linear layer → sigmoid, predicting per-sentence importance
- **Training loss:** `L = L_summarization + λ * L_salience`

```python
from src.novel_newssumm_model import SalienceAwareLongT5

model = SalienceAwareLongT5(model_name="google/long-t5-tglobal-base")
```

See [`phase4/novel_model_spec.md`](phase4/novel_model_spec.md) for the full mathematical formulation.

---

## 📊 Benchmark Results

Evaluated on the **NewsSumm test split** using ROUGE-1, ROUGE-2, ROUGE-L (F1), and BERTScore (F1).

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
10   | SA-LongT5 (Proposed)| Hierarchical | 237M   | 4096    | Fine-tuned  | 0.3196  | 0.1589  | 0.2342  | 0.8764

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
Generates `data/splits/train.jsonl` and `data/splits/val.jsonl` (90/10 split).

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
Output: `results/phase4/novel/test_predictions.jsonl`

### 5. Evaluate
```bash
python scripts/evaluate_test_split.py \
  --predictions results/phase4/novel/test_predictions.jsonl \
  --output results/phase4/novel/test_metrics.json
```

### 6. Plot Baseline Scores
```bash
python scripts/plot_baseline_scores.py
```
Saves bar charts to `results/plots/`.

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
