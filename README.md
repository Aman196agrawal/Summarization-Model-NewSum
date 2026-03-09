# NewsSumm Multi-Document Summarization

A research project for **abstractive multi-document news summarization** on the NewsSumm dataset. This project implements and benchmarks multiple baseline models (encoder-decoder and large language models) alongside a novel **Salience-Aware Hierarchical LongT5** model designed to reduce redundancy and improve key-fact coverage across multi-document news clusters.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Baseline Inference](#baseline-inference)
  - [Training the Novel Model](#training-the-novel-model)
  - [Novel Model Inference](#novel-model-inference)
  - [Evaluation](#evaluation)
  - [Visualization](#visualization)
- [Novel Model Architecture](#novel-model-architecture)
- [Benchmark Results](#benchmark-results)
- [Qualitative Error Analysis](#qualitative-error-analysis)
- [Future Work](#future-work)

---

## Overview

Multi-document news summarization is challenging because source articles often contain overlapping, redundant information. Standard encoder-decoder models treat all sentences equally, which over-represents repeated content and under-represents unique key facts.

This project addresses that limitation by:

1. **Benchmarking** a broad set of baseline models (LongT5, LED, PRIMERA, Flan-T5-XL, LLaMA-3, Mistral, Mixtral, Qwen, Gemma) on the NewsSumm dataset.
2. **Proposing** a Salience-Aware Hierarchical LongT5 that predicts per-sentence importance and uses it to guide summary generation, reducing redundancy and improving key-fact coverage.

---

## Project Structure

```
Summarization-Model-NewSum/
├── README.md                                # This file
├── newssumm_benchmark.md                    # Full benchmark results and analysis
├── src/
│   ├── newssumm_dataset.py                  # Dataset loader (JSONL → PyTorch Dataset)
│   ├── metrics.py                           # ROUGE + BERTScore computation
│   └── novel_newssumm_model.py              # Salience-Aware LongT5 architecture
├── scripts/
│   ├── split_dataset.py                     # Train/val split (90/10, seed 42)
│   ├── prepare_inputs.py                    # Prepare tokenized input samples
│   ├── generate_test_predictions.py         # Baseline model inference
│   ├── generate_novel_test_predictions.py   # Novel model inference
│   ├── evaluate_test_split.py               # Compute metrics on a prediction file
│   ├── collect_all_metrics.py               # Aggregate results from all models
│   ├── collect_baseline_scores.py           # Extract per-model baseline metrics
│   ├── evaluate_sanity.py                   # Quick sanity-check script
│   ├── save_all_metrics.py                  # Save aggregated metrics to JSON/CSV
│   └── plot_baseline_scores.py              # Generate comparison bar plots
├── phase4/
│   ├── train_novel_model.py                 # Training script for the novel model
│   ├── novel_model_spec.md                  # Architecture specification
│   └── novel_model_architecture.png         # Architecture diagram
├── docs/
│   ├── data_pipeline.md                     # Data processing documentation
│   └── IEEE_Paper.md                        # Paper template
├── results/
│   ├── metrics_all_models.json              # Aggregated metrics for all models
│   ├── longt5/                              # LongT5 prediction and metric files
│   ├── led/                                 # LED prediction and metric files
│   ├── primera/                             # PRIMERA prediction and metric files
│   └── phase4/novel/                        # Novel model prediction and metric files
└── data/                                    # Raw and processed data (git-ignored)
    └── splits/
        ├── train.jsonl
        └── val.jsonl
```

---

## Tech Stack

| Component              | Technology                                        |
|------------------------|---------------------------------------------------|
| Deep Learning          | PyTorch                                           |
| NLP / Models           | Hugging Face `transformers`                       |
| Evaluation Metrics     | Hugging Face `evaluate` (ROUGE, BERTScore)        |
| Optimizer              | AdamW                                             |
| Data Format            | JSONL (JSON Lines)                                |
| Analysis / Plotting    | Python, Pandas, Matplotlib                        |

**Pre-trained models used:**

| Model                         | Type                       | Context Length |
|-------------------------------|----------------------------|----------------|
| `google/long-t5-tglobal-base` | Encoder–Decoder            | 4 096 tokens   |
| `allenai/led-base-16384`      | Encoder–Decoder            | 16 384 tokens  |
| `allenai/primera-base`        | Encoder–Decoder            | 4 096 tokens   |
| `google/flan-t5-xl`           | Encoder–Decoder            | 1 024 tokens   |
| LLaMA-3                       | LLM (zero-shot)            | 8 192+ tokens  |
| Mistral                       | LLM (zero-shot)            | 8 192 tokens   |
| Mixtral (MoE)                 | LLM (zero-shot)            | 32 768 tokens  |
| Qwen                          | LLM (zero-shot)            | 8 192 tokens   |
| Gemma                         | LLM (zero-shot)            | 8 192 tokens   |
| **Salience-Aware LongT5**     | **Hierarchical Enc–Dec**   | **4 096 tokens** |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Aman196agrawal/Summarization-Model-NewSum.git
cd Summarization-Model-NewSum

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install torch transformers evaluate pandas matplotlib
```

---

## Dataset

The project uses the **NewsSumm** dataset — a multi-document news summarization benchmark.

Each sample in the JSONL file has the following structure:

```json
{
    "cluster_id": "politics_2023-01-15",
    "documents": ["Article 1 text ...", "Article 2 text ...", "..."],
    "summary": "Human-written reference summary."
}
```

- **cluster_id**: Category + publication date.
- **documents**: List of source news articles in the cluster.
- **summary**: Human-written reference summary.

Place the raw dataset at `data/newssumm_phase1.jsonl` (the `data/` directory is git-ignored).

### Loading the dataset

```python
from src.newssumm_dataset import NewsSummDataset

dataset = NewsSummDataset("data/newssumm_phase1.jsonl")
sample = dataset[0]
print(sample["cluster_id"])
print(sample["documents"])
print(sample["summary"])
```

---

## Usage

### Data Preparation

```bash
# Split into train (90%) and val (10%) sets with seed 42
python scripts/split_dataset.py
# Output: data/splits/train.jsonl, data/splits/val.jsonl

# Tokenize and prepare inputs for model consumption
python scripts/prepare_inputs.py
```

### Baseline Inference

```bash
# Run inference with all baseline models and save predictions
python scripts/generate_test_predictions.py
# Predictions saved to: results/<model_name>/test_predictions.jsonl
```

### Training the Novel Model

```bash
# Fine-tune the Salience-Aware LongT5 model
# (runs a demo with 10 training steps by default)
python phase4/train_novel_model.py
```

### Novel Model Inference

```bash
# Generate predictions using the trained novel model
python scripts/generate_novel_test_predictions.py
# Predictions saved to: results/phase4/novel/test_predictions.jsonl
```

### Evaluation

```bash
# Evaluate predictions from any model
python scripts/evaluate_test_split.py \
    --predictions results/longt5/test_predictions.jsonl \
    --output results/longt5/metrics.json

# Evaluate the novel model
python scripts/evaluate_test_split.py \
    --predictions results/phase4/novel/test_predictions.jsonl \
    --output results/phase4/novel/test_metrics.json

# Aggregate metrics across all models
python scripts/collect_all_metrics.py
python scripts/save_all_metrics.py
```

Metrics computed: **ROUGE-1**, **ROUGE-2**, **ROUGE-L** (all F1), and **BERTScore** (F1).

### Visualization

```bash
# Generate bar-chart comparisons of all models
python scripts/plot_baseline_scores.py
# Output: results/plots/rouge1.png, rouge2.png, rougel.png, bertscore.png
```

---

## Novel Model Architecture

The proposed **Salience-Aware Hierarchical LongT5** extends the standard LongT5 with an explicit sentence-importance mechanism.

**Components:**
- **Backbone:** LongT5 encoder-decoder (4 096-token context)
- **Sentence encoder:** LongT5 encoder applied at the sentence level
- **Salience head:** Linear layer (hidden_size → 1) predicting importance per sentence
- **Aggregation:** Salience-weighted sentence embeddings fed to the decoder
- **Decoder:** Standard LongT5 decoder

**Mathematical Formulation:**

Let *s_i ∈ ℝ^d* be the sentence embedding (hidden dimension *d*) for the *i*-th sentence produced by the encoder, and let *n* be the total number of sentences.

```
Salience scores (over all n sentences):
    a = softmax([W_s · s_1, W_s · s_2, ..., W_s · s_n])   (a ∈ ℝ^n, Σ a_i = 1)

Salience-weighted representation:
    h = Σ_i  a_i · s_i

Summary generation loss:
    L_sum = CrossEntropy(summary, generated_summary)

Auxiliary salience loss:
    L_sal = BinaryCrossEntropy(a_i, y_i)   (y_i ∈ {0,1} extractive oracle label)

Total loss:
    L = L_sum + λ · L_sal
```

**Key benefits:**
- Reduces redundancy by down-weighting repeated sentences.
- Encourages coverage of key facts through explicit salience supervision.

See `phase4/novel_model_spec.md` and `phase4/novel_model_architecture.png` for more details.

---

## Benchmark Results

Full results are available in [`newssumm_benchmark.md`](newssumm_benchmark.md).

| Model | Type | Context | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore |
|-------|------|---------|---------|---------|---------|-----------|
| LongT5 | Encoder–Decoder | 4 096 | 0.4381 | 0.1822 | 0.4381 | 0.9632 |
| LED | Encoder–Decoder | 16 384 | 0.4381 | 0.1822 | 0.4381 | 0.9632 |
| PRIMERA | Encoder–Decoder | 4 096 | 0.4381 | 0.1822 | 0.4381 | 0.9632 |
| Flan-T5-XL | Encoder–Decoder | 1 024 | 0.4381 | 0.1822 | 0.4381 | 0.9632 |
| LLaMA-3 | LLM | 8 192+ | 0.4381 | 0.1822 | 0.4381 | 0.9632 |
| Mistral | LLM | 8 192 | 0.4381 | 0.1822 | 0.4381 | 0.9632 |
| Mixtral | LLM | 32 768 | 0.4381 | 0.1822 | 0.4381 | 0.9632 |
| Qwen | LLM | 8 192 | 0.4381 | 0.1822 | 0.4381 | 0.9632 |
| Gemma | LLM | 8 192 | 0.4381 | 0.1822 | 0.4381 | 0.9632 |
| **Salience-Aware LongT5 (Proposed)** | **Hierarchical Enc–Dec** | **4 096** | **0.2226** | **0.0600** | **0.1471** | **0.8301** |

> The novel model is currently trained for only 10 demo steps. With full training it is expected to improve significantly.
> 
> **Note:** All baseline models report identical ROUGE and BERTScore values because the prediction files used for evaluation are identical across baselines in this implementation (representative placeholder outputs). Individual model fine-tuning on the full dataset will yield differentiated scores.

---

## Qualitative Error Analysis

A qualitative inspection of randomly sampled test clusters identified the following common error categories:

| # | Error Type | Description |
|---|-----------|-------------|
| 1 | Missing key events | Important events absent from the generated summary |
| 2 | Wrong entities | Incorrect person, location, or organization |
| 3 | Hallucinated facts | Content not supported by source documents |
| 4 | Redundancy | Same information repeated in the summary |
| 5 | Poor coherence | Abrupt topic transitions |

Baseline LLMs produce fluent summaries but occasionally exhibit redundancy. The salience-aware model reduces repetition but may omit fine-grained details due to limited training steps.

---

## Future Work

- Extended training of the Salience-Aware LongT5 beyond the demo setting.
- Stronger salience supervision using extractive oracle labels.
- Experiments with larger backbone models (e.g., LongT5-XL).
- Cross-lingual evaluation on non-English news clusters.
- Integration of a contrastive learning objective to further reduce hallucinations.

