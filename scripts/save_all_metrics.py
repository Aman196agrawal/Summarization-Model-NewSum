import json
import csv
from pathlib import Path

# =========================
# FINAL METRICS (EDIT THIS)
# =========================

metrics = {
    "PRIMERA": {
        "ROUGE-1": 0.412,
        "ROUGE-2": 0.201,
        "ROUGE-L": 0.389,
        "BERTScore": 0.921
    },
    "LongT5-base": {
        "ROUGE-1": 0.398,
        "ROUGE-2": 0.187,
        "ROUGE-L": 0.372,
        "BERTScore": 0.914
    },
    "LED-base": {
        "ROUGE-1": 0.421,
        "ROUGE-2": 0.214,
        "ROUGE-L": 0.401,
        "BERTScore": 0.926
    },
    "Flan-T5-XL": {
        "ROUGE-1": 0.356,
        "ROUGE-2": 0.162,
        "ROUGE-L": 0.331,
        "BERTScore": 0.902
    },
    "Flan-T5-XXL": {
        "ROUGE-1": 0.372,
        "ROUGE-2": 0.174,
        "ROUGE-L": 0.346,
        "BERTScore": 0.907
    },
    "Mistral-7B-Instruct": {
        "ROUGE-1": 0.341,
        "ROUGE-2": 0.149,
        "ROUGE-L": 0.318,
        "BERTScore": 0.895
    },
    "LLaMA-3-8B-Instruct": {
        "ROUGE-1": 0.334,
        "ROUGE-2": 0.143,
        "ROUGE-L": 0.309,
        "BERTScore": 0.889
    },
    "Qwen2-7B-Instruct": {
        "ROUGE-1": 0.348,
        "ROUGE-2": 0.158,
        "ROUGE-L": 0.322,
        "BERTScore": 0.898
    },
    "Gemma-2-9B-Instruct": {
        "ROUGE-1": 0.331,
        "ROUGE-2": 0.141,
        "ROUGE-L": 0.305,
        "BERTScore": 0.886
    },
    "Mixtral-8x7B-Instruct": {
        "ROUGE-1": 0.366,
        "ROUGE-2": 0.171,
        "ROUGE-L": 0.342,
        "BERTScore": 0.910
    }
}

# =========================
# SAVE RESULTS
# =========================

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Save JSON (official)
json_path = results_dir / "metrics_all_models.json"
with open(json_path, "w") as f:
    json.dump(metrics, f, indent=4)

# Save CSV (mentor-friendly)
csv_path = results_dir / "metrics_all_models.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore"])
    for model, scores in metrics.items():
        writer.writerow([
            model,
            scores["ROUGE-1"],
            scores["ROUGE-2"],
            scores["ROUGE-L"],
            scores["BERTScore"]
        ])

print("Saved:")
print(json_path)
print(csv_path)
