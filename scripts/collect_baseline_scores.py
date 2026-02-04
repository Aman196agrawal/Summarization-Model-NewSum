import json
import csv
from pathlib import Path

# -------------------------------------------------
# CONFIG: baseline metadata
# -------------------------------------------------
BASELINES = [
    {
        "model_name": "LongT5",
        "model_dir": "longt5",
        "params_m": 248,
        "context_len": 4096,
        "training_type": "zero-shot"
    }
    # Add more models here later
]

OUT_FILE = "results/newssumm_baselines_scores.csv"

# -------------------------------------------------
# COLLECT METRICS
# -------------------------------------------------
rows = []

for b in BASELINES:
    metrics_path = Path(f"results/phase3/{b['model_dir']}/test_metrics.json")
    if not metrics_path.exists():
        print(f"Skipping {b['model_name']} (metrics not found)")
        continue

    with open(metrics_path) as f:
        metrics = json.load(f)

    rows.append({
        "model_name": b["model_name"],
        "params_M": b["params_m"],
        "context_len": b["context_len"],
        "training_type": b["training_type"],
        "ROUGE-1": round(metrics["ROUGE-1"], 4),
        "ROUGE-2": round(metrics["ROUGE-2"], 4),
        "ROUGE-L": round(metrics["ROUGE-L"], 4),
        "BERTScore": round(metrics["BERTScore"], 4),
    })

# -------------------------------------------------
# WRITE CSV
# -------------------------------------------------
with open(OUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved baseline scores to {OUT_FILE}")
