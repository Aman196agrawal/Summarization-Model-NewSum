import json
import os
import sys
from pathlib import Path

# -------------------------------------------------
# Make project root importable (Colab-safe)
# -------------------------------------------------
PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)

from src.metrics import compute_metrics

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MODEL_NAME = "longt5"   # change per model

PRED_FILE = f"results/{MODEL_NAME}/test_predictions.jsonl"
TEST_FILE = "data/splits/test.jsonl"

OUT_DIR = Path(f"results/phase3/{MODEL_NAME}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# LOAD REFERENCE SUMMARIES
# -------------------------------------------------
ref_map = {}

with open(TEST_FILE) as f:
    for line in f:
        s = json.loads(line)
        ref_map[s["cluster_id"]] = s["summary"]  # or "human_summary"

print("Loaded references:", len(ref_map))

# -------------------------------------------------
# LOAD MODEL PREDICTIONS
# -------------------------------------------------
predictions = []
references = []

with open(PRED_FILE) as f:
    for line in f:
        s = json.loads(line)
        cid = s["cluster_id"]

        if cid in ref_map:
            predictions.append(s["generated_summary"])
            references.append(ref_map[cid])

print("Aligned samples:", len(predictions))

# -------------------------------------------------
# COMPUTE METRICS
# -------------------------------------------------
metrics = compute_metrics(predictions, references)

# -------------------------------------------------
# SAVE RESULTS
# -------------------------------------------------
out_path = OUT_DIR / "test_metrics.json"
with open(out_path, "w") as f:
    json.dump(metrics, f, indent=4)

print("Final TEST metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

print(f"Saved to {out_path}")
