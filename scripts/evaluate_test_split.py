import os
import sys
import json
import argparse
import evaluate

# -------------------------------------------------
# Fix Python path
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.metrics import compute_metrics

# -------------------------------------------------
# Argument parsing
# -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--predictions", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

PRED_FILE = args.predictions
OUT_FILE = args.output

# -------------------------------------------------
# Load predictions and references
# -------------------------------------------------
predictions = []
references = []

with open(PRED_FILE) as f:
    for line in f:
        sample = json.loads(line)
        predictions.append(sample["generated_summary"])
        references.append(sample["reference_summary"])

print(f"Loaded predictions: {len(predictions)}")
print(f"Loaded references: {len(references)}")

# -------------------------------------------------
# Compute metrics
# -------------------------------------------------
metrics = compute_metrics(predictions, references)

print("\nFinal TEST metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# -------------------------------------------------
# Save metrics
# -------------------------------------------------
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

with open(OUT_FILE, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nSaved to {OUT_FILE}")
