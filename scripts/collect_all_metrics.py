import json
import csv
from pathlib import Path

RESULTS_DIR = Path("results")
OUTPUT_JSON = RESULTS_DIR / "metrics_all_models.json"
OUTPUT_CSV = RESULTS_DIR / "metrics_all_models.csv"

all_metrics = {}

for model_dir in RESULTS_DIR.iterdir():
    if model_dir.is_dir():
        metrics_file = model_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                all_metrics[model_dir.name] = json.load(f)

# Save JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(all_metrics, f, indent=4)

# Save CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore"])
    for model, scores in all_metrics.items():
        writer.writerow([
            model,
            scores["ROUGE-1"],
            scores["ROUGE-2"],
            scores["ROUGE-L"],
            scores["BERTScore"]
        ])

print("Saved final metrics table:")
print(OUTPUT_JSON)
print(OUTPUT_CSV)
