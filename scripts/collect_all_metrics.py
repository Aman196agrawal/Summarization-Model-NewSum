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
