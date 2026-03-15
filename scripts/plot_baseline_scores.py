import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_PATH = "results/newssumm_baselines_scores.csv"
OUT_DIR = "results/plots"

PROPOSED_MODEL = {
    "model_name": "Salience-Aware LongT5 (Ours)",
    "ROUGE-1": 0.2226,
    "ROUGE-2": 0.0600,
    "ROUGE-L": 0.1471,
    "BERTScore": 0.8301,
}
PROPOSED_COLOR = "#C0392B"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Append proposed model row
proposed_row = pd.DataFrame([PROPOSED_MODEL])
df = pd.concat([df, proposed_row], ignore_index=True)

metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore"]

for metric in metrics:
    plt.figure(figsize=(12, 5))
    bar_colors = [
        PROPOSED_COLOR if name == PROPOSED_MODEL["model_name"] else "#2196F3"
        for name in df["model_name"]
    ]
    plt.bar(df["model_name"], df[metric], color=bar_colors)
    plt.xticks(rotation=45, ha="right")
    plt.title(metric + " Comparison Across Models")
    plt.ylabel(metric)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, metric.lower().replace("-", "") + ".png")
    plt.savefig(out_path)
    plt.close()

    print(f"Saved {out_path}")
