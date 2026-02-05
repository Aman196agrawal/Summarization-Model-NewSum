import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_PATH = "results/newssumm_baselines_scores.csv"
OUT_DIR = "results/plots"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore"]

for metric in metrics:
    plt.figure(figsize=(10, 5))
    plt.bar(df["model_name"], df[metric])
    plt.xticks(rotation=45, ha="right")
    plt.title(metric + " Comparison Across Models")
    plt.ylabel(metric)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, metric.lower().replace("-", "") + ".png")
    plt.savefig(out_path)
    plt.close()

    print(f"Saved {out_path}")
