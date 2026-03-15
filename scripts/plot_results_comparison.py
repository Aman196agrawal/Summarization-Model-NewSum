import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUTPUT_DIR = "results/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')

# =============================================================================
# DATA
# =============================================================================

models_data = {
    "PRIMERA":             {"ROUGE-1": 0.412, "ROUGE-2": 0.201, "ROUGE-L": 0.389, "BERTScore": 0.921},
    "LED-base":            {"ROUGE-1": 0.421, "ROUGE-2": 0.214, "ROUGE-L": 0.401, "BERTScore": 0.926},
    "LongT5-base":         {"ROUGE-1": 0.398, "ROUGE-2": 0.187, "ROUGE-L": 0.372, "BERTScore": 0.914},
    "Flan-T5-XL":          {"ROUGE-1": 0.356, "ROUGE-2": 0.162, "ROUGE-L": 0.331, "BERTScore": 0.902},
    "Flan-T5-XXL":         {"ROUGE-1": 0.372, "ROUGE-2": 0.174, "ROUGE-L": 0.346, "BERTScore": 0.907},
    "Mistral-7B":          {"ROUGE-1": 0.341, "ROUGE-2": 0.149, "ROUGE-L": 0.318, "BERTScore": 0.895},
    "LLaMA-3-8B":          {"ROUGE-1": 0.334, "ROUGE-2": 0.143, "ROUGE-L": 0.309, "BERTScore": 0.889},
    "Qwen2-7B":            {"ROUGE-1": 0.348, "ROUGE-2": 0.158, "ROUGE-L": 0.322, "BERTScore": 0.898},
    "Gemma-2-9B":          {"ROUGE-1": 0.331, "ROUGE-2": 0.141, "ROUGE-L": 0.305, "BERTScore": 0.886},
    "Mixtral-8x7B":        {"ROUGE-1": 0.366, "ROUGE-2": 0.171, "ROUGE-L": 0.342, "BERTScore": 0.910},
    "Salience-Aware\nLongT5 (Ours)": {"ROUGE-1": 0.2226, "ROUGE-2": 0.0600, "ROUGE-L": 0.1471, "BERTScore": 0.8301},
}

PROPOSED_KEY = "Salience-Aware\nLongT5 (Ours)"
PROPOSED_COLOR = "#C0392B"
BASELINE_COLOR = "#2E86AB"

model_names = list(models_data.keys())
metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore"]

# =============================================================================
# Chart 1 — Grouped Bar Chart: All 4 Metrics Side by Side
# =============================================================================

def plot_grouped_bar():
    fig, ax = plt.subplots(figsize=(16, 8))

    n_models = len(model_names)
    n_metrics = len(metrics)
    bar_width = 0.18
    x = np.arange(n_models)

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
    offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * bar_width

    # Highlight proposed model group with a light red background
    proposed_idx = model_names.index(PROPOSED_KEY)
    rect = FancyBboxPatch(
        (proposed_idx - 0.45, 0), 0.9, 1.0,
        boxstyle="round,pad=0.02",
        linewidth=1.5,
        edgecolor=PROPOSED_COLOR,
        facecolor="#FADBD8",
        transform=ax.get_xaxis_transform(),
        clip_on=False,
        zorder=0,
    )
    ax.add_patch(rect)

    for i, (metric, color, offset) in enumerate(zip(metrics, colors, offsets)):
        values = [models_data[m][metric] for m in model_names]
        ax.bar(x + offset, values, bar_width, label=metric, color=color, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Benchmark Comparison: All Metrics Across 11 Models", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "grouped_bar_all_metrics.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# =============================================================================
# Chart 2 — Horizontal Bar Chart: ROUGE-1 Ranking
# =============================================================================

def plot_rouge1_ranking():
    sorted_models = sorted(model_names, key=lambda m: models_data[m]["ROUGE-1"])
    rouge1_scores = [models_data[m]["ROUGE-1"] for m in sorted_models]
    bar_colors = [PROPOSED_COLOR if m == PROPOSED_KEY else BASELINE_COLOR for m in sorted_models]

    fig, ax = plt.subplots(figsize=(12, 8))

    y = np.arange(len(sorted_models))
    bars = ax.barh(y, rouge1_scores, color=bar_colors, height=0.6)

    # Value labels on each bar
    for bar, val in zip(bars, rouge1_scores):
        ax.text(
            bar.get_width() + 0.004, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=9,
        )

    # Vertical dashed line at proposed model's score
    proposed_rouge1 = models_data[PROPOSED_KEY]["ROUGE-1"]
    ax.axvline(x=proposed_rouge1, color=PROPOSED_COLOR, linestyle="--", linewidth=1.5, label=f"Proposed ({proposed_rouge1:.4f})")

    display_names = [m.replace("\n", " ") for m in sorted_models]
    ax.set_yticks(y)
    ax.set_yticklabels(display_names, fontsize=10)
    ax.set_xlabel("ROUGE-1 F1 Score", fontsize=12)
    ax.set_xlim(0.0, 0.5)
    ax.set_title("ROUGE-1 F1 Score — Model Ranking", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "rouge1_ranking.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# =============================================================================
# Chart 3 — Radar / Spider Chart: Multi-Metric Profile (Top 5 + Proposed)
# =============================================================================

def plot_radar_chart():
    # Top 5 baselines by ROUGE-1
    baselines = {k: v for k, v in models_data.items() if k != PROPOSED_KEY}
    top5 = sorted(baselines, key=lambda m: baselines[m]["ROUGE-1"], reverse=True)[:5]
    selected = top5 + [PROPOSED_KEY]

    categories = metrics
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    radar_colors = plt.cm.tab10(np.linspace(0, 0.6, len(top5)))

    for i, model in enumerate(selected):
        values = [models_data[model][m] for m in categories]
        values += values[:1]

        if model == PROPOSED_KEY:
            color = PROPOSED_COLOR
            lw = 2.5
            alpha = 0.5
            label = model.replace("\n", " ")
        else:
            color = radar_colors[i]
            lw = 1.5
            alpha = 0.3
            label = model

        ax.plot(angles, values, color=color, linewidth=lw, linestyle="solid")
        ax.fill(angles, values, color=color, alpha=alpha)
        ax.plot([], [], color=color, linewidth=lw, label=label)  # legend proxy

    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title("Multi-Metric Radar: Top 5 Baselines vs Proposed Model",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "radar_chart.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# =============================================================================
# Chart 4 — BERTScore vs ROUGE-1 Scatter Plot
# =============================================================================

def plot_bertscore_vs_rouge1():
    fig, ax = plt.subplots(figsize=(12, 8))

    proposed_r1 = models_data[PROPOSED_KEY]["ROUGE-1"]
    proposed_bs = models_data[PROPOSED_KEY]["BERTScore"]

    # Dashed reference lines
    ax.axhline(y=proposed_bs, color=PROPOSED_COLOR, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(x=proposed_r1, color=PROPOSED_COLOR, linestyle="--", linewidth=1.2, alpha=0.8)

    for model, scores in models_data.items():
        r1 = scores["ROUGE-1"]
        bs = scores["BERTScore"]
        is_proposed = model == PROPOSED_KEY

        ax.scatter(
            r1, bs,
            color=PROPOSED_COLOR if is_proposed else "#2196F3",
            s=200 if is_proposed else 100,
            zorder=5,
        )

        display_name = model.replace("\n", " ")
        offset_x = 0.004
        offset_y = 0.001
        ax.annotate(
            display_name,
            xy=(r1, bs),
            xytext=(r1 + offset_x, bs + offset_y),
            fontsize=8,
            color=PROPOSED_COLOR if is_proposed else "black",
            fontweight="bold" if is_proposed else "normal",
        )

    ax.set_xlabel("ROUGE-1", fontsize=12)
    ax.set_ylabel("BERTScore", fontsize=12)
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(0.80, 1.0)
    ax.set_title("BERTScore vs ROUGE-1: All 11 Models", fontsize=14, fontweight="bold")

    proposed_patch = mpatches.Patch(color=PROPOSED_COLOR, label="Salience-Aware LongT5 (Ours)")
    baseline_patch = mpatches.Patch(color="#2196F3", label="Baselines")
    ax.legend(handles=[baseline_patch, proposed_patch], fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "bertscore_vs_rouge1.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# =============================================================================
# Run all charts
# =============================================================================

if __name__ == "__main__":
    plot_grouped_bar()
    plot_rouge1_ranking()
    plot_radar_chart()
    plot_bertscore_vs_rouge1()
    print("All 4 charts saved to results/plots/")
