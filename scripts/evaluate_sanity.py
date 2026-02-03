import sys, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.metrics import compute_metrics

def main():
    # Dummy predictions (simulate model output)
    predictions = [
        "The prime minister addressed the nation regarding COVID.",
        "The economy is expected to recover after reforms."
    ]

    # Ground truth summaries
    references = [
        "PM spoke to the nation about COVID.",
        "Economic recovery is expected after reforms."
    ]

    metrics = compute_metrics(predictions, references)

    print("Sanity Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
