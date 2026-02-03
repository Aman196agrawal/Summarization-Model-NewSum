import sys
import os

# --- FIX: add project root to Python path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.newssumm_dataset import NewsSummDataset

DOC_TOKEN = "[DOC]"

def build_input(documents):
    return " ".join([f"{DOC_TOKEN} {doc}" for doc in documents])


if __name__ == "__main__":
    dataset = NewsSummDataset("data/newssumm_phase1.jsonl")

    sample = dataset[0]
    model_input = build_input(sample["documents"])

    print("INPUT SAMPLE:\n")
    print(model_input[:1500])
    print("\nTARGET SUMMARY:\n")
    print(sample["summary"])
  
