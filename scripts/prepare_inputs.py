import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.newssumm_dataset import NewsSummDataset
from src.sentence_utils import build_cluster_input


class _EosOnlyTokenizer:
    """Minimal stand-in exposing just the `.eos_token` attribute
    build_cluster_input needs, for previewing input construction without
    loading a full tokenizer."""
    eos_token = "</s>"


if __name__ == "__main__":
    dataset = NewsSummDataset("data/newssumm_phase1.jsonl")

    sample = dataset[0]
    model_input, sentences = build_cluster_input(sample["documents"], _EosOnlyTokenizer())

    print("INPUT SAMPLE:\n")
    print(model_input[:1500])
    print(f"\n({len(sentences)} sentences detected)")
    print("\nTARGET SUMMARY:\n")
    print(sample["summary"])
