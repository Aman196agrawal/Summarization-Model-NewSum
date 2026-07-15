import json
import random
from pathlib import Path

INPUT_FILE = "data/newssumm_phase1.jsonl"
OUTPUT_DIR = Path("data/splits")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# test gets the remaining 15%
SEED = 42


def compute_split_indices(n, train_ratio, val_ratio):
    """Return (split_train, split_val) boundary indices for a list of length n."""
    split_train = int(n * train_ratio)
    split_val = int(n * (train_ratio + val_ratio))
    return split_train, split_val


def split_samples(samples, train_ratio, val_ratio):
    """Split `samples` (already shuffled by the caller if desired) into
    (train, val, test) lists using cluster-level ratios -- each element of
    `samples` is one cluster, so this is a cluster-level split."""
    split_train, split_val = compute_split_indices(len(samples), train_ratio, val_ratio)
    train = samples[:split_train]
    val = samples[split_train:split_val]
    test = samples[split_val:]
    return train, val, test


def save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x) + "\n")


def main():
    random.seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    random.shuffle(samples)

    train_samples, val_samples, test_samples = split_samples(samples, TRAIN_RATIO, VAL_RATIO)

    save(OUTPUT_DIR / "train.jsonl", train_samples)
    save(OUTPUT_DIR / "val.jsonl", val_samples)
    save(OUTPUT_DIR / "test.jsonl", test_samples)

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")


if __name__ == "__main__":
    main()
