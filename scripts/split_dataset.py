import json
import random
from pathlib import Path

INPUT_FILE = "data/newssumm_phase1.jsonl"
OUTPUT_DIR = Path("data/splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# test gets the remaining 10%
SEED = 42

random.seed(SEED)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f]

random.shuffle(samples)

split_train = int(len(samples) * TRAIN_RATIO)
split_val = int(len(samples) * (TRAIN_RATIO + VAL_RATIO))
train_samples = samples[:split_train]
val_samples = samples[split_train:split_val]
test_samples = samples[split_val:]

def save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x) + "\n")

save(OUTPUT_DIR / "train.jsonl", train_samples)
save(OUTPUT_DIR / "val.jsonl", val_samples)
save(OUTPUT_DIR / "test.jsonl", test_samples)

print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
