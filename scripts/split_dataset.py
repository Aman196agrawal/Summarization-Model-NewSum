import json
import random
from pathlib import Path

INPUT_FILE = "data/newssumm_phase1.jsonl"
OUTPUT_DIR = Path("data/splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.9
SEED = 42

random.seed(SEED)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f]

random.shuffle(samples)

split_idx = int(len(samples) * TRAIN_RATIO)
train_samples = samples[:split_idx]
val_samples = samples[split_idx:]

def save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x) + "\n")

save(OUTPUT_DIR / "train.jsonl", train_samples)
save(OUTPUT_DIR / "val.jsonl", val_samples)

print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
