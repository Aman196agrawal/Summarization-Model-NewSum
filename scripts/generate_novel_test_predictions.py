import os
import sys
import json
import torch
from transformers import AutoTokenizer

# -------------------------------------------------
# FIX PYTHON PATH (CRITICAL)
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.novel_newssumm_model import SalienceAwareLongT5

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
TEST_FILE = "data/splits/test.jsonl"
OUT_FILE = "results/phase4/novel/test_predictions.jsonl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading tokenizer and novel model...")
tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
model = SalienceAwareLongT5()
model.to(device)
model.eval()

count = 0

with open(TEST_FILE) as fin, open(OUT_FILE, "w") as fout:
    for line in fin:
        sample = json.loads(line)

        # Sentence segmentation: join documents with separator token
        input_text = " [SEP] ".join(sample["documents"])
        inputs = tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=4096,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            generated_ids = model.base_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=128
            )

        generated_summary = tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )

        fout.write(json.dumps({
            "cluster_id": sample["cluster_id"],
            "generated_summary": generated_summary,
            "reference_summary": sample["summary"]
        }) + "\n")

        count += 1
        if count % 20 == 0:
            print(f"Generated {count} summaries")

print(f"Saved predictions to {OUT_FILE}")
