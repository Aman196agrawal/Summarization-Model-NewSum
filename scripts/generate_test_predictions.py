import json
import os
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------------------------------
# Project root
# -------------------------------------------------
PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
HF_MODEL = "google/long-t5-tglobal-base"
MODEL_TAG = "longt5"

TEST_FILE = "data/splits/test.jsonl"
OUT_DIR = Path(f"results/{MODEL_TAG}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "test_predictions.jsonl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(
    HF_MODEL,
    device_map="auto",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)
model.eval()
print("Model loaded")

# -------------------------------------------------
# GENERATE PREDICTIONS
# -------------------------------------------------
count = 0
written = 0

with open(TEST_FILE) as fin, open(OUT_FILE, "w") as fout:
    for line in fin:
        sample = json.loads(line)
        count += 1

        # ---- FIXED INPUT HANDLING ----
        if "documents" not in sample:
            raise KeyError(f"'documents' not found. Keys: {sample.keys()}")

        documents = sample["documents"]
        cluster_id = sample["cluster_id"]

        # Concatenate documents with delimiter
        input_text = "\n[DOC]\n".join(documents)

        inputs = tokenizer(
            input_text,
            truncation=True,
            max_length=4096,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=256,
                num_beams=4
            )

        summary = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        fout.write(json.dumps({
            "cluster_id": cluster_id,
            "generated_summary": summary
        }) + "\n")

        written += 1

        if written % 10 == 0:
            print(f"Generated {written} summaries")

print(f"✅ Finished. Read {count} samples, wrote {written} predictions.")
print(f"Saved to {OUT_FILE}")
