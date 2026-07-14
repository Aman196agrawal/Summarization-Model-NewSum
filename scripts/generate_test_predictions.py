import json
import os
import sys
from pathlib import Path
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.sentence_utils import build_cluster_input

HF_MODEL = "google/long-t5-tglobal-base"
MODEL_TAG = "longt5_baseline"
TEST_FILE = "data/splits/test.jsonl"
OUT_DIR = Path(f"results/{MODEL_TAG}")
MAX_INPUT_LENGTH = 4096
MAX_OUTPUT_LENGTH = 256
NUM_BEAMS = 4


def generate_predictions(model, tokenizer, samples, device, max_input_length=MAX_INPUT_LENGTH,
                          max_output_length=MAX_OUTPUT_LENGTH, num_beams=NUM_BEAMS):
    """Baseline LongT5 predictions using the same </s>-joined input
    construction and the same beam=4/max_length=256 decoding as the novel
    model's script, so any ROUGE gap between the two reflects the
    architecture rather than mismatched input formatting or decoding
    settings.
    """
    for sample in samples:
        input_text, _ = build_cluster_input(sample["documents"], tokenizer)
        inputs = tokenizer(
            input_text, truncation=True, max_length=max_input_length, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_length=max_output_length, num_beams=num_beams
            )

        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        yield {
            "cluster_id": sample["cluster_id"],
            "generated_summary": summary,
            "reference_summary": sample["summary"],
        }


def main():
    from transformers import T5Tokenizer, AutoModelForSeq2SeqLM

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    tokenizer = T5Tokenizer.from_pretrained(HF_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        HF_MODEL,
        device_map="auto",
        torch_dtype=torch.float16 if device_str == "cuda" else torch.float32,
    )
    model.eval()
    print("Model loaded")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "test_predictions.jsonl"

    count = 0
    with open(TEST_FILE) as fin, open(out_file, "w") as fout:
        samples = (json.loads(line) for line in fin)
        for record in generate_predictions(model, tokenizer, samples, device=model.device):
            fout.write(json.dumps(record) + "\n")
            count += 1
            if count % 10 == 0:
                print(f"Generated {count} summaries")

    print(f"Finished. Wrote {count} predictions to {out_file}")


if __name__ == "__main__":
    main()
