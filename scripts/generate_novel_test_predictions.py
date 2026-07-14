import os
import sys
import json
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.sentence_utils import build_cluster_input

MODEL_NAME = "google/long-t5-tglobal-base"
TEST_FILE = "data/splits/test.jsonl"
OUT_FILE = "results/sa_longt5/test_predictions.jsonl"
CHECKPOINT_PATH = "results/sa_longt5/checkpoints/model_final.pt"
MAX_INPUT_LENGTH = 4096
MAX_OUTPUT_LENGTH = 256
NUM_BEAMS = 4


def generate_predictions(model, tokenizer, samples, device, max_input_length=MAX_INPUT_LENGTH,
                          max_output_length=MAX_OUTPUT_LENGTH, num_beams=NUM_BEAMS):
    """Run the salience-aware model over `samples` (dicts with 'cluster_id',
    'documents', 'summary') and yield one prediction record per sample.
    Calls model.generate(...) -- the SalienceAwareLongT5 wrapper's own
    generate(), not model.base_model.generate() -- so beam search actually
    depends on the salience-aware constrained hidden states.
    """
    for sample in samples:
        input_text, _ = build_cluster_input(sample["documents"], tokenizer)
        inputs = tokenizer(
            input_text, truncation=True, max_length=max_input_length, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_output_length,
                num_beams=num_beams,
            )

        generated_summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        yield {
            "cluster_id": sample["cluster_id"],
            "generated_summary": generated_summary,
            "reference_summary": sample["summary"],
        }


def load_trained_model(model_name, checkpoint_path, device):
    """Load a trained SalienceAwareLongT5 checkpoint. Raises FileNotFoundError
    (rather than silently evaluating untrained weights) if no checkpoint has
    been produced yet -- run phase4/train_novel_model.py first.
    """
    from src.novel_newssumm_model import SalienceAwareLongT5

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No trained checkpoint found at {checkpoint_path}. Run "
            "phase4/train_novel_model.py first -- evaluating an untrained "
            "model would not reflect the salience-aware architecture."
        )

    model = SalienceAwareLongT5(model_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def main():
    from transformers import T5Tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading tokenizer and trained novel model...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = load_trained_model(MODEL_NAME, CHECKPOINT_PATH, device)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    count = 0
    with open(TEST_FILE) as fin, open(OUT_FILE, "w") as fout:
        samples = (json.loads(line) for line in fin)
        for record in generate_predictions(model, tokenizer, samples, device):
            fout.write(json.dumps(record) + "\n")
            count += 1
            if count % 20 == 0:
                print(f"Generated {count} summaries")

    print(f"Saved {count} predictions to {OUT_FILE}")


if __name__ == "__main__":
    main()
