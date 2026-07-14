import os
import random
import torch
from torch.optim import AdamW
from transformers import T5Tokenizer

from src.novel_newssumm_model import SalienceAwareLongT5
from src.newssumm_dataset import NewsSummDataset
from src.sentence_utils import build_cluster_input
from src.salience_oracle import compute_oracle_labels

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
DEMO_MODE = True          # Set to False for full training
MAX_STEPS = 10            # Used only when DEMO_MODE = True; None = full training
MODEL_NAME = "google/long-t5-tglobal-base"
CHECKPOINT_DIR = "results/sa_longt5/checkpoints"
CHECKPOINT_EVERY = 500    # Save a checkpoint every N steps (full training)
MAX_INPUT_LENGTH = 4096
MAX_SUMMARY_LENGTH = 128


def prepare_batch(sample, tokenizer, device, max_input_length=MAX_INPUT_LENGTH,
                   max_summary_length=MAX_SUMMARY_LENGTH):
    """Tokenize one training sample end to end:
    - builds the </s>-joined multi-document input (src.sentence_utils),
    - computes greedy ROUGE-oracle salience labels aligned to those sentences,
    - masks summary padding with -100 so the loss ignores it (fixes the bug
      where the model was being trained to predict <pad> tokens).
    """
    input_text, sentences = build_cluster_input(sample["documents"], tokenizer)

    inputs = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=max_input_length,
        return_tensors="pt",
    )

    target = tokenizer(
        sample["summary"],
        truncation=True,
        padding="max_length",
        max_length=max_summary_length,
        return_tensors="pt",
    )
    labels = target["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    oracle_labels = compute_oracle_labels(sentences, sample["summary"])
    salience_labels = torch.tensor([oracle_labels], dtype=torch.float)

    return {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
        "labels": labels.to(device),
        "salience_labels": salience_labels.to(device),
    }


def train():
    # -----------------------------
    # Reproducibility
    # -----------------------------
    torch.manual_seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # -----------------------------
    # Load tokenizer and model
    # -----------------------------
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = SalienceAwareLongT5(MODEL_NAME)
    model.to(device)

    # -----------------------------
    # Load dataset (train split)
    # -----------------------------
    dataset = NewsSummDataset("data/splits/train.jsonl")

    optimizer = AdamW(model.parameters(), lr=2e-5)

    model.train()

    step_limit = MAX_STEPS if DEMO_MODE else None

    # -----------------------------
    # Training loop
    # -----------------------------
    for step, sample in enumerate(dataset):
        if step_limit is not None and step >= step_limit:
            break

        batch = prepare_batch(sample, tokenizer, device)

        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            salience_labels=batch["salience_labels"],
        )

        loss = output["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")

        if not DEMO_MODE and CHECKPOINT_EVERY is not None and CHECKPOINT_EVERY > 0 and (step + 1) % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_step_{step + 1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    # Save final checkpoint
    final_ckpt = os.path.join(CHECKPOINT_DIR, "model_final.pt")
    torch.save(model.state_dict(), final_ckpt)
    print(f"Final checkpoint saved: {final_ckpt}")

    mode_label = "demo run" if DEMO_MODE else "full training"
    print(f"Training finished ({mode_label})")


if __name__ == "__main__":
    train()
