import os
import random
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer
from src.novel_newssumm_model import SalienceAwareLongT5
from src.newssumm_dataset import NewsSummDataset

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
DEMO_MODE = True          # Set to False for full training
MAX_STEPS = 10            # Used only when DEMO_MODE = True; None = full training
CHECKPOINT_DIR = "results/phase4/checkpoints"
CHECKPOINT_EVERY = 500    # Save a checkpoint every N steps (full training)


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
    tokenizer = AutoTokenizer.from_pretrained(
        "google/long-t5-tglobal-base"
    )
    model = SalienceAwareLongT5()
    model.to(device)

    # -----------------------------
    # Load dataset (train split)
    # -----------------------------
    dataset = NewsSummDataset("data/splits/train.jsonl")

    optimizer = AdamW(model.parameters(), lr=2e-5)

    model.train()

    # Determine step limit
    step_limit = MAX_STEPS if DEMO_MODE else None

    # -----------------------------
    # Training loop
    # -----------------------------
    for step, sample in enumerate(dataset):
        if step_limit is not None and step >= step_limit:
            break

        # Sentence segmentation: join documents with [SEP] delimiter
        input_text = " [SEP] ".join(sample["documents"])
        inputs = tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=4096,
            return_tensors="pt"
        )

        labels = tokenizer(
            sample["summary"],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )["input_ids"]

        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels
        )

        loss = output["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")

        # Save checkpoint every N steps (full training mode)
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
