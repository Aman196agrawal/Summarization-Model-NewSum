import torch
from torch.optim import AdamW
from transformers import AutoTokenizer
from src.novel_newssumm_model import SalienceAwareLongT5
from src.newssumm_dataset import NewsSummDataset


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # -----------------------------
    # Lightweight training loop
    # -----------------------------
    for step, sample in enumerate(dataset):
        if step >= 10:  # VERY SMALL run (demo only)
            break

        # Sentence segmentation: join documents with separator token
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

        print(f"Step {step} | Loss: {loss.item():.4f}")

    print("Training finished (demo run)")


if __name__ == "__main__":
    train()
