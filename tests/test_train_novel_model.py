import torch
from transformers import T5Tokenizer

from phase4.train_novel_model import prepare_batch, MODEL_NAME


def test_prepare_batch_masks_padding_and_produces_aligned_salience_labels():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    sample = {
        "documents": [
            "The prime minister announced a new policy today. Markets reacted calmly."
        ],
        "summary": "The prime minister announced a new policy today.",
    }

    batch = prepare_batch(
        sample, tokenizer, device=torch.device("cpu"),
        max_input_length=64, max_summary_length=16,
    )

    assert (batch["labels"] == tokenizer.pad_token_id).sum().item() == 0
    assert (batch["labels"] == -100).any()
    assert batch["salience_labels"].shape[0] == 1
    assert batch["salience_labels"].sum().item() >= 1
