import json

import torch
from torch.optim import AdamW
from transformers import LongT5Config, LongT5ForConditionalGeneration, T5Tokenizer

from src.novel_newssumm_model import SalienceAwareLongT5
from src.sentence_utils import build_cluster_input
from src.salience_oracle import compute_oracle_labels
from src.metrics import compute_metrics
from scripts.split_dataset import split_samples
from scripts.generate_novel_test_predictions import generate_predictions


def _tiny_base_model(vocab_size):
    config = LongT5Config(
        vocab_size=vocab_size,
        d_model=8,
        d_kv=4,
        d_ff=16,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        local_radius=4,
        global_block_size=4,
        feed_forward_proj="relu",
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    return LongT5ForConditionalGeneration(config)


def test_full_pipeline_runs_on_sample_data_with_a_tiny_model(monkeypatch):
    with open("data/sample.jsonl") as f:
        samples = [json.loads(line) for line in f]

    train, val, test = split_samples(samples, train_ratio=0.70, val_ratio=0.15)
    assert len(train) + len(val) + len(test) == len(samples)
    assert len(test) > 0

    tokenizer = T5Tokenizer.from_pretrained("google/long-t5-tglobal-base")
    model = SalienceAwareLongT5.from_base_model(_tiny_base_model(tokenizer.vocab_size))

    device = torch.device("cpu")
    optimizer = AdamW(model.parameters(), lr=1e-3)
    model.train()

    for sample in train[:2]:
        input_text, sentences = build_cluster_input(sample["documents"], tokenizer)
        inputs = tokenizer(
            input_text, truncation=True, padding="max_length", max_length=64, return_tensors="pt"
        )
        target = tokenizer(
            sample["summary"], truncation=True, padding="max_length", max_length=16, return_tensors="pt"
        )
        labels = target["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        oracle_labels = compute_oracle_labels(sentences, sample["summary"])
        salience_labels = torch.tensor([oracle_labels], dtype=torch.float)

        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
            salience_labels=salience_labels,
        )
        output["loss"].backward()
        optimizer.step()
        optimizer.zero_grad()
        assert torch.isfinite(output["loss"])

    model.eval()
    predictions = list(generate_predictions(
        model, tokenizer, test, device, max_input_length=64, max_output_length=8, num_beams=2
    ))
    assert len(predictions) == len(test)
    for record in predictions:
        assert isinstance(record["generated_summary"], str)

    monkeypatch.setattr(
        "src.metrics.rouge.compute",
        lambda predictions, references: {
            "rouge1": 0.1, "rouge2": 0.05, "rougeL": 0.02, "rougeLsum": 0.08
        },
    )
    monkeypatch.setattr(
        "src.metrics.bertscore.compute",
        lambda predictions, references, lang: {"f1": [0.5] * len(predictions)},
    )
    metrics = compute_metrics(
        [r["generated_summary"] for r in predictions],
        [r["reference_summary"] for r in predictions],
    )
    assert metrics["ROUGE-L"] == 0.08
