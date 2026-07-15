import torch
import pytest

from scripts.generate_novel_test_predictions import generate_predictions, load_trained_model


class _FakeBatchEncoding(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    eos_token = "</s>"

    def __call__(self, text, **kwargs):
        return _FakeBatchEncoding({
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        })

    def decode(self, ids, skip_special_tokens=True):
        return "generated summary text"


class _FakeModel:
    def generate(self, input_ids, attention_mask, max_length, num_beams):
        return torch.tensor([[9, 9, 9]])


def test_generate_predictions_calls_model_generate_and_formats_output():
    samples = [{"cluster_id": "c1", "documents": ["Doc one."], "summary": "Reference summary."}]

    records = list(
        generate_predictions(_FakeModel(), _FakeTokenizer(), samples, device=torch.device("cpu"))
    )

    assert records == [{
        "cluster_id": "c1",
        "generated_summary": "generated summary text",
        "reference_summary": "Reference summary.",
    }]


def test_load_trained_model_raises_when_checkpoint_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.pt"

    with pytest.raises(FileNotFoundError):
        load_trained_model("irrelevant-model-name", str(missing_path), torch.device("cpu"))
