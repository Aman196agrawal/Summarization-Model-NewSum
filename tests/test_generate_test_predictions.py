import torch

from scripts.generate_test_predictions import generate_predictions


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
        return "baseline generated summary"


class _FakeModel:
    def generate(self, **kwargs):
        assert kwargs["max_length"] == 256
        assert kwargs["num_beams"] == 4
        return torch.tensor([[7, 7, 7]])


def test_generate_predictions_uses_shared_input_builder_and_unified_decoding():
    samples = [{"cluster_id": "c1", "documents": ["Doc one. Doc two."], "summary": "Reference."}]

    records = list(
        generate_predictions(_FakeModel(), _FakeTokenizer(), samples, device=torch.device("cpu"))
    )

    assert records == [{
        "cluster_id": "c1",
        "generated_summary": "baseline generated summary",
        "reference_summary": "Reference.",
    }]
