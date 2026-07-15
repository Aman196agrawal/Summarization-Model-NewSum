import pytest

import src.metrics as metrics_module


def test_compute_metrics_maps_rougelsum_to_rouge_l_key(monkeypatch):
    monkeypatch.setattr(
        metrics_module.rouge,
        "compute",
        lambda predictions, references: {
            "rouge1": 0.5,
            "rouge2": 0.3,
            "rougeL": 0.1,
            "rougeLsum": 0.4,
        },
    )
    monkeypatch.setattr(
        metrics_module.bertscore,
        "compute",
        lambda predictions, references, lang: {"f1": [0.9, 0.8]},
    )

    result = metrics_module.compute_metrics(["a"], ["b"])

    assert result["ROUGE-1"] == 0.5
    assert result["ROUGE-2"] == 0.3
    assert result["ROUGE-L"] == 0.4
    assert result["BERTScore"] == pytest.approx(0.85)
