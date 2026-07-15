import json

from scripts.collect_all_metrics import aggregate_metrics, write_outputs


def test_aggregate_metrics_reads_test_metrics_json_from_each_model_dir(tmp_path):
    model_dir = tmp_path / "longt5_baseline"
    model_dir.mkdir()
    (model_dir / "test_metrics.json").write_text(json.dumps({
        "ROUGE-1": 0.1, "ROUGE-2": 0.2, "ROUGE-L": 0.3, "BERTScore": 0.4
    }))

    result = aggregate_metrics(tmp_path)

    assert result == {
        "longt5_baseline": {"ROUGE-1": 0.1, "ROUGE-2": 0.2, "ROUGE-L": 0.3, "BERTScore": 0.4}
    }


def test_aggregate_metrics_ignores_dirs_without_test_metrics_json(tmp_path):
    (tmp_path / "empty_dir").mkdir()

    result = aggregate_metrics(tmp_path)

    assert result == {}


def test_write_outputs_creates_json_and_csv(tmp_path):
    all_metrics = {"m1": {"ROUGE-1": 0.1, "ROUGE-2": 0.2, "ROUGE-L": 0.3, "BERTScore": 0.4}}
    json_path = tmp_path / "out.json"
    csv_path = tmp_path / "out.csv"

    write_outputs(all_metrics, json_path, csv_path)

    assert json.loads(json_path.read_text()) == all_metrics
    assert "m1" in csv_path.read_text()
