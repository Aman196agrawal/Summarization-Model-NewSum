import json

from torch.utils.data import Dataset

from src.newssumm_dataset import NewsSummDataset


def test_newssumm_dataset_is_a_torch_dataset_subclass(tmp_path):
    jsonl_path = tmp_path / "sample.jsonl"
    jsonl_path.write_text(
        json.dumps({"cluster_id": "c1", "documents": ["doc a"], "summary": "sum a"}) + "\n"
        + json.dumps({"cluster_id": "c2", "documents": ["doc b"], "summary": "sum b"}) + "\n"
    )

    dataset = NewsSummDataset(str(jsonl_path))

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2
    assert dataset[0] == {"cluster_id": "c1", "documents": ["doc a"], "summary": "sum a"}
    assert dataset[1]["cluster_id"] == "c2"
