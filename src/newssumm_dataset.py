import json

class NewsSummDataset:
    """
    Dataset loader for NewsSumm multi-document summarization.
    """

    def __init__(self, jsonl_path):
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "cluster_id": sample["cluster_id"],
            "documents": sample["documents"],
            "summary": sample["summary"]
        }
