from scripts.split_dataset import compute_split_indices, split_samples


def test_compute_split_indices_70_15_15_on_100_samples():
    split_train, split_val = compute_split_indices(100, train_ratio=0.70, val_ratio=0.15)

    assert split_train == 70
    assert split_val == 85


def test_split_samples_produces_70_15_15_partition():
    samples = list(range(20))

    train, val, test = split_samples(samples, train_ratio=0.70, val_ratio=0.15)

    assert len(train) == 14
    assert len(val) == 3
    assert len(test) == 3
    assert sorted(train + val + test) == samples
