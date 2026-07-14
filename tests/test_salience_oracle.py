from src.salience_oracle import compute_oracle_labels


def test_compute_oracle_labels_picks_the_matching_sentence():
    sentences = [
        "The prime minister announced a new policy today.",
        "Cricket match ends in a draw after rain.",
        "Stock markets closed higher on Friday.",
    ]
    reference = "The prime minister announced a new policy today."

    labels = compute_oracle_labels(sentences, reference)

    assert labels == [1, 0, 0]


def test_compute_oracle_labels_can_pick_more_than_one_sentence():
    sentences = [
        "The prime minister announced a new policy today.",
        "The policy focuses on rural schools and funding.",
        "Cricket match ends in a draw after rain.",
    ]
    reference = "The prime minister announced a new policy focused on rural schools and funding."

    labels = compute_oracle_labels(sentences, reference)

    assert labels[0] == 1
    assert labels[1] == 1
    assert labels[2] == 0


def test_compute_oracle_labels_returns_empty_list_for_no_sentences():
    assert compute_oracle_labels([], "some reference") == []


def test_compute_oracle_labels_respects_max_sentences_cap():
    sentences = [
        "The prime minister announced a new policy today.",
        "The policy focuses on rural schools and funding.",
        "Officials confirmed the rollout begins next year.",
    ]
    reference = "The prime minister announced a new policy focused on rural schools, funding, and next year's rollout."

    labels = compute_oracle_labels(sentences, reference, max_sentences=1)

    assert sum(labels) <= 1
