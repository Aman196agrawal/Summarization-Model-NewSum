from src.sentence_utils import split_sentences, build_cluster_input


class _FakeTokenizer:
    eos_token = "</s>"


def test_split_sentences_splits_on_terminal_punctuation():
    text = "Hello world. This is a test! Is this working?"
    assert split_sentences(text) == [
        "Hello world.",
        "This is a test!",
        "Is this working?",
    ]


def test_split_sentences_handles_empty_and_whitespace_only():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_split_sentences_strips_extra_whitespace_between_sentences():
    text = "First sentence.    Second sentence."
    assert split_sentences(text) == ["First sentence.", "Second sentence."]


def test_build_cluster_input_flattens_all_documents_and_joins_with_eos():
    documents = ["Sentence one. Sentence two.", "Sentence three."]
    text, sentences = build_cluster_input(documents, _FakeTokenizer())

    assert sentences == ["Sentence one.", "Sentence two.", "Sentence three."]
    assert text == "Sentence one. </s> Sentence two. </s> Sentence three."


def test_build_cluster_input_with_single_document_single_sentence():
    documents = ["Only one sentence here."]
    text, sentences = build_cluster_input(documents, _FakeTokenizer())

    assert sentences == ["Only one sentence here."]
    assert text == "Only one sentence here."
