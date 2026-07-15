import re

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text):
    """Split `text` into sentences on '.', '!', or '?' followed by whitespace.

    Returns an empty list for empty/whitespace-only input.
    """
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_BOUNDARY_RE.split(text) if s.strip()]


def build_cluster_input(documents, tokenizer):
    """Flatten every sentence across all documents in a cluster into one
    `</s>`-delimited input string, using the tokenizer's real eos token as the
    delimiter (not a fake `[SEP]`/`[DOC]` placeholder) so sentence boundaries
    are recoverable from `input_ids` after tokenization.

    Sentences are joined with the eos token *between* them, relying on the
    tokenizer's automatic end-of-sequence append to mark the final sentence's
    boundary too -- this keeps exactly one eos token per sentence.

    Returns (input_text, sentences) where `sentences` is the flat list used
    for oracle labeling and must stay index-aligned with the eos positions
    that will appear in the tokenized `input_ids`.
    """
    sentences = []
    for document in documents:
        sentences.extend(split_sentences(document))

    input_text = (" " + tokenizer.eos_token + " ").join(sentences)
    return input_text, sentences
