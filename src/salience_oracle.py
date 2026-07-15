from rouge_score import rouge_scorer

_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)


def _combined_rouge_f1(candidate_sentences, reference_summary):
    if not candidate_sentences:
        return 0.0
    candidate = " ".join(candidate_sentences)
    scores = _SCORER.score(reference_summary, candidate)
    return (scores["rouge1"].fmeasure + scores["rouge2"].fmeasure) / 2.0


def compute_oracle_labels(sentences, reference_summary, max_sentences=None):
    """Greedy ROUGE-oracle extractive labeling.

    Returns a 0/1 label per sentence in `sentences` (same length, same
    order), marking sentences that were greedily selected because adding
    them to the running selection improved the combined ROUGE-1/ROUGE-2 F1
    against `reference_summary`. Selection stops when no remaining sentence
    improves the score, or when `max_sentences` selections have been made.
    """
    n = len(sentences)
    if n == 0:
        return []

    limit = n if max_sentences is None else max_sentences
    labels = [0] * n
    selected_indices = []
    remaining = set(range(n))
    best_score = 0.0

    while remaining and len(selected_indices) < limit:
        best_idx = None
        best_candidate_score = best_score

        for idx in remaining:
            candidate_score = _combined_rouge_f1(
                [sentences[i] for i in selected_indices] + [sentences[idx]],
                reference_summary,
            )
            if candidate_score > best_candidate_score:
                best_candidate_score = candidate_score
                best_idx = idx

        if best_idx is None:
            break

        selected_indices.append(best_idx)
        remaining.discard(best_idx)
        labels[best_idx] = 1
        best_score = best_candidate_score

    return labels
