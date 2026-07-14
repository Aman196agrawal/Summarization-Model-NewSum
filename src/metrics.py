import evaluate

# Load once (cached by HF)
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

def compute_metrics(predictions, references):
    """
    Compute evaluation metrics for summarization.

    Metrics computed:
      - ROUGE-1 (F1): Unigram overlap between prediction and reference.
      - ROUGE-2 (F1): Bigram overlap between prediction and reference.
      - ROUGE-L (F1): Longest common subsequence-based overlap.
      - BERTScore (F1): Semantic similarity using contextual BERT embeddings.

    Note: BLEU and METEOR are NOT used in this project. Only ROUGE and BERTScore
    are reported, as described in the accompanying research paper.

    Args:
        predictions (List[str]): List of model-generated summaries.
        references  (List[str]): List of reference (gold) summaries.

    Returns:
        dict: {
            "ROUGE-1": float,
            "ROUGE-2": float,
            "ROUGE-L": float,
            "BERTScore": float  # mean F1 across all samples
        }
    """

    rouge_scores = rouge.compute(
        predictions=predictions,
        references=references
    )

    bert_scores = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en"
    )

    return {
        "ROUGE-1": rouge_scores["rouge1"],
        "ROUGE-2": rouge_scores["rouge2"],
        "ROUGE-L": rouge_scores["rougeLsum"],
        "BERTScore": sum(bert_scores["f1"]) / len(bert_scores["f1"])
    }
