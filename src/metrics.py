import evaluate

# Load once (cached by HF)
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

def compute_metrics(predictions, references):
    """
    predictions: List[str]
    references:  List[str]
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
        "ROUGE-L": rouge_scores["rougeL"],
        "BERTScore": sum(bert_scores["f1"]) / len(bert_scores["f1"])
    }
