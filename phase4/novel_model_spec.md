# Phase 4: Salience-Aware Hierarchical LongT5 for NewsSumm

## Motivation
Multi-document news summarization suffers from redundancy and missing key facts.
Baseline encoder-decoder models treat all sentences equally, leading to over-representation
of repeated information.

We propose a salience-aware hierarchical summarization model that explicitly predicts
sentence importance and uses it to guide summary generation.

## Architecture
- Backbone: LongT5 (encoder-decoder)
- Sentence encoder: LongT5 encoder applied at sentence level
- Salience head: Linear layer predicting importance score per sentence
- Aggregation: Salience-weighted sentence embeddings
- Decoder: Standard LongT5 decoder

## Mathematical Formulation
Let s_i be sentence embeddings from encoder.

Salience scores:
    a_i = softmax(W_s s_i)

Weighted document representation:
    h = Σ a_i * s_i

Main loss:
    L_sum = CrossEntropy(summary, generated_summary)

Auxiliary salience loss:
    L_sal = BinaryCrossEntropy(a_i, y_i)

Total loss:
    L = L_sum + λ L_sal

## Justification
- Reduces redundancy by down-weighting repeated sentences
- Encourages coverage of important facts
- Suitable for Indian multi-source news clusters
