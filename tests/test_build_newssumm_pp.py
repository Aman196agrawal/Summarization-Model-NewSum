import numpy as np
import pandas as pd

from scripts.build_newssumm_pp import (
    clean_text,
    map_category,
    remove_nulls_and_duplicates,
    filter_near_duplicates,
    filter_cluster_size,
    build_newssumm_pp,
)


def test_clean_text_lowercases_and_strips_punctuation():
    assert clean_text("Hello, World!! 123") == "hello world 123"


def test_map_category_matches_known_keywords():
    assert map_category("Local Politics News") == "politics"
    assert map_category("Cricket World Cup Final") == "sports"
    assert map_category("Completely unrelated topic xyz") == "other"
    # Regression: "ai" must match only as a whole word, not as a mid-word
    # substring of ordinary words like "said"/"campaign" (word-boundary fix).
    assert map_category("He said the campaign would continue") == "other"
    # Regression: plural/inflected forms must still match after switching to
    # strict word-boundary matching (previously covered by substring matching).
    assert map_category("Rising crimes in the city") == "crime"
    assert map_category("New laws passed by the government") == "law"
    assert map_category("The courts will decide the case") == "law"
    assert map_category("Local businesses struggle") == "business"
    assert map_category("The team's defenses were tested today") == "defense"
    assert map_category("Latest movies released this year") == "entertainment"
    assert map_category("Several criminals were arrested today") == "crime"
    assert map_category("Editorials criticized the new policy") == "opinion"


def test_remove_nulls_and_duplicates_drops_null_empty_and_duplicate_rows():
    df = pd.DataFrame({
        "article_text": ["a", "a", None, "b", ""],
        "human_summary": ["s1", "s1", "s2", "s3", "s4"],
    })

    result = remove_nulls_and_duplicates(df)

    assert result["article_text"].tolist() == ["a", "b"]


def test_filter_near_duplicates_removes_highly_similar_rows():
    df = pd.DataFrame({
        "clean_article": [
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps over the lazy dog today",
            "completely different content about space exploration",
        ]
    })

    result = filter_near_duplicates(df, threshold=0.80)

    assert len(result) == 2


def test_filter_cluster_size_keeps_only_clusters_in_range():
    df = pd.DataFrame({"cluster_id": [0, 0, 0] + [1] * 9 + [2]})

    result = filter_cluster_size(df, min_docs=7, max_docs=10)

    assert set(result["cluster_id"].unique()) == {1}
    assert len(result) == 9


def test_build_newssumm_pp_end_to_end_with_fake_embedder():
    df = pd.DataFrame({
        "article_text": [f"unique political article number {i + 10} about elections" for i in range(7)]
                        + [f"unique sports article number {i + 10} about cricket matches" for i in range(7)],
        "human_summary": [f"summary {i}" for i in range(14)],
        "headline": [f"headline {i}" for i in range(14)],
        "news_category": (["Politics"] * 7) + (["Sports"] * 7),
    })

    def fake_embedder(texts):
        return np.array([[0.0, 0.0] if "political" in t else [1.0, 1.0] for t in texts])

    result = build_newssumm_pp(
        df, fake_embedder, n_clusters=2, near_dup_threshold=0.99, min_docs=7, max_docs=10
    )

    assert set(result["mapped_category"].unique()) <= {"politics", "sports"}
    assert result["cluster_id"].nunique() == 2
    assert len(result) == 14
