import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans

# The 22 fixed categories NewsSumm++ maps the original ~4,485 free-text
# categories onto (ported from the author's Colab pipeline).
FIXED_CATEGORIES = [
    "politics", "business", "economy", "technology", "science",
    "health", "sports", "entertainment", "world", "education",
    "environment", "crime", "law", "defense", "travel",
    "lifestyle", "culture", "opinion", "weather",
    "finance", "energy", "media",
]

_CATEGORY_KEYWORDS = [
    ("politics", "politics"),
    ("political", "politics"),
    ("politician", "politics"),
    ("politicians", "politics"),
    ("politic", "politics"),
    ("business", "business"),
    ("businesses", "business"),
    ("economy", "economy"),
    ("economies", "economy"),
    ("economic", "economy"),
    ("technology", "technology"),
    ("technologies", "technology"),
    ("technical", "technology"),
    ("tech", "technology"),
    ("ai", "technology"),
    ("science", "science"),
    ("sciences", "science"),
    ("scientific", "science"),
    ("health", "health"),
    ("sports", "sports"),
    ("sporting", "sports"),
    ("sport", "sports"),
    ("cricket", "sports"),
    ("entertainment", "entertainment"),
    ("entertaining", "entertainment"),
    ("entertain", "entertainment"),
    ("movie", "entertainment"),
    ("movies", "entertainment"),
    ("world", "world"),
    ("international", "world"),
    ("education", "education"),
    ("educational", "education"),
    ("climate", "environment"),
    ("environment", "environment"),
    ("environmental", "environment"),
    ("crime", "crime"),
    ("crimes", "crime"),
    ("criminal", "crime"),
    ("law", "law"),
    ("laws", "law"),
    ("legal", "law"),
    ("court", "law"),
    ("courts", "law"),
    ("defense", "defense"),
    ("defenses", "defense"),
    ("military", "defense"),
    ("travel", "travel"),
    ("lifestyle", "lifestyle"),
    ("lifestyles", "lifestyle"),
    ("culture", "culture"),
    ("cultures", "culture"),
    ("cultural", "culture"),
    ("opinion", "opinion"),
    ("opinions", "opinion"),
    ("editorial", "opinion"),
    ("weather", "weather"),
    ("finance", "finance"),
    ("finances", "finance"),
    ("financial", "finance"),
    ("stock", "finance"),
    ("stocks", "finance"),
    ("energy", "energy"),
    ("energies", "energy"),
    ("media", "media"),
    ("news", "media"),
]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def map_category(text):
    text = str(text).lower()
    for keyword, category in _CATEGORY_KEYWORDS:
        if re.search(r"\b" + re.escape(keyword) + r"\b", text):
            return category
    return "other"


def remove_nulls_and_duplicates(df):
    df = df.dropna(subset=["article_text", "human_summary"])
    df = df[(df["article_text"].str.strip() != "") & (df["human_summary"].str.strip() != "")]
    df = df.drop_duplicates(subset=["article_text"])
    return df


def filter_near_duplicates(df, threshold=0.90, max_features=20000, batch_size=2000):
    """Drop rows whose TF-IDF cosine similarity to any other row exceeds
    `threshold` (keeps the first occurrence, drops the rest)."""
    texts = df["clean_article"].tolist()
    if len(texts) < 2:
        return df

    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)

    to_remove = set()
    for start in range(0, X.shape[0], batch_size):
        end = min(start + batch_size, X.shape[0])
        sim_matrix = cosine_similarity(X[start:end], X)
        for row_idx in range(sim_matrix.shape[0]):
            global_row = start + row_idx
            for col_idx in range(sim_matrix.shape[1]):
                if global_row != col_idx and global_row not in to_remove and sim_matrix[row_idx, col_idx] > threshold:
                    to_remove.add(col_idx)

    keep_positions = [i for i in range(len(df)) if i not in to_remove]
    return df.iloc[keep_positions]


def cluster_documents(df, embedder, n_clusters, batch_size=10000, random_state=42):
    """Cluster rows by semantic similarity of `headline + clean_article`.
    `embedder` is any callable(list[str]) -> array-like of embeddings --
    production code passes SentenceTransformer('all-MiniLM-L6-v2').encode.
    """
    texts = (df["headline"].astype(str) + " " + df["clean_article"]).tolist()
    embeddings = embedder(texts)

    n_clusters = max(1, min(n_clusters, len(df)))
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=random_state)

    df = df.copy()
    df["cluster_id"] = kmeans.fit_predict(embeddings)
    return df


def filter_cluster_size(df, min_docs=7, max_docs=10):
    cluster_sizes = df["cluster_id"].value_counts()
    valid_clusters = cluster_sizes[(cluster_sizes >= min_docs) & (cluster_sizes <= max_docs)].index
    return df[df["cluster_id"].isin(valid_clusters)]


def build_newssumm_pp(df, embedder, n_clusters, near_dup_threshold=0.90, min_docs=7, max_docs=10):
    """Full NewsSumm -> NewsSumm++ pipeline: null/duplicate removal, text
    normalization, near-duplicate filtering, semantic clustering, cluster-size
    filtering (7-10 docs/cluster), and category mapping to the 22 fixed
    categories -- ported from the author's working Colab notebook
    (more_about_dataset_filtering.ipynb), not previously present in this repo.
    """
    df = remove_nulls_and_duplicates(df)
    df = df.copy()
    df["clean_article"] = df["article_text"].apply(clean_text)
    df["clean_summary"] = df["human_summary"].apply(clean_text)

    df = filter_near_duplicates(df, threshold=near_dup_threshold)
    df = cluster_documents(df, embedder, n_clusters=n_clusters)
    df = filter_cluster_size(df, min_docs=min_docs, max_docs=max_docs)

    df = df.copy()
    df["mapped_category"] = df["news_category"].apply(map_category)
    df = df[df["mapped_category"] != "other"]

    return df


def main():
    from sentence_transformers import SentenceTransformer

    input_file = "data/newssumm_raw.csv"
    output_file = "data/newssumm_pp.csv"

    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip().str.replace("\n", "")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embedder = lambda texts: embedding_model.encode(texts, batch_size=64, show_progress_bar=True)

    n_clusters = max(1, len(df) // 8)
    result = build_newssumm_pp(df, embedder, n_clusters=n_clusters)

    result.to_csv(output_file, index=False)
    print(f"NewsSumm++ saved: {len(result)} rows, {result['cluster_id'].nunique()} clusters")


if __name__ == "__main__":
    main()
