from sentence_transformers import CrossEncoder

# Lightweight reranker model
model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank(query: str, results: list, top_k: int = 3):
    """
    Takes hybrid search results and reorders them
    based on true semantic relevance.
    """

    if not results:
        return []

    # Prepare pairs for the Cross-Encoder: [(query, doc1), (query, doc2), ...]
    pairs = [
        (query, item["text"])
        for item in results
    ]

    # Predict relevance scores for all pairs at once
    scores = model.predict(pairs)

    reranked = []

    for item, score in zip(results, scores):
        reranked.append({
            **item,
            "rerank_score": float(score)
        })

    # Sort documents so the highest relevance score is at the top
    reranked.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked[:top_k]