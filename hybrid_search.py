from rank_bm25 import BM25Okapi


# In-memory keyword store (simple version for now)
documents = []


def add_to_keyword_index(text: str):
    documents.append(text)


def build_bm25():
    tokenized_docs = [doc.lower().split() for doc in documents]
    return BM25Okapi(tokenized_docs)


def hybrid_search(query: str, top_k: int = 3):
    # this was added here instead of top of file to avoid circular 
    # import or infinite loop
    from chromadb_service import search_documents
    # 1. VECTOR SEARCH
    vector_results = search_documents(query, top_k=top_k)

    # 2. KEYWORD SEARCH
    bm25 = build_bm25()
    tokenized_query = query.lower().split()
    keyword_scores = bm25.get_scores(tokenized_query)

    keyword_ranked = sorted(
        zip(documents, keyword_scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    # 3. MERGE RESULTS
    combined = []

    for item in vector_results:
        combined.append({
            "text": item["text"],
            "source": "vector",
            "score": item["distance"]
        })

    for text, score in keyword_ranked:
        combined.append({
            "text": text,
            "source": "keyword",
            "score": float(score)
        })

    return combined