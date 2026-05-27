import chromadb
from sentence_transformers import SentenceTransformer

# Embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Persistent database path
client = chromadb.PersistentClient(
    path="/app/chroma_storage"
)

# Create or load collection
collection = client.get_or_create_collection(
    name="infrastructure_knowledge"
)


def add_document(
    text: str,
    metadata: dict,
    doc_id: str
):
    embedding = model.encode(text).tolist()

    collection.add(
        documents=[text],
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[doc_id]
    )


def search_documents(
    query: str,
    top_k: int = 3,
    filters: dict = None
):
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=filters
    )

    formatted_results = []

    for i in range(
        len(results["documents"][0])
    ):
        formatted_results.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    return formatted_results