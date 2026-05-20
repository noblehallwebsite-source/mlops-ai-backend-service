import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

# Lightweight embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

class VectorMemoryStore:

    def __init__(self):

        # Embedding dimension for all-MiniLM-L6-v2
        self.dimension = 384

        # FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)

        # Store original text separately
        self.text_metadata = []

    def add_document(self, text: str):

        # Convert text to embedding vector
        embedding = model.encode([text])

        # Convert to float32 for FAISS compatibility
        vector = np.array(embedding).astype("float32")

        # Store vector
        self.index.add(vector)

        # Store original text
        self.text_metadata.append(text)

    def search(self, query: str, top_k: int = 3):

        if self.index.ntotal == 0:
            return []

        # Convert query into embedding
        query_embedding = model.encode([query])

        query_vector = np.array(query_embedding).astype("float32")

        # Perform semantic search
        distances, indices = self.index.search(
            query_vector,
            top_k
        )

        results = []

        for distance, idx in zip(
            distances[0],
            indices[0]
        ):

            if idx != -1:

                results.append({
                    "text": self.text_metadata[idx],
                    "distance": float(distance)
                })

        return results


# Global memory store
memory_store = VectorMemoryStore()