# import faiss
# import numpy as np

# from sentence_transformers import SentenceTransformer

# # Lightweight embedding model
# model = SentenceTransformer(
#     "sentence-transformers/all-MiniLM-L6-v2"
# )

# class VectorMemoryStore:

#     def __init__(self):

#         # Embedding dimension for all-MiniLM-L6-v2
#         self.dimension = 384

#         # FAISS index
#         self.index = faiss.IndexFlatL2(self.dimension)

#         # Store original text separately
#         self.text_metadata = []

#     def add_document(self, text: str):

#         # Convert text to embedding vector
#         embedding = model.encode([text])

#         # Convert to float32 for FAISS compatibility
#         vector = np.array(embedding).astype("float32")

#         # Store vector
#         self.index.add(vector)

#         # Store original text
#         self.text_metadata.append(text)

#     def search(self, query: str, top_k: int = 3):

#         if self.index.ntotal == 0:
#             return []

#         # Convert query into embedding
#         query_embedding = model.encode([query])

#         query_vector = np.array(query_embedding).astype("float32")

#         # Perform semantic search
#         distances, indices = self.index.search(
#             query_vector,
#             top_k
#         )

#         results = []

#         for distance, idx in zip(
#             distances[0],
#             indices[0]
#         ):

#             if idx != -1:

#                 results.append({
#                     "text": self.text_metadata[idx],
#                     "distance": float(distance)
#                 })

#         return results


# # Global memory store
# memory_store = VectorMemoryStore()
# the above is for when the faiss is stored in memory and is temporary below is when its stored in storage

import os
import json
import faiss
import numpy as np

class VectorMemoryStore:

    def __init__(self):

        self.dimension = 384

        self.index_file = "storage/faiss.index"
        self.metadata_file = "storage/metadata.json"

        # Load existing index if present
        if os.path.exists(self.index_file):

            self.index = faiss.read_index(
                self.index_file
            )

        else:

            self.index = faiss.IndexFlatL2(
                self.dimension
            )

        # Load metadata if present
        if os.path.exists(self.metadata_file):

            with open(self.metadata_file, "r") as f:
                self.text_metadata = json.load(f)

        else:
            self.text_metadata = []

    def save(self):

        # Save FAISS index
        faiss.write_index(
            self.index,
            self.index_file
        )

        # Save metadata
        with open(self.metadata_file, "w") as f:
            json.dump(
                self.text_metadata,
                f
            )

    def add_document(self, text: str):

        vector = model.encode(
            [text]
        ).astype("float32")

        self.index.add(vector)

        self.text_metadata.append(text)

        # Persist immediately
        self.save()

    def search(self, query: str, top_k: int = 3):

        if self.index.ntotal == 0:
            return []

        query_vector = model.encode(
            [query]
        ).astype("float32")

        distances, indices = self.index.search(
            query_vector,
            top_k
        )

        results = []

        for dist, idx in zip(
            distances[0],
            indices[0]
        ):

            if idx != -1:

                results.append({
                    "text": self.text_metadata[idx],
                    "distance": float(dist)
                })

        return results
# Global application memory store instance
memory_store = VectorMemoryStore()