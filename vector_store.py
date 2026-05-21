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
from sentence_transformers import SentenceTransformer

# Lightweight embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

class VectorMemoryStore:

    def __init__(self):
        # Embedding dimension for all-MiniLM-L6-v2
        self.dimension = 384

        # Define file paths inside your persistent docker volume directory
        self.index_file = "/app/storage/faiss.index"
        self.metadata_file = "/app/storage/metadata.json"

        # Ensure the directory exists inside the container volume path
        os.makedirs("/app/storage", exist_ok=True)

        # 🔄 RECOVER: Load existing vector database if it exists on your hard drive
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            print("💾 Found existing knowledge base. Loading vectors...")
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, "r") as f:
                self.text_metadata = json.load(f)
        else:
            print("✨ Creating fresh in-memory FAISS index...")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.text_metadata = []

    def save_to_disk(self):
        """Saves current memory index straight to the mounted Docker Volume"""
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, "w") as f:
            json.dump(self.text_metadata, f)
        print("💾 Knowledge base safely backed up to host server folder.")

    def add_document(self, text: str):
        # Convert text to embedding vector
        embedding = model.encode([text])

        # Convert to float32 for FAISS compatibility
        vector = np.array(embedding).astype("float32")

        # Store vector
        self.index.add(vector)

        # Store original text
        self.text_metadata.append(text)

        # 🔥 PERSIST IMMEDIATELY: Save data to disk so it survives restarts
        self.save_to_disk()

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
        for distance, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.text_metadata):
                results.append({
                    "text": self.text_metadata[idx],
                    "distance": float(distance)
                })

        return results

# Global memory store instance
memory_store = VectorMemoryStore()