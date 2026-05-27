# from fastapi import FastAPI
# from pydantic import BaseModel
# from prometheus_fastapi_instrumentator import Instrumentator

# app = FastAPI()

# Instrumentator().instrument(app).expose(app)

# class InputData(BaseModel):
#     text: str

# @app.get("/")
# def root():
#     return {"message": "AI Service Running"}

# @app.post("/analyze")
# def analyze(data: InputData):
#     if "error" in data.text.lower():
#         result = "Potential issue detected"
#     else:
#         result = "System normal"

#     return {
#         "input": data.text,
#         "result": result
#     }

import os

from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

from dotenv import load_dotenv
from openai import OpenAI

from embedding_service import (
    generate_embedding,
    calculate_similarity
)
from chunking_service import chunk_text

# The below is when i want to use vector_store.py which is for
# faise raw persistent store
# from vector_store import memory_store

from chromadb_service import (
    add_document,
    search_documents
)

# ADD THESE TWO IMPORTS
from hybrid_search import hybrid_search
from reranker import rerank

load_dotenv()

app = FastAPI()

Instrumentator().instrument(app).expose(app)

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


# =========================
# Request Models
# =========================

class InputData(BaseModel):
    text: str


class EmbeddingRequest(BaseModel):
    text: str


class SimilarityRequest(BaseModel):
    text1: str
    text2: str

# This is for the faiss persistent store
# class DocumentRequest(BaseModel):
    # text: str

# This is for the new chromadb
class DocumentRequest(BaseModel):
    text: str
    source: str
    environment: str
    severity: str

class SearchRequest(BaseModel):
    query: str

class FilteredSearchRequest(BaseModel):
    query: str
    environment: str

class HybridSearchRequest(BaseModel):
    query: str

class SmartSearchRequest(BaseModel):
    query: str

class RagQueryRequest(BaseModel):
    query: str

class LargeDocumentRequest(BaseModel):
    text: str

# =========================
# Root Endpoint
# =========================

@app.get("/")
def root():
    return {
        "message": "AI Service Running"
    }


# =========================
# LLM Analysis Endpoint
# =========================

@app.post("/analyze")
def analyze(data: InputData):

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are an infrastructure AI assistant."
            },
            {
                "role": "user",
                "content": data.text
            }
        ]
    )

    result = response.choices[0].message.content

    return {
        "input": data.text,
        "ai_response": result
    }


# =========================
# Embedding Endpoint
# =========================

@app.post("/embedding")
def embedding(data: EmbeddingRequest):

    vector = generate_embedding(data.text)

    return {
        "text": data.text,
        "embedding_dimension": len(vector),
        "embedding": vector
    }


# =========================
# Similarity Endpoint
# =========================

@app.post("/similarity")
def similarity(data: SimilarityRequest):

    score = calculate_similarity(
        data.text1,
        data.text2
    )

    return {
        "text1": data.text1,
        "text2": data.text2,
        "similarity_score": score
    }

# ===========================
# Document storage endpoint 
# ===========================
# THis for the old faiss storage
# @app.post("/documents")
# def add_document(data: DocumentRequest):

#     memory_store.add_document(data.text)

#     return {
#         "message": "Document stored successfully",
#         "stored_text": data.text
#     }


# This is for the new chromadb
@app.post("/documents")
def store_document(data: DocumentRequest):

    import uuid

    metadata = {
        "source": data.source,
        "environment": data.environment,
        "severity": data.severity
    }

    add_document(
        text=data.text,
        metadata=metadata,
        doc_id=str(uuid.uuid4())
    )

    return {
        "message": "Document stored successfully",
        "metadata": metadata
    }

# ===========================
# Search endpoint 
# ===========================
# @app.post("/search")
# def search_documents(data: SearchRequest):

#     results = memory_store.search(data.query)

#     return {
#         "query": data.query,
#         "results": results
#     }


# New chromadb search
@app.post("/search")
def search(data: SearchRequest):

    results = search_documents(
        query=data.query
    )

    return {
        "query": data.query,
        "results": results
    }

@app.post("/search-filtered")
def filtered_search(
    data: FilteredSearchRequest
):

    results = search_documents(
        query=data.query,
        filters={
            "environment": data.environment
        }
    )

    return {
        "query": data.query,
        "filters": {
            "environment": data.environment
        },
        "results": results
    }


@app.post("/hybrid-search")
def hybrid(data: HybridSearchRequest):

    results = hybrid_search(data.query)

    return {
        "query": data.query,
        "results": results
    }

# ADD THE NEW ENDPOINT
@app.post("/search-smart")
def smart_search(data: SmartSearchRequest):

    # 1. Hybrid retrieval (broad recall - gathers candidates from vector + keyword)
    candidates = hybrid_search(data.query)

    # 2. Rerank (precision layer - compares query directly against each text block)
    top_results = rerank(data.query, candidates)

    return {
        "query": data.query,
        "results": top_results
    }



# ===========================
# RAG endpoint 
# ===========================
@app.post("/rag")
def rag_query(data: RagQueryRequest):

    # 1. Search vector memory
    search_results = memory_store.search(
        data.query,
        top_k=3
    )

    # 2. Combine retrieved context
    context = "\n".join([
        item["text"]
        for item in search_results
    ])

    # 3. Build augmented prompt
    augmented_prompt = f"""
You are an AI infrastructure assistant.

Use the provided infrastructure context
to answer the user's question.

Context:
{context}

User Question:
{data.query}
"""

    # 4. Send to LLM
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a Kubernetes and infrastructure AI assistant."
            },
            {
                "role": "user",
                "content": augmented_prompt
            }
        ]
    )

    answer = response.choices[0].message.content

    return {
        "query": data.query,
        "retrieved_context": search_results,
        "ai_answer": answer
    }


@app.post("/add-large-document")
def add_large_document(
    data: LargeDocumentRequest
):

    chunks = chunk_text(
        data.text
    )

    for chunk in chunks:

        memory_store.add_document(
            chunk
        )

    return {
        "message": "Large document added",
        "chunks_created": len(chunks)
    }