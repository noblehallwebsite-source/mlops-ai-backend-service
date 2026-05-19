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