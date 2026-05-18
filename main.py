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

load_dotenv()

app = FastAPI()

Instrumentator().instrument(app).expose(app)

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

class InputData(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "AI Service Running"}

@app.post("/analyze")
def analyze(data: InputData):

    response = client.chat.completions.create(
        model="llama3-8b-8192",
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