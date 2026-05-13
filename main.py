from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

Instrumentator().instrument(app).expose(app)

class InputData(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "AI Service Running"}

@app.post("/analyze")
def analyze(data: InputData):

    if "error" in data.text.lower():
        result = "Potential issue detected"
    else:
        result = "System normal"

    return {
        "input": data.text,
        "result": result
    }