from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "MLOps AI Service is running"}

@app.post("/analyze")
def analyze(data: InputData):
    text = data.text
    
    # 1. Initialize the variable with a default or None
    result = "" 

    # 2. Run your logic
    if "error" in text.lower():
        result = "Potential issue detected"
    else:
        result = "System looks normal"

    # 3. Now the IDE knows 'result' definitely exists
    return {
        "input": text,
        "result": result
    }