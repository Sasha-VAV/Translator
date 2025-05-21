from typing import Optional

from pydantic import BaseModel
from fastapi import FastAPI
from Translator import Writer

app = FastAPI()
model = Writer.from_pretrained()

class WriteModel(BaseModel):
    text: str
    temperature: Optional[float] = 2.

@app.get("/")
def root():
    return """Available endpoints:
    POST /write {"text": "Starting phrase", "temperature": 2}
    """

@app.post("/write")
def write(write_model: WriteModel):
    return model(input_seq=write_model.text, temperature=write_model.temperature)