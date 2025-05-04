from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    text = input.text
    # Giả sử nếu có "họp" hoặc "mai" là yêu cầu đặt lịch
    if "họp" in text.lower() or "mai" in text.lower():
        return {"is_schedule_request": True}
    return {"is_schedule_request": False}
