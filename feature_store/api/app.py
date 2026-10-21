from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.predict import predict_fraud

app = FastAPI(title="Fraud Detection API")
BASE_DIR = Path(__file__).resolve().parent


class PredictionRequest(BaseModel):
    customer_id: int
    force_model: Optional[str] = None


@app.get("/")
def home():
    return {"message": "Real-Time Fraud Detection API"}


@app.get("/test")
def test_page():
    return FileResponse(BASE_DIR / "test_page.html")


@app.post("/predict")
def predict(request: PredictionRequest):
    return predict_fraud(request.customer_id, force_model=request.force_model)
