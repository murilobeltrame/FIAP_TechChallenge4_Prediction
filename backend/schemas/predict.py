# backend/app/schemas/predict.py
from pydantic import BaseModel
from typing import List, Optional

class PredictRequest(BaseModel):
    ticker: str
    days: Optional[int] = 7
    look_back: Optional[int] = 60

class PredictionItem(BaseModel):
    date: str
    predicted: float

class PredictResponse(BaseModel):
    ticker: str
    predictions: List[PredictionItem]
