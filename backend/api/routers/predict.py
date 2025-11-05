from fastapi import APIRouter, HTTPException
from typing import Optional
from services.lstm_service import predict_future
from schemas.predict import PredictRequest, PredictResponse
import yfinance as yf
import datetime

router = APIRouter()

@router.post("/predict")
async def predict(req: PredictRequest, response_model=PredictResponse):
    """
    Com base no modelo treinado, retorna a predição de preços futuros para a ação selecionada.
    """
    try:
        preds = predict_future(req.ticker, req.days, req.look_back)
        df = yf.download(req.ticker, period="2y")
        last_date = df.index[-1]
        dates = [(last_date + datetime.timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(len(preds))]
        return {"ticker": req.ticker, "predictions": [{"date": d, "predicted": p} for d,p in zip(dates,preds)]}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
