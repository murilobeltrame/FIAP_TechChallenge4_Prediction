from fastapi import APIRouter, HTTPException
from schemas.train_model import TrainRequest
from pathlib import Path
import sys
import traceback
from datetime import datetime
from train_lstm import train

router = APIRouter()

def save_log(message: str):
    """Salva logs em backend/training.log"""
    log_file = Path(__file__).resolve().parents[2] / "training.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

@router.post("/train")
async def train_model(request: TrainRequest):
    """
    Treina um novo modelo LSTM para o ticker informado.
    Exemplo:
    {
      "symbol": "PETR4.SA",
      "epochs": 60,
      "look_back": 60,
      "batch_size": 32
    }
    """
    try:
        model_dir = str(Path(__file__).resolve().parents[2] / "ml_models")

        result = train(
            symbol=request.symbol,
            start=request.start,
            end=request.end,
            look_back=request.look_back,
            epochs=request.epochs,
            batch_size=request.batch_size,
            model_dir=model_dir
        )
        save_log(
            f"✅ Treino finalizado: {request.symbol} | MAE={result['mae']:.4f} | RMSE={result['rmse']:.4f}"
            )

        return {
            "status": "success",
            "message": f"Treinamento do modelo para {request.symbol} concluído.",
            "result": result
        }

    except Exception as e:
        save_log(f"❌ ERRO durante treino para {request.symbol}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao treinar modelo: {str(e)}")
