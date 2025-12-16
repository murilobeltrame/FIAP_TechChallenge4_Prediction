from fastapi import APIRouter, HTTPException
from schemas.train_model import TrainRequest
from pathlib import Path
import sys
import traceback
from datetime import datetime
import os

# Get absolute path to backend root directory
backend_root = Path(os.environ.get("BACKEND_ROOT", Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(backend_root))

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
        # Lazy import to ensure PYTHONPATH is properly set
        from train_lstm import train
        
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

        formatted_result = {
            "symbol": result["symbol"],
            "mae": round(result["mae"], 4),
            "rmse": round(result["rmse"], 4),
            "mape": f"{result['mape']:.2f}%"
        }

        return {
            "status": "success",
            "message": f"Treinamento do modelo para {request.symbol} concluído.",
            "result": formatted_result
        }

    except Exception as e:
        save_log(f"❌ ERRO durante treino para {request.symbol}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao treinar modelo: {str(e)}")
