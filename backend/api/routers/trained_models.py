# backend/routes/trained_models.py
from fastapi import APIRouter, HTTPException
from pathlib import Path

router = APIRouter()

ML_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "ml_models"

@router.get("/trained-tickers")
async def list_trained_tickers():
    """
    Retorna uma lista com todos os tickers que possuem modelo treinado.
    Cada subpasta dentro de ml_models é considerada um ticker treinado.
    """
    if not ML_MODELS_DIR.exists():
        raise HTTPException(status_code=404, detail="Diretório de modelos não encontrado")

    tickers = [d.name for d in ML_MODELS_DIR.iterdir() if d.is_dir()]

    if not tickers:
        raise HTTPException(status_code=404, detail="Nenhum modelo treinado encontrado")

    return {"trained_tickers": sorted(tickers)}
