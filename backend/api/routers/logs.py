from fastapi import APIRouter, HTTPException, Query
from pathlib import Path

router = APIRouter()

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
API_LOG = LOG_DIR / "api_requests.log"
TRAIN_LOG = LOG_DIR / "training.log"

@router.get("/logs")
def list_logs():
    """Lista os logs disponíveis."""
    return {
        "available_logs": ["api_requests.log", "training.log"],
        "path": str(LOG_DIR)
    }

@router.get("/logs/api_requests")
def get_api_logs(lines: int = Query(50, ge=1, le=500)):
    """Retorna as últimas linhas do log de requisições realizados nas APIs."""
    if not API_LOG.exists():
        raise HTTPException(status_code=404, detail="api_requests.log não encontrado")
    with open(API_LOG, "r", encoding="utf-8") as f:
        content = f.readlines()
    return {"log": content[-lines:]}

@router.get("/logs/training")
def get_training_logs(lines: int = Query(50, ge=1, le=500)):
    """Retorna as últimas linhas do log de treinamento."""
    if not TRAIN_LOG.exists():
        raise HTTPException(status_code=404, detail="training.log não encontrado")
    with open(TRAIN_LOG, "r", encoding="utf-8") as f:
        content = f.readlines()
    return {"log": content[-lines:]}
