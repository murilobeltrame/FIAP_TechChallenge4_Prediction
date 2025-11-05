from fastapi import APIRouter
from utils.monitor import get_system_metrics

router = APIRouter()

@router.get("/monitor")
def monitor_system():
    """
    Retorna métricas de monitoramento do sistema.
    """
    return {"status": "ok", "metrics": get_system_metrics()}
