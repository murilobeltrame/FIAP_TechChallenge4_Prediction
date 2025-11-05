from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from api.routers import markets, quotes, predict, trained_models, train_model, monitor, logs  
from pathlib import Path
import logging
import time

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
API_LOG_FILE = LOG_DIR / "api_requests.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(API_LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="📈 Stock Prediction API (FastAPI + PyTorch)",
              description="API para consulta de ações e previsão de preços utilizando modelos LSTM com PyTorch.",
              version="1.0.0",)

# CORS - em dev permitir localhost:5173 (Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # em produção restringir
    allow_methods=["*"],
    allow_headers=["*"],
)

# incluir routers
app.include_router(markets.router, prefix="/api", tags=["markets"])
app.include_router(trained_models.router, prefix="/api", tags=["trained_models"])
app.include_router(monitor.router, prefix="/api", tags=["monitor"])
app.include_router(logs.router, prefix="/api", tags=["logs"])
app.include_router(quotes.router, prefix="/api", tags=["quotes"])
app.include_router(predict.router, prefix="/api", tags=["predict"])
app.include_router(train_model.router, prefix="/api", tags=["train_model"])


# montar o build do react (após `npm run build` em frontend -> copia para backend/static)
app.mount("/", StaticFiles(directory="static", html=True), name="static",)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {duration:.3f}s - {response.status_code}")
    return response