from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from api.routers import markets, quotes, predict, trained_models, train_model  

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
app.include_router(quotes.router, prefix="/api", tags=["quotes"])
app.include_router(predict.router, prefix="/api", tags=["predict"])
app.include_router(train_model.router, prefix="/api", tags=["train_model"])


# montar o build do react (após `npm run build` em frontend -> copia para backend/static)
app.mount("/", StaticFiles(directory="static", html=True), name="static",)
