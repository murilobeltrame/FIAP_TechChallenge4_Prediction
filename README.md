# 🧠 Tech Challenge — Fase 4 (Previsão de Ações com FastAPI + PyTorch + React)

Este projeto é uma aplicação completa com **FastAPI** (backend em Python) e **React** (frontend), que permite:

📊 **Consultar informações de ações brasileiras e americanas**,  
📈 **Treinar modelos LSTM em PyTorch**, e  
🤖 **Prever preços futuros com base nos modelos treinados.**

Além disso, o sistema conta com **monitoramento de recursos (CPU, memória e GPU)** e **logs acessíveis via API**.
---

## 🚀 Funcionalidades Principais

✅ Seleção de mercado (Brasil / EUA)  
✅ Listagem dinâmica de ações por origem  
✅ Consulta de informações da ação (nome, ticker, segmento)  
✅ Gráfico com histórico dos últimos 30 dias  
✅ Treinamento de modelos LSTM em PyTorch   
✅ Predição de valores futuros  
✅ Monitoramento do sistema (CPU, RAM e GPU)  
✅ Consulta de logs via API  
✅ Métricas (MAE, RMSE e MAPE)
✅ Frontend integrado (React + Chart.js)

---

## 🧩 Estrutura do Projeto

```
FIAP_TechChallenge4_Prediction/
│
├── backend/
│   ├── api/
│   │   └── routes/
│   │       ├── logs.py                # Endpoints para consultar logs
│   │       ├── markets.py             # Listagem de ações por mercado
│   │       ├── monitor.py             # Endpoint de monitoramento do sistema
│   │       ├── predict.py             # Predição de preços usando LSTM
│   │       ├── quotes.py              # Consulta de dados e cotações
│   │       ├── train_model.py         # endpoint para treinar modelos
│   │       └── trained_models.py      # Listagem de modelos treinados
│   │
│   ├── schemas/                       # Schemas Pydantic (validação de entrada)
|   |   ├── predict.py
|   |   ├── train_model.py            
│   │   └── stock.py                   
│   │
│   ├── services/
|   |   ├── lstm_service.py            # Serviço para predição do preço de ações
│   │   └── stock_provider.py          # Serviço para buscar dados de mercado
│   │
│   ├── utils/
|   |   └── monitor.py                 # Funções para obter uso de CPU, RAM e GPU
│   │
│   ├── logs/
|   |   ├── api_requests.log           # Logs das requisições HTTP
│   │   └── training.log               # Logs dos treinos de modelo
│   │
│   ├── ml_models/                     # Modelos treinados (um diretório por ticker)
│   │   └── PETR4.SA/
│   │       ├── model.pt
│   │       └── scaler.pkl
│   │
│   ├── static/                        # Frontend build (React)
│   ├── train_lstm.py                  # Script de treino em PyTorch (LSTM)
│   ├── training.log                   # Log dos treinos executados via API
│   ├── main.py                        # Ponto de entrada do backend FastAPI
│   └── pyproject.toml
│
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── MarketSelector.jsx
    │   │   ├── QuoteResult.jsx
    │   │   ├── TickerSelector.jsx
    │   │   └── Prediction.jsx          # Exibe previsões gráficas
    │   ├── App.jsx                     # Estrutura principal do app React
    │   └── main.jsx
    ├── package.json
    ├── vite.config.js
    └── index.html
```

---

## ⚙️ Instalação e Execução

### 1️⃣ Clonar o repositório

```bash
git clone https://github.com/milerazevedo0/FIAP_TechChallenge4_Prediction.git
cd FIAP_TechChallenge4_Prediction
```

---

### 2️⃣ Backend (FastAPI + Poetry + PyTorch)

> Certifique-se de ter o **Python 3.10+** e o **Poetry** instalados.  
> Instale o Poetry seguindo as instruções: https://python-poetry.org/docs/#installation

#### Instalar dependências e ativar o ambiente virtual:
```bash
cd backend
poetry install
poetry shell
```

#### Rodar o servidor FastAPI:
```bash
poetry run uvicorn main:app --reload
```

- API: `http://127.0.0.1:8000`  
- Swagger Docs: `http://127.0.0.1:8000/docs`  

---

### 3️⃣ Frontend (React + Vite)

Abra outro terminal:
```bash
cd frontend
npm install
npm run dev
```

O frontend será iniciado em:  
➡️ [http://localhost:5173](http://localhost:5173)

---

### 4️⃣ Servir o frontend pelo backend (modo produção)

Após gerar o build:
```bash
npm run build
```

Copie o conteúdo de `frontend/dist/` para `backend/static/`:
```bash
# Linux/Mac
cp -r frontend/dist/* backend/static/
# Windows
xcopy frontend\dist backend\static /E /I /Y
```

Agora o backend servirá o frontend diretamente em:  
➡️ [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

---

## 🧠 Endpoints (Modelos e Predições)

### 🔹 `/api/train` → Inicia um treino de modelo
**POST**
```json
{
  "symbol": "PETR4.SA",
  "epochs": 60,
  "look_back": 60,
  "batch_size": 32
}
```

📤 **Resposta:**
```json
{
  "status": "success",
  "message": "Treinamento do modelo para PETR4.SA concluído.",
  "result": {
    "symbol": "PETR4.SA",
    "mae": 0.3963932991027832,
    "rmse": 0.5205498082609175
  }
}
```

---

### 🔹 `/api/predict` → Realiza predição para um modelo já treinado

**POST**
```json
{
  "ticker": "PETR4.SA",
  "days": 7
}
```

📥 **Resposta:**
```json
{
  "ticker": "PETR4.SA",
  "predictions": [
    {"date": "2025-11-01", "predicted": 37.85},
    {"date": "2025-11-02", "predicted": 38.12},
    {"date": "2025-11-03", "predicted": 38.54}
  ]
}
```
---

---

### 🔹 `/api/monitor` → Retorna métricas de sistema:

📥 **Resposta:**
```json
{
  "status": "ok",
  "metrics": {
    "cpu_percent": 12.5,
    "memory_percent": 58.3,
    "gpu_name": "NVIDIA RTX 3060",
    "gpu_memory_allocated_MB": 412.25,
    "gpu_memory_total_MB": 6144.00
  }
}

```
---

---

### 🔹 `/api/logs` → Listar e consultar logs salvos:

##/api/logs → Lista logs disponíveis

##/api/logs/api_requests → Últimas requisições da API

##/api/logs/training → Últimos treinos registrados

📁 Todos os logs ficam em:
backend/logs/
├── api_requests.log
└── training.log

Exemplo log de treinamento:
```
[2025-11-05 13:39:24] 🚀 Iniciando treino: PETR4.SA
[2025-11-05 13:39:36] Treino finalizado: PETR4.SA | MAE=0.4398 | RMSE=0.5693 | MAPE=1.37%
```
Exemplo log de requisições:
```
2025-11-05 13:38:46,423 | INFO | GET /api/monitor - 0.520s - 200
2025-11-05 13:39:36,723 | INFO | POST /api/train - 23.328s - 200
```
## 📊 Monitoramento e Logs

| Tipo | Arquivo | Endpoint |
|------|----------|-----------|
| Requisições API | `logs/api_requests.log` | `/api/logs/api` |
| Treinos LSTM | `logs/training.log` | `/api/logs/training` |
| Recursos do sistema | — | `/api/monitor` |
```
---

## 🧠 Tecnologias Utilizadas

**Backend**
- Python 3.10+
- FastAPI
- PyTorch
- yfinance
- scikit-learn
- Uvicorn
- Poetry
- Pydantic

**Frontend**
- React
- Vite
- Chart.js
- Fetch API

---

## 🧰 Próximos Passos

- Adicionar Docker para build completo (API + Frontend)
- Criar cache de resultados de predições
- Adicionar autenticação JWT (usuário/treino)
- Dashboard com histórico de treinos

---

## 🧑‍💻 Autores

| Nome | 
|------|
| **Miler Azevedo** | 
| **Arthur** | 
| **Murilo** | 
| **Kaio** | 

---

> 📘 Projeto desenvolvido para o **FIAP Tech Challenge — Fase 4**,  
> com foco em integração entre **APIs** e **Machine Learning (PyTorch)**.
