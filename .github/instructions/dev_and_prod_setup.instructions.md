---
alwaysApply: true
always_on: true
trigger: always_on
applyTo: "**"
description: Development and Production Setup Guidelines
---

# FIAP Tech Challenge 4 — Development & Production Setup

This document provides comprehensive guidelines for running the application in both development and production environments, covering the full-stack setup for the Stock Prediction API (FastAPI + PyTorch + React).

## 📋 Project Overview

**Technology Stack:**
- **Backend:** FastAPI + PyTorch + Uvicorn (Python 3.12+)
- **Frontend:** React + Vite + Chart.js (Node 24+)
- **ML Models:** LSTM-based predictions with scikit-learn scalers
- **Monitoring:** Prometheus + psutil (CPU, Memory, GPU metrics)
- **Package Management:** Poetry (Python), npm (Node)

---

## 🔧 Prerequisites

### System Requirements
- Python 3.12+ (3.15 not supported)
- Node.js 24+
- Poetry >= 2.0.0
- Git

### Optional but Recommended
- Docker & Docker Compose (for containerized deployment)
- NVIDIA CUDA (for GPU acceleration with PyTorch)
- Virtual environment manager (Poetry handles this automatically)

---

## 🚀 Development Setup

### Backend Development

#### 1. Install Dependencies
```bash
cd backend
poetry install
```

**What this does:**
- Creates isolated Python virtual environment
- Installs all dependencies from `pyproject.toml`:
  - FastAPI, Uvicorn (API framework)
  - PyTorch (ML/LSTM)
  - yfinance, investpy (market data)
  - Pandas, scikit-learn, numpy (data processing)
  - Pydantic (validation)
  - prometheus-fastapi-instrumentator (metrics)
  - pytest, pytest-custom_exit_code (testing)

#### 2. Activate Poetry Environment
```bash
poetry shell
```

Or run commands directly with `poetry run`:
```bash
poetry run uvicorn main:app --reload
```

#### 3. Run Backend Development Server
```bash
# With hot reload enabled (watches file changes)
poetry run uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Or from within activated poetry shell
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

**Server Details:**
- API Base URL: `http://127.0.0.1:8000`
- Swagger UI Documentation: `http://127.0.0.1:8000/docs`
- ReDoc Documentation: `http://127.0.0.1:8000/redoc`
- OpenAPI Schema: `http://127.0.0.1:8000/openapi.json`

**Key Features:**
- CORS enabled for all origins (use restrictive settings in production)
- Request/response logging to `logs/api_requests.log`
- Automatic API documentation generation
- Hot reload for development efficiency

### Frontend Development

#### 1. Install Dependencies
```bash
cd frontend
npm install
```

**What this does:**
- Installs Node dependencies:
  - React 19+
  - Vite (build tool & dev server)
  - Chart.js (data visualization)
  - ESLint (code quality)
  - React SWC plugin (fast refresh)

#### 2. Run Development Server
```bash
npm run dev
```

**Server Details:**
- Dev Server URL: `http://localhost:5173`
- Features:
  - Hot Module Replacement (HMR)
  - Fast refresh on file changes
  - ESLint integration
  - Built-in error overlay

#### 3. Lint Code
```bash
npm run lint
```

Checks code quality using ESLint configuration.

### Full-Stack Development Mode

To run both services simultaneously:

**Terminal 1 - Backend:**
```bash
cd backend
poetry install
poetry shell
uvicorn main:app --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install
npm run dev
```

**Access the Application:**
- Frontend: `http://localhost:5173`
- Backend API: `http://127.0.0.1:8000`
- API Docs: `http://127.0.0.1:8000/docs`

---

## 📦 Production Setup

### Backend Production

#### 1. Install Dependencies (Production)
```bash
cd backend
poetry install --only main
```

Or disable dev dependencies:
```bash
poetry install --no-dev
```

#### 2. Run Backend Production Server

**Standard Production (Single Process):**
```bash
poetry run uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Configuration Parameters:**
- `--host 0.0.0.0`: Listen on all interfaces (required for containerized deployments)
- `--port 8000`: Production port (change if needed)
- `--workers 4`: Number of Uvicorn worker processes (adjust based on CPU cores)
- Remove `--reload` flag (never use in production)

**Environment Configuration:**
Create a `.env` file in the `backend/` directory:
```bash
# Security & CORS
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
DEBUG=false

# ML/Training settings
MODEL_DIR=./ml_models
BATCH_SIZE=32
EPOCHS=100

# Monitoring
ENABLE_METRICS=true
```

**Update CORS Settings:**
In `backend/main.py`, change:
```python
# Development (current - open to all)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Production (recommended - restrict origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
```

#### 3. Serve Logs
Logs are automatically written to:
- **API Logs:** `backend/logs/api_requests.log`
- **Training Logs:** `backend/logs/training.log`

Access via API endpoint: `GET /api/logs/`

### Frontend Production

#### 1. Build for Production
```bash
cd frontend
npm run build
```

**Output:**
- Build artifacts generated in `frontend/dist/`
- Optimized, minified JavaScript and CSS
- Static assets ready for deployment

#### 2. Deploy Frontend with Backend (Integrated Mode)

**Option A: Copy Build to Backend Static Directory**

```bash
# Linux/macOS
cp -r frontend/dist/* backend/static/

# Windows (PowerShell)
Copy-Item -Path frontend\dist\* -Destination backend\static\ -Recurse -Force

# Windows (CMD)
xcopy frontend\dist backend\static /E /I /Y
```

Then serve everything from backend:
```bash
# Frontend will be accessible at http://your-domain.com/
cd backend
poetry run uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Option B: Serve Frontend Separately (Recommended for Scale)**

Use a CDN or separate web server for static files:
```bash
# Using nginx
server {
    listen 80;
    server_name yourdomain.com;
    
    location / {
        root /path/to/frontend/dist;
        try_files $uri $uri/ /index.html;
    }
    
    location /api {
        proxy_pass http://backend-api:8000;
    }
}
```

#### 3. Environment Configuration

Create `frontend/.env.production` for production-specific config:
```bash
VITE_API_BASE_URL=https://api.yourdomain.com
VITE_APP_TITLE=Stock Prediction API
```

---

## 🐳 Docker Deployment

### Backend Dockerfile

Create `backend/Dockerfile`:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy pyproject.toml
COPY pyproject.toml poetry.lock* /app/

# Install dependencies (production only)
RUN poetry install --only main --no-root

# Copy application
COPY . /app/

# Expose port
EXPOSE 8000

# Run Uvicorn
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Frontend Dockerfile

Create `frontend/Dockerfile`:
```dockerfile
# Build stage
FROM node:24-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

# Production stage
FROM node:24-alpine

WORKDIR /app
RUN npm install -g serve
COPY --from=builder /app/dist ./dist

EXPOSE 5173

CMD ["serve", "-s", "dist", "-l", "5173"]
```

### Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DEBUG=false
      - ALLOWED_ORIGINS=http://frontend:5173
    volumes:
      - ./backend/logs:/app/logs
      - ./backend/ml_models:/app/ml_models
    depends_on:
      - frontend

  frontend:
    build: ./frontend
    ports:
      - "5173:5173"
    environment:
      - VITE_API_BASE_URL=http://localhost:8000

volumes:
  logs:
  ml_models:
```

**Deploy with Docker:**
```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop services
docker-compose down
```

---

## 🧪 Testing

### Backend Tests

```bash
cd backend
poetry install  # Includes dev dependencies
poetry run pytest tests/

# With coverage
poetry run pytest tests/ --cov=api --cov=services
```

### Frontend Tests

Currently, no test files exist. To add testing:

```bash
cd frontend
npm install --save-dev vitest @testing-library/react @testing-library/jest-dom

# Run tests
npm run test
```

---

## 🔍 Monitoring & Debugging

### Health Check Endpoint
```bash
GET /api/monitor
```

Returns system metrics:
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

### View Application Logs
```bash
# API request logs
GET /api/logs/api_requests

# Training logs
GET /api/logs/training

# List all available logs
GET /api/logs
```

### File-Based Logs
```bash
backend/logs/api_requests.log   # HTTP request/response logs
backend/logs/training.log       # LSTM model training logs
```

---

## 🚀 Performance Optimization

### Backend Optimization

1. **Worker Count:** Set based on CPU cores
   ```bash
   # For 4-core system
   uvicorn main:app --workers 4
   ```

2. **Connection Pooling:** Use environment variables for database connections (if applicable)

3. **Cache Predictions:** Implement result caching for frequently requested predictions

4. **Async Operations:** All endpoints are async-ready for better concurrency

### Frontend Optimization

1. **Build Analysis:** Check bundle size
   ```bash
   npm run build -- --analyze
   ```

2. **Lazy Load Components:** Split routes with React.lazy()

3. **Image Optimization:** Use modern image formats (WebP)

4. **Minification:** Vite automatically minifies in production

---

## 🔐 Security Checklist for Production

- [ ] Update CORS origins to specific domains (not `*`)
- [ ] Enable HTTPS/TLS certificates
- [ ] Set `DEBUG=false` environment variable
- [ ] Use environment variables for sensitive data (keys, tokens)
- [ ] Enable request rate limiting
- [ ] Add authentication/authorization (JWT recommended)
- [ ] Regular dependency updates: `poetry update`, `npm update`
- [ ] Run security audits: `poetry audit`, `npm audit`
- [ ] Implement request validation (Pydantic handles this)
- [ ] Use secrets management (AWS Secrets Manager, HashiCorp Vault)

---

## 📝 Common Commands Reference

### Backend
```bash
# Setup
poetry install
poetry shell

# Development
uvicorn main:app --reload

# Production
poetry run uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Testing
poetry run pytest tests/

# Dependency management
poetry add package_name
poetry update
poetry audit
```

### Frontend
```bash
# Setup
npm install

# Development
npm run dev

# Production
npm run build

# Preview build
npm run preview

# Linting
npm run lint

# Dependency management
npm install package_name
npm update
npm audit
```

---

## 🐛 Troubleshooting

### Backend Issues

**Port already in use:**
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
uvicorn main:app --port 8001
```

**Poetry virtual environment not activating:**
```bash
# Refresh poetry environment
poetry env use python3.12

# Remove and recreate
poetry env remove python3.12
poetry install
```

**Module import errors:**
```bash
# Reinstall dependencies
poetry install --no-cache

# Clear Python cache
find . -type d -name __pycache__ -exec rm -r {} +
```

### Frontend Issues

**Port 5173 already in use:**
```bash
# Use different port
npm run dev -- --port 5174
```

**Module not found errors:**
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Build issues:**
```bash
# Clear Vite cache
rm -rf dist .vite
npm run build
```

---

## 📚 Resource Limits & Scaling

### Recommended Specifications

**Development Machine:**
- CPU: 4+ cores
- RAM: 8 GB+
- Storage: 50 GB+
- GPU: Optional (beneficial for PyTorch)

**Production Server:**
- CPU: 8+ cores (or auto-scaling)
- RAM: 16+ GB
- Storage: 100+ GB (for models and logs)
- GPU: Recommended for ML inference
- Bandwidth: 10 Mbps+ (for API traffic)

### Scaling Strategies

1. **Horizontal Scaling:** Use load balancer with multiple backend instances
2. **Model Caching:** Cache predictions to reduce computation
3. **Async Background Tasks:** Use Celery for long-running training jobs
4. **Database:** Add PostgreSQL for persistent data storage
5. **Message Queue:** Use Redis/RabbitMQ for job queuing

---

## 🔄 CI/CD Integration

### GitHub Actions Example

Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - uses: actions/setup-node@v3
        with:
          node-version: '24'
      
      - name: Install and test backend
        run: |
          cd backend
          pip install poetry
          poetry install
          poetry run pytest
      
      - name: Build frontend
        run: |
          cd frontend
          npm install
          npm run build
      
      - name: Deploy to production
        run: echo "Deploy steps here"
```

---

## ✅ Final Verification Checklist

- [ ] Python 3.12+ installed
- [ ] Node 24+ installed
- [ ] Poetry installed
- [ ] Backend dependencies installed and environment activated
- [ ] Frontend dependencies installed
- [ ] Backend server running on 127.0.0.1:8000
- [ ] Frontend server running on localhost:5173
- [ ] API documentation accessible at /docs
- [ ] CORS properly configured for your domain
- [ ] Logs being written to `logs/` directory
- [ ] All environment variables set
- [ ] Security settings reviewed for production

---

## 📞 Support & Resources

- **FastAPI Documentation:** https://fastapi.tiangolo.com/
- **PyTorch Documentation:** https://pytorch.org/docs/
- **React Documentation:** https://react.dev/
- **Vite Documentation:** https://vitejs.dev/
- **Poetry Documentation:** https://python-poetry.org/docs/

---

**Last Updated:** December 15, 2025  
**Version:** 1.0.0
