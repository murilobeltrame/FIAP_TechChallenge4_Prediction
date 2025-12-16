# Build stage
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system dependencies and Poetry
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy dependency files
COPY pyproject.toml poetry.lock* /app/

# Install Python dependencies (production only)
RUN poetry install --only main --no-root --no-directory

# Production stage
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /root/.cache/pypoetry/virtualenvs /root/.cache/pypoetry/virtualenvs

# Copy application code
COPY . /app/

# Install Poetry for runtime (lightweight)
RUN pip install --no-cache-dir poetry

# Set backend root environment variable
ENV BACKEND_ROOT=/app
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/monitor')" || exit 1

# Run Uvicorn with workers
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
