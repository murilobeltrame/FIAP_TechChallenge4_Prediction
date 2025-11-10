# Multi-stage build for production-ready container
FROM python:3.14-slim AS base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update \
    && apt upgrade -y \
    && apt install -y ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

FROM base AS builder

# Install build dependencies
RUN apt update \
    && apt install -y build-essential curl gcc pipx \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

# Install Poetry
RUN pip install poetry

# Create virtual environment
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV POETRY_NO_INTERACTION=1

# Copy dependency files
COPY pyproject.toml .

# Install Python dependencies
RUN poetry install --no-root

# # Production stage
# FROM base AS production

# # Set environment variables
# ENV PYTHONUNBUFFERED=1 \
#     PYTHONDONTWRITEBYTECODE=1 \
#     PATH="/opt/venv/bin:$PATH" \
#     PORT=8000

# # Install runtime dependencies and security updates
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     ca-certificates \
#     && apt-get upgrade -y \
#     && rm -rf /var/lib/apt/lists/* \
#     && apt-get clean

# # Create non-root user for security
# RUN groupadd -r appuser && useradd -r -g appuser appuser

# # Copy virtual environment from builder
# COPY --from=builder /opt/venv /opt/venv

# # Set working directory
# WORKDIR /app

# # Copy application code with proper ownership
# COPY --chown=appuser:appuser . .

# # Ensure ml_models directory exists and has proper permissions
# RUN mkdir -p /app/ml_models && \
#     chown -R appuser:appuser /app && \
#     chmod -R 755 /app

# # Switch to non-root user
# USER appuser

# # Create directory for logs with proper permissions
# RUN mkdir -p /app/logs

# # Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
#     CMD python -c "import requests; requests.get('http://localhost:${PORT}/docs', timeout=5)" || exit 1

# # Expose port
# EXPOSE ${PORT}

# # Use Gunicorn for production with optimal settings
# CMD ["gunicorn", \
#      "--bind", "0.0.0.0:8000", \
#      "--workers", "4", \
#      "--worker-class", "uvicorn.workers.UvicornWorker", \
#      "--worker-connections", "1000", \
#      "--max-requests", "1000", \
#      "--max-requests-jitter", "50", \
#      "--timeout", "120", \
#      "--keep-alive", "5", \
#      "--log-level", "info", \
#      "--access-logfile", "-", \
#      "--error-logfile", "-", \
#      "main:app"]