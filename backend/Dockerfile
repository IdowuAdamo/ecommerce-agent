# ── Stage 1: Builder — install heavy deps ────────────────────────────────────
FROM python:3.11-slim AS builder

# System build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy and install Python deps into an isolated prefix
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime — lean image ────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime system deps only (curl for health check)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application code
COPY app/ ./app/

# Non-root user for security
RUN useradd -m -u 1000 naijashop && chown -R naijashop:naijashop /app
USER naijashop

# Render injects $PORT — uvicorn must bind to it
ENV PORT=8000
EXPOSE $PORT

# Health check (start-period covers ML model loading ~60s)
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Render sets $PORT automatically; we pass it through
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1
