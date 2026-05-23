# NaijaShop AI Backend — Dockerfile (v2.0.0, API-only)
#
# Significantly lighter than v1: no PyTorch, no local ML models.
# All ML inference delegated to OpenAI API.
# Image size: ~250 MB (was ~3 GB with PyTorch).
# Cold-start time: ~5 seconds (was 60-120 seconds loading local models).

FROM python:3.11-slim

# curl is needed for the HEALTHCHECK; no compiler tools required (no C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (no torch = fast, ~2 min vs 10+ min before)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Non-root user for security
RUN useradd -m -u 1000 naijashop && chown -R naijashop:naijashop /app
USER naijashop

# Render and Railway inject $PORT; fallback to 8000 for local dev
ENV PORT=8000
EXPOSE $PORT

# Health check — 30s start period (was 120s; instant startup now)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1
