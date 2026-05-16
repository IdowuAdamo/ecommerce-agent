"""
NaijaShop AI — FastAPI Application Entry Point.

Provides:
  - /api/v1/chat (POST + WebSocket streaming)
  - /api/v1/recommend
  - /api/v1/reviews/simulate
  - /api/v1/evaluate/task-a and task-b
  - /health and /metrics
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.agents.orchestrator import AgentOrchestrator
from app.agents.review_simulation import ReviewSimulationAgent
from app.evaluation.evaluator import EvaluationRunner
from app.models.price_predictor import PricePredictorModel
from app.models.embeddings import EmbeddingModel
from app.schemas.agent import ChatRequest, ChatResponse, ChatMessage
from app.schemas.review import ReviewRequest
from app.schemas.recommendation import RecommendationRequest

logger = structlog.get_logger(__name__)

# ── Lifespan: warm up models on startup ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    logger.info("[STARTUP] NaijaShop AI starting up...")
    # Pre-load ML models (blocking, done once)
    try:
        EmbeddingModel.get_instance()
        logger.info("[OK] Embedding model ready")
    except Exception as e:
        logger.warning(f"Embedding model warmup failed: {e}")

    try:
        PricePredictorModel.get_instance()
        logger.info("[OK] Price predictor ready")
    except Exception as e:
        logger.warning(f"Price predictor warmup failed (will load on first request): {e}")

    app.state.orchestrator = AgentOrchestrator()
    app.state.review_agent = ReviewSimulationAgent()
    app.state.evaluator = EvaluationRunner()
    logger.info("[READY] NaijaShop AI is ready!")
    yield
    logger.info("NaijaShop AI shutting down...")


# ── App Setup ─────────────────────────────────────────────────────────────────
s = get_settings()
app = FastAPI(
    title=s.app_name,
    version=s.app_version,
    description="Multi-agent AI commerce intelligence platform for Nigerian e-commerce",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=s.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session message store (use Redis in production)
_session_history: dict[str, list[ChatMessage]] = {}


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "app": s.app_name, "version": s.app_version}


@app.get("/", tags=["System"])
async def root():
    return {"message": "Welcome to NaijaShop AI 🇳🇬", "docs": "/docs"}


# ── Chat (HTTP) ───────────────────────────────────────────────────────────────
@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Main conversational endpoint.
    Runs the full 7-agent pipeline and returns recommendations + explanations.
    """
    session_id = request.session_id
    history = _session_history.get(session_id, [])

    orchestrator: AgentOrchestrator = app.state.orchestrator
    result = await orchestrator.run(session_id, request.message, history)

    # Update session history
    history.append(ChatMessage(role="user", content=request.message))
    if result.final_response:
        history.append(ChatMessage(role="assistant", content=result.final_response))
    _session_history[session_id] = history[-20:]  # Keep last 20 turns

    return ChatResponse(
        session_id=session_id,
        message=result.final_response or "I'm working on finding the best options for you!",
        recommendations=result.ranked_products,
        explanations=result.explanations,
        agent_steps=result.agent_trace,
        is_final=True,
    )


# ── WebSocket Chat (Streaming) ────────────────────────────────────────────────
@app.websocket("/ws/chat/{session_id}")
async def ws_chat(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for streaming agent steps in real-time."""
    await websocket.accept()
    orchestrator: AgentOrchestrator = app.state.orchestrator
    history = _session_history.get(session_id, [])

    try:
        while True:
            data = await websocket.receive_json()
            query = data.get("message", "")
            if not query:
                continue

            # Stream: send agent trace steps as they complete
            await websocket.send_json({"type": "start", "session_id": session_id})

            result = await orchestrator.run(session_id, query, history)

            # Stream agent trace
            for step in result.agent_trace:
                await websocket.send_json({"type": "agent_step", "content": step})

            # Stream final response
            history.append(ChatMessage(role="user", content=query))
            if result.final_response:
                history.append(ChatMessage(role="assistant", content=result.final_response))
            _session_history[session_id] = history[-20:]

            await websocket.send_json({
                "type": "complete",
                "message": result.final_response,
                "recommendations": [r.model_dump() for r in result.ranked_products],
                "explanations": [e.model_dump() for e in result.explanations],
            })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")


# ── Review Simulation (Task A) ────────────────────────────────────────────────
@app.post("/api/v1/reviews/simulate", tags=["Task A"])
async def simulate_reviews(request: ReviewRequest):
    """Generate Nigerian-style product reviews (Task A)."""
    agent: ReviewSimulationAgent = app.state.review_agent
    reviews = await agent.simulate_reviews(request)
    return {
        "product_id": request.product_id,
        "reviews": [r.model_dump() for r in reviews],
        "count": len(reviews),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────
@app.post("/api/v1/evaluate/task-a", tags=["Evaluation"])
async def evaluate_task_a(payload: dict):
    """
    Run Task A evaluation: ROUGE + BERTScore + Price RMSE.

    Expected payload:
    {
      "generated_reviews": ["...", "..."],
      "reference_reviews": ["...", "..."],
      "actual_prices": [125000, ...],     // optional
      "predicted_prices": [130000, ...]   // optional
    }
    """
    evaluator: EvaluationRunner = app.state.evaluator
    results = evaluator.evaluate_task_a(
        generated_reviews=payload.get("generated_reviews", []),
        reference_reviews=payload.get("reference_reviews", []),
        actual_prices=payload.get("actual_prices"),
        predicted_prices=payload.get("predicted_prices"),
    )
    return {"task": "A", "metrics": results}


@app.post("/api/v1/evaluate/task-b", tags=["Evaluation"])
async def evaluate_task_b(payload: dict):
    """
    Run Task B evaluation: NDCG@10 + HitRate + MRR.

    Expected payload:
    {
      "relevance_scores": [[1, 0, 1, ...], ...],
      "predicted_scores": [[0.9, 0.3, ...], ...],
      "relevant_items": [["id1", "id2"], ...],
      "recommended_items": [["id2", "id5", ...], ...],
      "k": 10
    }
    """
    evaluator: EvaluationRunner = app.state.evaluator
    results = evaluator.evaluate_task_b(
        relevance_scores=payload.get("relevance_scores", []),
        predicted_scores=payload.get("predicted_scores", []),
        relevant_items=payload.get("relevant_items", []),
        recommended_items=payload.get("recommended_items", []),
        k=payload.get("k", 10),
    )
    return {"task": "B", "metrics": results}


# ── Price Prediction ──────────────────────────────────────────────────────────
@app.post("/api/v1/predict-price", tags=["Price Model"])
async def predict_price(payload: dict):
    """
    Directly call the DeBERTa price predictor.

    Payload: {"product_text": "category: laptops | name: HP 255 G9 | ..."}
    """
    text = payload.get("product_text", "")
    if not text:
        raise HTTPException(400, "product_text is required")

    predictor = PricePredictorModel.get_instance()
    price = predictor.predict(text)
    return {
        "predicted_price_naira": int(price),
        "formatted": f"₦{price:,.0f}",
        "model": get_settings().price_model_id,
    }
