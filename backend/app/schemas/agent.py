"""
Schemas for LangGraph agent state and WebSocket chat messages.
"""
from __future__ import annotations
from typing import Optional, Any, Annotated
from pydantic import BaseModel, Field
from .user import UserPersona
from .product import Product, RankedProduct
from .recommendation import ExplanationCard


class ChatMessage(BaseModel):
    role: str                                  # "user" | "assistant" | "system"
    content: str
    metadata: dict = Field(default_factory=dict)


class ChatRequest(BaseModel):
    session_id: str
    message: str
    stream: bool = True


class ChatResponse(BaseModel):
    session_id: str
    message: str
    recommendations: list[RankedProduct] = Field(default_factory=list)
    explanations: list[ExplanationCard] = Field(default_factory=list)
    agent_steps: list[str] = Field(default_factory=list)  # agent reasoning trace
    is_final: bool = True


class AgentState(BaseModel):
    """
    Shared state object passed through the LangGraph DAG.
    Each agent reads and writes to this state.
    """
    # Input
    session_id: str
    user_query: str
    conversation_history: list[ChatMessage] = Field(default_factory=list)

    # Discovery Agent output
    extracted_intent: Optional[dict] = None
    budget_min: Optional[int] = None
    budget_max: Optional[int] = None
    category: Optional[str] = None          # raw category from discovery
    resolved_category: Optional[str] = None  # canonical Jumia category key
    use_case: Optional[str] = None
    business_context: Optional[str] = None   # e.g. "commercial", "personal"
    location_hint: Optional[str] = None
    urgency: str = "low"
    clarification_needed: bool = False
    clarification_question: Optional[str] = None

    # User Modeling Agent output
    user_persona: Optional[UserPersona] = None

    # Commerce Intelligence Agent output
    raw_products: list[Product] = Field(default_factory=list)
    retrieval_source: str = "mock"             # "jumia" | "konga" | "mock"

    # Trust & Value Agent output
    scored_products: list[Product] = Field(default_factory=list)

    # Recommendation Agent output
    ranked_products: list[RankedProduct] = Field(default_factory=list)

    # Explanation Agent output
    explanations: list[ExplanationCard] = Field(default_factory=list)
    final_response: Optional[str] = None

    # Meta
    agent_trace: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    latency_ms: Optional[int] = None
