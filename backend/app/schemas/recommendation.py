"""
Schemas for recommendation requests, responses, and explanation cards.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from .product import RankedProduct
from .user import UserPersona


class RecommendationRequest(BaseModel):
    session_id: str
    query: str
    budget_max: Optional[int] = None           # override profile budget
    category_hint: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=50)
    explain: bool = True


class ExplanationCard(BaseModel):
    """Human-readable explanation for a recommendation."""
    product_id: str
    headline: str
    reasons: list[str]                         # bullet points
    warnings: list[str] = Field(default_factory=list)  # trust flags
    value_verdict: str = ""                    # "Great deal", "Fair price", "Overpriced"
    nigerian_context: str = ""                 # localized insight


class RecommendationResponse(BaseModel):
    session_id: str
    query: str
    ranked_products: list[RankedProduct]
    explanations: list[ExplanationCard]
    user_persona_used: Optional[str] = None
    agent_reasoning: Optional[str] = None      # chain-of-thought trace
    evaluation: Optional[dict] = None          # NDCG@10, HitRate if computed
    latency_ms: Optional[int] = None
