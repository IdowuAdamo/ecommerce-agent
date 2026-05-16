"""
Schemas for Task A: Nigerian-style review simulation.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from .user import NigerianPersonaType, NigerianLocation


class ReviewRequest(BaseModel):
    product_id: str
    product_name: str
    product_category: str
    product_description: str = ""
    actual_price: int
    predicted_fair_price: Optional[int] = None
    rating: Optional[float] = None             # if None, model predicts it
    persona_type: Optional[NigerianPersonaType] = None  # if None, randomly assigned
    location: Optional[NigerianLocation] = None
    num_reviews: int = Field(default=1, ge=1, le=10)


class SimulatedReview(BaseModel):
    """A single Nigerian-style simulated product review (Task A output)."""
    product_id: str
    persona_type: NigerianPersonaType
    location: NigerianLocation
    star_rating: float = Field(ge=1.0, le=5.0)
    review_text: str
    sentiment: str = "neutral"                 # positive | neutral | negative
    uses_nigerian_slang: bool = True
    key_themes: list[str] = Field(default_factory=list)  # ["value", "delivery", "battery"]

    # Evaluation metrics (filled post-generation)
    rouge1: Optional[float] = None
    rouge2: Optional[float] = None
    rougeL: Optional[float] = None
    bert_score_f1: Optional[float] = None
    behavioral_fidelity: Optional[float] = None  # 0-1 how well it matches persona
