"""
Pydantic schemas for Nigerian user profiles and persona modeling.
"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class NigerianPersonaType(str, Enum):
    """
    Archetypal Nigerian shopper profiles used for cold-start modeling
    and review simulation.
    """
    BUDGET_STUDENT = "budget_student"           # NYSC, university student, tight budget
    LAGOS_PROFESSIONAL = "lagos_professional"   # Working class, brand conscious
    NIGERIAN_MUM = "nigerian_mum"               # Family-focused, value-driven
    TECH_ENTHUSIAST = "tech_enthusiast"         # Specs-driven, early adopter
    ONLINE_SKEPTIC = "online_skeptic"           # Wary of scams, low trust
    MARKET_TRADER = "market_trader"             # Bulk buyer, price negotiator
    UNKNOWN = "unknown"                          # Not yet classified


class NigerianLocation(str, Enum):
    LAGOS = "Lagos"
    ABUJA = "Abuja"
    PORT_HARCOURT = "Port Harcourt"
    KANO = "Kano"
    IBADAN = "Ibadan"
    ENUGU = "Enugu"
    BENIN = "Benin City"
    OWERRI = "Owerri"
    UNKNOWN = "Unknown"


class UserProfile(BaseModel):
    """Persistent user profile stored in PostgreSQL."""
    user_id: str
    session_id: str
    budget_min: Optional[int] = None           # in Naira
    budget_max: Optional[int] = None           # in Naira
    preferred_categories: list[str] = Field(default_factory=list)
    location: NigerianLocation = NigerianLocation.UNKNOWN
    persona_type: NigerianPersonaType = NigerianPersonaType.UNKNOWN
    price_sensitivity: float = Field(default=0.5, ge=0.0, le=1.0)
    brand_affinity: dict[str, float] = Field(default_factory=dict)
    shopping_intent: str = "browsing"          # browsing | comparing | ready_to_buy
    cold_start: bool = True
    interaction_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class UserPersona(BaseModel):
    """
    Enriched persona used by agents during inference.
    Combines profile data with inferred behavioral priors.
    """
    profile: UserProfile
    inferred_budget_range: tuple[int, int] = (10_000, 500_000)
    category_affinity_scores: dict[str, float] = Field(default_factory=dict)
    price_sensitivity_label: str = "moderate"  # low | moderate | high
    urgency_level: str = "low"                  # low | medium | high
    trust_threshold: float = 0.6               # min trust score to recommend
    preferred_brand_tier: str = "mid_range"    # budget | mid_range | premium
    nigerian_signals: list[str] = Field(default_factory=list)  # ["nysc", "january_blues"]
    conversation_history: list[dict] = Field(default_factory=list)
