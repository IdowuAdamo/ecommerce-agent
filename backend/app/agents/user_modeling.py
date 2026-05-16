"""
Agent 1: User Modeling Agent.

Builds and maintains Nigerian shopper profiles.
Handles cold-start via persona archetypes, and enriches returning user profiles
from interaction history.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.schemas.agent import AgentState
from app.schemas.user import (
    UserProfile, UserPersona, NigerianPersonaType, NigerianLocation
)
from app.services.user_store import UserStoreService

logger = logging.getLogger(__name__)

# Cold-start priors per persona archetype
PERSONA_PRIORS: dict[NigerianPersonaType, dict] = {
    NigerianPersonaType.BUDGET_STUDENT: {
        "budget_range": (15_000, 150_000),
        "price_sensitivity": 0.9,
        "brand_tier": "budget",
        "trust_threshold": 0.5,
        "categories": ["phones", "laptops", "accessories"],
        "price_sensitivity_label": "high",
    },
    NigerianPersonaType.LAGOS_PROFESSIONAL: {
        "budget_range": (100_000, 1_000_000),
        "price_sensitivity": 0.4,
        "brand_tier": "mid_range",
        "trust_threshold": 0.7,
        "categories": ["laptops", "phones", "fashion_men", "fashion_women"],
        "price_sensitivity_label": "low",
    },
    NigerianPersonaType.NIGERIAN_MUM: {
        "budget_range": (20_000, 300_000),
        "price_sensitivity": 0.7,
        "brand_tier": "mid_range",
        "trust_threshold": 0.65,
        "categories": ["appliances", "baby", "health", "furniture"],
        "price_sensitivity_label": "moderate",
    },
    NigerianPersonaType.TECH_ENTHUSIAST: {
        "budget_range": (80_000, 2_000_000),
        "price_sensitivity": 0.3,
        "brand_tier": "premium",
        "trust_threshold": 0.75,
        "categories": ["laptops", "phones", "cameras", "gaming", "computing"],
        "price_sensitivity_label": "low",
    },
    NigerianPersonaType.ONLINE_SKEPTIC: {
        "budget_range": (10_000, 200_000),
        "price_sensitivity": 0.8,
        "brand_tier": "budget",
        "trust_threshold": 0.85,  # Very high trust threshold
        "categories": ["phones", "appliances"],
        "price_sensitivity_label": "high",
    },
    NigerianPersonaType.MARKET_TRADER: {
        "budget_range": (50_000, 500_000),
        "price_sensitivity": 0.85,
        "brand_tier": "budget",
        "trust_threshold": 0.6,
        "categories": ["appliances", "fashion_men", "fashion_women", "phones"],
        "price_sensitivity_label": "high",
    },
    NigerianPersonaType.UNKNOWN: {
        "budget_range": (20_000, 500_000),
        "price_sensitivity": 0.5,
        "brand_tier": "mid_range",
        "trust_threshold": 0.65,
        "categories": [],
        "price_sensitivity_label": "moderate",
    },
}

# Nigerian signal → persona mapping
SIGNAL_TO_PERSONA: dict[str, NigerianPersonaType] = {
    "nysc": NigerianPersonaType.BUDGET_STUDENT,
    "student": NigerianPersonaType.BUDGET_STUDENT,
    "university": NigerianPersonaType.BUDGET_STUDENT,
    "pocket money": NigerianPersonaType.BUDGET_STUDENT,
    "january blues": NigerianPersonaType.ONLINE_SKEPTIC,
    "lagos": NigerianPersonaType.LAGOS_PROFESSIONAL,
    "abuja": NigerianPersonaType.LAGOS_PROFESSIONAL,
    "professional": NigerianPersonaType.LAGOS_PROFESSIONAL,
    "office": NigerianPersonaType.LAGOS_PROFESSIONAL,
    "mum": NigerianPersonaType.NIGERIAN_MUM,
    "mom": NigerianPersonaType.NIGERIAN_MUM,
    "family": NigerianPersonaType.NIGERIAN_MUM,
    "cooking": NigerianPersonaType.NIGERIAN_MUM,
    "specs": NigerianPersonaType.TECH_ENTHUSIAST,
    "gaming": NigerianPersonaType.TECH_ENTHUSIAST,
    "developer": NigerianPersonaType.TECH_ENTHUSIAST,
    "ml": NigerianPersonaType.TECH_ENTHUSIAST,
    "machine learning": NigerianPersonaType.TECH_ENTHUSIAST,
    "i dont trust": NigerianPersonaType.ONLINE_SKEPTIC,
    "fake": NigerianPersonaType.ONLINE_SKEPTIC,
    "scam": NigerianPersonaType.ONLINE_SKEPTIC,
    "bulk": NigerianPersonaType.MARKET_TRADER,
    "resell": NigerianPersonaType.MARKET_TRADER,
    "shop": NigerianPersonaType.MARKET_TRADER,
}


class UserModelingAgent:
    """
    Builds and enriches user personas.
    Cold-start: assigns Nigerian archetype from signals.
    Returning: loads from DB + updates behavioral priors.
    """

    def __init__(self):
        self.store = UserStoreService()

    async def run(self, state: AgentState) -> AgentState:
        state.agent_trace.append("👤 User Modeling Agent: Building user persona...")
        logger.info(f"User Modeling Agent for session: {state.session_id}")

        try:
            profile = await self.store.get_or_create_profile(state.session_id)
            persona = self._build_persona(profile, state)
            state.user_persona = persona

            # Persist any updates from this query
            await self._update_profile_from_state(profile, state)

            ptype_str = persona.profile.persona_type.value if hasattr(persona.profile.persona_type, "value") else str(persona.profile.persona_type)
            trace = (
                f"  → Persona: {ptype_str}, "
                f"Cold-start: {profile.cold_start}, "
                f"Budget: ₦{persona.inferred_budget_range[0]:,}–₦{persona.inferred_budget_range[1]:,}"
            )
            state.agent_trace.append(trace)
            logger.info(trace)

        except Exception as e:
            logger.error(f"User Modeling Agent failed: {e}")
            state.errors.append(f"UserModeling: {e}")
            # Default persona as fallback
            state.user_persona = _default_persona(state.session_id)

        return state

    def _build_persona(self, profile: UserProfile, state: AgentState) -> UserPersona:
        """Build enriched persona from profile + current query signals."""
        intent = state.extracted_intent or {}
        nigerian_signals = intent.get("nigerian_signals", [])

        # 1. Infer/update persona type
        persona_type = profile.persona_type
        if persona_type == NigerianPersonaType.UNKNOWN or profile.cold_start:
            persona_type = self._infer_persona(
                state.user_query, nigerian_signals, intent
            )
            profile.persona_type = persona_type

        priors = PERSONA_PRIORS.get(persona_type, PERSONA_PRIORS[NigerianPersonaType.UNKNOWN])

        # 2. Infer budget range
        budget_range = priors["budget_range"]
        if state.budget_min is not None and state.budget_max is not None:
            budget_range = (state.budget_min, state.budget_max)
        elif state.budget_max is not None:
            budget_range = (0, state.budget_max)
        elif profile.budget_max:
            budget_range = (profile.budget_min or 0, profile.budget_max)

        # 3. Infer location
        location = profile.location
        if state.location_hint:
            location = _parse_location(state.location_hint)

        # 4. Category affinity scores
        affinity: dict[str, float] = {}
        for cat in (profile.preferred_categories or []):
            affinity[cat] = affinity.get(cat, 0) + 0.3
        if state.category:
            affinity[state.category] = affinity.get(state.category, 0) + 0.5
        for cat in priors["categories"]:
            affinity[cat] = affinity.get(cat, 0) + 0.1

        profile.location = location

        return UserPersona(
            profile=profile,
            inferred_budget_range=budget_range,
            category_affinity_scores=affinity,
            price_sensitivity_label=priors["price_sensitivity_label"],
            urgency_level=state.urgency,
            trust_threshold=priors["trust_threshold"],
            preferred_brand_tier=priors["brand_tier"],
            nigerian_signals=nigerian_signals,
            conversation_history=[
                {"role": m.role, "content": m.content}
                for m in state.conversation_history[-10:]
            ],
        )

    def _infer_persona(
        self, query: str, signals: list[str], intent: dict
    ) -> NigerianPersonaType:
        """Infer Nigerian persona archetype from query signals."""
        query_lower = query.lower()
        for signal in signals:
            persona = SIGNAL_TO_PERSONA.get(signal.lower())
            if persona:
                return persona
        for keyword, persona in SIGNAL_TO_PERSONA.items():
            if keyword in query_lower:
                return persona
        return NigerianPersonaType.UNKNOWN

    async def _update_profile_from_state(self, profile: UserProfile, state: AgentState) -> None:
        updates: dict = {
            "persona_type": profile.persona_type.value if hasattr(profile.persona_type, "value") else profile.persona_type,
            "cold_start": False,
            "interaction_count": profile.interaction_count + 1,
        }
        if state.budget_max:
            updates["budget_max"] = state.budget_max
        if state.budget_min:
            updates["budget_min"] = state.budget_min
        if state.category and state.category not in (profile.preferred_categories or []):
            cats = list(profile.preferred_categories or []) + [state.category]
            updates["preferred_categories"] = cats[:10]  # cap at 10
        if profile.location != state.user_persona.profile.location:
            updates["location"] = profile.location.value if hasattr(profile.location, "value") else profile.location

        await self.store.update_profile(state.session_id, updates)


def _parse_location(hint: str) -> NigerianLocation:
    hint = hint.lower()
    for loc in NigerianLocation:
        if loc.value.lower() in hint:
            return loc
    return NigerianLocation.UNKNOWN


def _default_persona(session_id: str) -> UserPersona:
    profile = UserProfile(
        user_id=session_id,
        session_id=session_id,
        persona_type=NigerianPersonaType.UNKNOWN,
    )
    return UserPersona(
        profile=profile,
        inferred_budget_range=(20_000, 500_000),
        trust_threshold=0.65,
    )
