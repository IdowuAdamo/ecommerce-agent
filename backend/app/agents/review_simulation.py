"""
Agent 6: Review Simulation Agent (Task A).

Generates realistic Nigerian-style product reviews using GPT-4o.
Simulates multiple Nigerian consumer archetypes with:
  - Nigerian Pidgin and Yoruba/Igbo/Hausa code-switching
  - Culture-specific concerns (NEPA/power outages, delivery, resale value)
  - Ratings calibrated by DeBERTa price fairness model
  - ROUGE and BERTScore optimized prompting

Optimizes for Task A metrics: ROUGE-1/2/L, BERTScore, RMSE, behavioral fidelity.
"""
from __future__ import annotations

import json
import logging
import random
from typing import Optional

from openai import AsyncOpenAI

from app.config import get_settings
from app.schemas.agent import AgentState
from app.schemas.review import ReviewRequest, SimulatedReview
from app.schemas.user import NigerianPersonaType, NigerianLocation

logger = logging.getLogger(__name__)

# ── Nigerian persona prompt templates ────────────────────────────────────────
PERSONA_CONTEXTS: dict[NigerianPersonaType, dict] = {
    NigerianPersonaType.BUDGET_STUDENT: {
        "description": "university student or NYSC corps member on a tight budget",
        "concerns": ["price", "durability", "battery life", "value for money"],
        "tone": "casual, enthusiastic but budget-conscious",
        "slang_examples": ["e do the work", "sha", "abeg", "e no too bad", "e hard sha"],
        "location_hints": ["hostel", "campus", "NYSC camp", "school"],
    },
    NigerianPersonaType.LAGOS_PROFESSIONAL: {
        "description": "Lagos-based working professional or entrepreneur",
        "concerns": ["quality", "brand reputation", "warranty", "after-sales service"],
        "tone": "confident, articulate, brand-aware",
        "slang_examples": ["I must commend", "for real though", "Lagos no dey carry last"],
        "location_hints": ["VI", "Lekki", "Ikeja", "Island", "Mainland"],
    },
    NigerianPersonaType.NIGERIAN_MUM: {
        "description": "Nigerian mother/parent shopping for family needs",
        "concerns": ["safety", "durability", "ease of use", "family value"],
        "tone": "practical, detailed, motherly",
        "slang_examples": ["e dey work well", "my family love am", "e last long", "quality no lie"],
        "location_hints": ["market", "home", "for my family"],
    },
    NigerianPersonaType.TECH_ENTHUSIAST: {
        "description": "Nigerian tech-savvy person, developer or gaming enthusiast",
        "concerns": ["specs", "performance", "GPU/CPU", "RAM", "display quality"],
        "tone": "technical, detailed, comparative",
        "slang_examples": ["the specs are insane", "benchmark is mad", "no cap", "W product"],
        "location_hints": ["for my workstation", "gaming setup", "dev environment"],
    },
    NigerianPersonaType.ONLINE_SKEPTIC: {
        "description": "Nigerian who was scammed before and is now very careful online",
        "concerns": ["authenticity", "seller reputation", "return policy", "packaging"],
        "tone": "suspicious initially, relieved if good, outraged if bad",
        "slang_examples": ["I was scared at first", "dem no cheat me", "I verify am well well"],
        "location_hints": ["was afraid to order", "checked reviews first"],
    },
    NigerianPersonaType.MARKET_TRADER: {
        "description": "Nigerian market trader or small business owner",
        "concerns": ["resale value", "durability", "price negotiation", "bulk pricing"],
        "tone": "pragmatic, commerce-focused, street-smart",
        "slang_examples": ["e go sell well", "customer go like am", "the markup good"],
        "location_hints": ["Alaba", "Computer Village", "Tejuosho", "my shop"],
    },
}

REVIEW_SYSTEM_PROMPT = """You are generating a realistic Nigerian product review for an e-commerce platform.

Write EXACTLY ONE review in the voice of the specified Nigerian consumer persona.
The review should:
1. Sound completely authentic — like a real Nigerian wrote it
2. Mix Nigerian Pidgin with standard English naturally (not forced)
3. Mention specific, concrete details about the product
4. Reference Nigerian-specific concerns: power cuts (NEPA), delivery (Jumia Express), resale value, etc.
5. Match the star rating emotionally
6. Be 60-150 words long
7. NOT use generic phrases like "This product is good" alone

Return JSON with: {"star_rating": float, "review_text": string, "sentiment": "positive"|"neutral"|"negative"}"""


class ReviewSimulationAgent:
    """Generates Task A: Nigerian-style product review simulation."""

    def __init__(self):
        s = get_settings()
        self.client = AsyncOpenAI(api_key=s.openai_api_key)
        self.model = "gpt-4o"  # Use full GPT-4o for quality reviews (Task A quality matters)

    async def simulate_reviews(
        self, request: ReviewRequest
    ) -> list[SimulatedReview]:
        """Generate num_reviews simulated reviews for a product."""
        reviews: list[SimulatedReview] = []

        personas_to_use = (
            [request.persona_type] * request.num_reviews
            if request.persona_type
            else random.choices(list(NigerianPersonaType)[:-1], k=request.num_reviews)
        )

        for persona_type in personas_to_use:
            review = await self._generate_review(request, persona_type)
            if review:
                reviews.append(review)

        return reviews

    async def _generate_review(
        self, request: ReviewRequest, persona_type: NigerianPersonaType
    ) -> Optional[SimulatedReview]:
        persona = PERSONA_CONTEXTS.get(persona_type, PERSONA_CONTEXTS[NigerianPersonaType.UNKNOWN if NigerianPersonaType.UNKNOWN in PERSONA_CONTEXTS else NigerianPersonaType.BUDGET_STUDENT])
        location = request.location or random.choice(list(NigerianLocation)[:-1])

        # Calibrate suggested rating using price fairness
        suggested_rating = request.rating
        if suggested_rating is None:
            if request.predicted_fair_price:
                deviation = (request.actual_price - request.predicted_fair_price) / request.predicted_fair_price
                base_rating = 4.2 - deviation * 1.5  # Overpriced → lower rating tendency
                suggested_rating = max(1.5, min(5.0, base_rating))
                # Add persona-specific bias
                if persona_type == NigerianPersonaType.ONLINE_SKEPTIC:
                    suggested_rating -= 0.3
                elif persona_type == NigerianPersonaType.TECH_ENTHUSIAST:
                    suggested_rating += 0.2
                suggested_rating = round(max(1.0, min(5.0, suggested_rating)) * 2) / 2
            else:
                suggested_rating = random.choice([3.5, 4.0, 4.0, 4.5, 4.5, 5.0])

        user_prompt = f"""Product: {request.product_name}
Category: {request.product_category}
Price: ₦{request.actual_price:,}
Description: {request.product_description[:200] or "No description available"}

Reviewer persona: {persona["description"]}
Location: {location.value if hasattr(location, 'value') else location}
Suggested star rating: {suggested_rating}/5 stars
Reviewer concerns: {", ".join(persona["concerns"][:3])}
Tone: {persona["tone"]}
Sample slang to naturally incorporate: {", ".join(random.sample(persona["slang_examples"], min(2, len(persona["slang_examples"]))))}

Write an authentic Nigerian review for this product."""

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.85,
                max_tokens=300,
            )
            data = json.loads(resp.choices[0].message.content)
            star_rating = float(data.get("star_rating", suggested_rating))
            review_text = data.get("review_text", "")
            sentiment = data.get("sentiment", "neutral")

            if not review_text or len(review_text) < 20:
                return None

            return SimulatedReview(
                product_id=request.product_id,
                persona_type=persona_type,
                location=location if isinstance(location, NigerianLocation) else NigerianLocation.UNKNOWN,
                star_rating=round(star_rating * 2) / 2,  # round to nearest 0.5
                review_text=review_text,
                sentiment=sentiment,
                uses_nigerian_slang=True,
                key_themes=persona["concerns"][:3],
            )

        except Exception as e:
            logger.error(f"Review generation failed for {persona_type}: {e}")
            return None

    async def run(self, state: AgentState) -> AgentState:
        """Generate reviews for top-ranked products in the agent pipeline."""
        state.agent_trace.append("✍️ Review Simulation Agent: Generating Nigerian reviews...")
        # In the main pipeline, review simulation is done for top-3 products
        top_products = (state.ranked_products or [])[:3]

        for ranked in top_products:
            p = ranked.product
            req = ReviewRequest(
                product_id=p.id,
                product_name=p.name,
                product_category=p.category,
                product_description=p.description,
                actual_price=p.price,
                predicted_fair_price=p.price_fairness.predicted_fair_price if p.price_fairness else None,
                num_reviews=1,
            )
            reviews = await self.simulate_reviews(req)
            if reviews:
                state.agent_trace.append(
                    f"  → Review for '{p.name[:30]}...': {reviews[0].star_rating}⭐"
                )

        return state
