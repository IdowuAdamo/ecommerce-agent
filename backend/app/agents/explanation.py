"""
Agent 7: Explanation Agent.

Generates transparent, human-readable recommendation explanations.
Produces "reasoning cards" that justify why each product was recommended.
Localizes explanations for Nigerian shopping context.
"""
from __future__ import annotations

import logging
from openai import AsyncOpenAI

from app.config import get_settings
from app.schemas.agent import AgentState
from app.schemas.product import RankedProduct
from app.schemas.recommendation import ExplanationCard

logger = logging.getLogger(__name__)

EXPLANATION_SYSTEM = """You are a friendly Nigerian shopping assistant explaining product recommendations.

For each product, generate a short explanation card (JSON) with:
{
  "headline": "one-line value proposition (max 10 words)",
  "reasons": ["3-4 specific bullet points why this product fits"],
  "warnings": ["any trust flags or concerns, empty if none"],
  "value_verdict": "Great Deal | Fair Price | Slightly Overpriced | Overpriced",
  "nigerian_context": "one sentence with Nigerian-specific insight (delivery, power usage, etc.)"
}

Keep language clear, honest, and conversational. Mention ₦ amounts when relevant."""


class ExplanationAgent:
    """Generates transparent explanation cards for recommendations."""

    def __init__(self):
        s = get_settings()
        self.client = AsyncOpenAI(api_key=s.openai_api_key)
        self.model = s.openai_model

    async def run(self, state: AgentState) -> AgentState:
        state.agent_trace.append("💡 Explanation Agent: Generating reasoning cards...")
        logger.info(f"Explaining {len(state.ranked_products)} recommendations")

        explanations: list[ExplanationCard] = []

        for ranked in state.ranked_products[:5]:  # Explain top 5 only
            try:
                card = await self._explain_product(ranked, state)
                explanations.append(card)
            except Exception as e:
                logger.warning(f"Failed to explain {ranked.product.name}: {e}")
                explanations.append(self._fallback_explanation(ranked))

        state.explanations = explanations

        # Attach explanation text back to RankedProduct objects
        exp_map = {e.product_id: e for e in explanations}
        for rp in state.ranked_products:
            if rp.product.id in exp_map:
                exp = exp_map[rp.product.id]
                rp.explanation = exp.headline + " | " + "; ".join(exp.reasons[:2])

        # Generate final conversational response
        state.final_response = await self._generate_final_response(state)
        state.agent_trace.append("  → Explanations and final response ready ✓")
        return state

    async def _explain_product(self, ranked: RankedProduct, state: AgentState) -> ExplanationCard:
        p = ranked.product
        fairness = p.price_fairness
        trust = p.trust_score

        context = f"""Product: {p.name}
Price: ₦{p.price:,} | Predicted Fair Price: ₦{fairness.predicted_fair_price:,} | Verdict: {fairness.verdict}
Rating: {p.rating or 'N/A'}/5 ({p.num_reviews or 0} reviews)
Trust Score: {trust.overall:.2f} | Flags: {', '.join(trust.flags) or 'None'}
Ranking Score: {ranked.composite_score:.3f}
User Query: {state.user_query}
Category: {p.category} | Use Case: {state.use_case or 'general'}"""

        import json
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": EXPLANATION_SYSTEM},
                {"role": "user", "content": context},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=256,
        )
        data = json.loads(resp.choices[0].message.content)

        return ExplanationCard(
            product_id=p.id,
            headline=data.get("headline", p.name[:50]),
            reasons=data.get("reasons", []),
            warnings=trust.flags if trust and trust.flags else data.get("warnings", []),
            value_verdict=data.get("value_verdict", fairness.verdict if fairness else ""),
            nigerian_context=data.get("nigerian_context", ""),
        )

    async def _generate_final_response(self, state: AgentState) -> str:
        """Generate the conversational assistant response."""
        if not state.ranked_products:
            return "Abeg, I no find products that match your request. Try adjusting your budget or category?"

        top3_names = [rp.product.name[:40] for rp in state.ranked_products[:3]]
        persona = state.user_persona.profile.persona_type.value if state.user_persona else "unknown"
        budget = f"₦{state.budget_max:,}" if state.budget_max else "your budget"

        prompt = f"""The user asked: "{state.user_query}"
Persona: {persona}
Top 3 products found: {top3_names}
Budget: {budget}

Write a friendly 2-3 sentence Nigerian shopping assistant response that:
1. Acknowledges their request
2. Briefly introduces the top recommendation
3. Mentions price value and why it fits them
Keep it conversational and warm. Use light Nigerian English (not heavy pidgin)."""

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150,
        )
        return resp.choices[0].message.content.strip()

    def _fallback_explanation(self, ranked: RankedProduct) -> ExplanationCard:
        p = ranked.product
        fairness = p.price_fairness
        trust = p.trust_score
        return ExplanationCard(
            product_id=p.id,
            headline=f"{p.name[:40]} — Rank #{ranked.rank}",
            reasons=[
                f"Relevance score: {ranked.semantic_score:.0%}",
                f"Price: ₦{p.price:,}" + (f" (Fair: ₦{fairness.predicted_fair_price:,})" if fairness else ""),
                f"Trust score: {trust.overall:.0%}" if trust else "Trust: N/A",
            ],
            warnings=trust.flags if trust else [],
            value_verdict=fairness.verdict if fairness else "unknown",
        )
