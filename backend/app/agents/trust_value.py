"""
Agent 4: Trust & Value Agent ★ (Core Competitive Differentiator)

CHANGELOG — v2.0.0:
  Price fairness scoring now uses the OpenAI API-based PricePredictorModel
  instead of the DeBERTa fine-tuned model. Key changes:
    - predict() is now async (HTTP call) → _score_product is now async
    - All products receive full ML-based scoring (no DeBERTa batch cap needed)
    - asyncio.to_thread() removed — no CPU-bound blocking; pure async HTTP
    - Products are scored concurrently via asyncio.gather() for maximum speed

Uses the OpenAI price predictor to:
  1. Estimate the fair market price for each product
  2. Calculate price fairness score (actual vs. predicted)
  3. Detect suspicious pricing, fake discounts, and poor value
  4. Generate trust scores from seller quality, rating authenticity, etc.
"""
from __future__ import annotations

import asyncio
import logging
import re

from app.models.price_predictor import PricePredictorModel
from app.schemas.agent import AgentState
from app.schemas.product import Product, PriceFairness, TrustScore

logger = logging.getLogger(__name__)


class TrustValueAgent:
    """
    Scores every product for price fairness and trustworthiness.
    Uses the OpenAI API price predictor for fair price estimation.
    """

    def __init__(self) -> None:
        self.price_model = PricePredictorModel.get_instance()

    async def run(self, state: AgentState) -> AgentState:
        state.agent_trace.append("\U0001f510 Trust & Value Agent: Scoring products...")
        products = state.raw_products
        logger.info(f"Trust & Value Agent processing {len(products)} products")

        # Score all products concurrently — each predict() is a fast async HTTP call.
        # No asyncio.to_thread needed (no CPU-bound blocking).
        tasks = [self._score_product(p) for p in products]
        scored = list(await asyncio.gather(*tasks, return_exceptions=False))

        state.scored_products = scored
        fair_count = sum(
            1 for p in scored
            if p.price_fairness and p.price_fairness.verdict in ("fair", "great_deal")
        )
        state.agent_trace.append(
            f"  \u2192 Scored {len(scored)} products (OpenAI API), "
            f"{fair_count} fair/great-deal priced"
        )
        return state

    async def _score_product(self, product: Product) -> Product:
        """Async: add price fairness and trust score to a product."""
        try:
            return await self._score_product_full(product)
        except Exception as e:
            logger.warning(f"Full scoring failed for {product.name!r}: {e}; using heuristic")
            return self._heuristic_score(product)

    async def _score_product_full(self, product: Product) -> Product:
        """Full scoring: OpenAI price prediction + trust heuristics."""
        # ── Price Fairness (OpenAI API) ───────────────────────────────────────
        input_text = product.to_model_input_text()
        predicted_price = await self.price_model.predict(input_text)
        fairness_score, deviation_pct, verdict = (
            self.price_model.compute_price_fairness_score(product.price, predicted_price)
        )

        # Discount legitimacy check
        is_fake_discount = False
        discount_legit = 1.0
        if product.discount_pct and product.old_price:
            expected_discount = (product.old_price - product.price) / product.old_price * 100
            diff = abs(expected_discount - product.discount_pct)
            if diff > 5:
                is_fake_discount = True
                discount_legit = 0.3
        elif product.discount_pct and product.discount_pct > 60:
            is_fake_discount = True
            discount_legit = 0.2
        elif product.discount_pct:
            discount_legit = min(1.0, 0.6 + (product.discount_pct / 100) * 0.4)

        product.price_fairness = PriceFairness(
            actual_price=product.price,
            predicted_fair_price=int(predicted_price),
            price_deviation_pct=round(deviation_pct, 1),
            verdict=verdict,
            fairness_score=round(fairness_score, 3),
            confidence=0.80,   # OpenAI API confidence
            explanation=_fairness_explanation(verdict, deviation_pct, predicted_price),
        )

        # ── Trust Score ───────────────────────────────────────────────────────
        flags: list[str] = []

        seller_score = _score_seller(product.seller)

        rating_score = 0.5
        if product.rating is not None and product.num_reviews is not None:
            rating_score = _score_rating_authenticity(product.rating, product.num_reviews)
        elif product.rating is not None:
            rating_score = 0.6

        review_score = _score_review_count(product.num_reviews)
        stock_score = 0.9 if product.stock_info == "In stock" else 0.3

        if is_fake_discount:
            flags.append("suspicious_discount")
        if seller_score < 0.4:
            flags.append("unknown_seller")
        if rating_score < 0.4:
            flags.append("low_rating")
        if verdict == "suspicious":
            flags.append("price_anomaly")

        overall = (
            seller_score * 0.25
            + rating_score * 0.25
            + discount_legit * 0.20
            + review_score * 0.15
            + stock_score * 0.15
        )

        product.trust_score = TrustScore(
            overall=round(overall, 3),
            seller_score=round(seller_score, 3),
            rating_authenticity=round(rating_score, 3),
            discount_legitimacy=round(discount_legit, 3),
            review_count_score=round(review_score, 3),
            stock_reliability=round(stock_score, 3),
            flags=flags,
        )
        return product

    def _heuristic_score(self, product: Product) -> Product:
        """
        Fast heuristic trust/fairness score — used as fallback when the
        OpenAI API call fails (network error, rate limit, etc.).
        """
        disc = product.discount_pct or 0
        if disc > 50:
            verdict, fairness = "suspicious", 0.3
        elif disc > 20:
            verdict, fairness = "great_deal", 0.85
        elif disc > 5:
            verdict, fairness = "fair", 0.75
        else:
            verdict, fairness = "fair", 0.65

        product.price_fairness = PriceFairness(
            actual_price=product.price,
            predicted_fair_price=product.price,
            price_deviation_pct=0.0,
            verdict=verdict,
            fairness_score=fairness,
            confidence=0.4,   # low confidence — heuristic only
        )
        trust_val = _score_seller(product.seller)
        rating_val = _score_rating_authenticity(product.rating, product.num_reviews or 0)
        review_val = _score_review_count(product.num_reviews)
        overall = round(0.5 * trust_val + 0.3 * rating_val + 0.2 * review_val, 3)
        product.trust_score = TrustScore(
            overall=overall,
            seller_quality=trust_val,
            rating_authenticity=rating_val,
            review_count_score=review_val,
            discount_legitimacy=min(1.0, disc / 30) if disc else 0.5,
        )
        return product


# ── Module-level helper functions ─────────────────────────────────────────────

def _score_seller(seller: str | None) -> float:
    """Score seller quality. Jumia/official stores score highest."""
    if not seller:
        return 0.4
    seller_lower = seller.lower()
    if "jumia" in seller_lower:
        return 0.95
    if any(kw in seller_lower for kw in ["official", "authorized", "nigeria"]):
        return 0.85
    if len(seller_lower) > 5:
        return 0.7
    return 0.5


def _score_rating_authenticity(rating: float, num_reviews: int) -> float:
    """Flag suspicious rating patterns (too perfect, too few reviews)."""
    if num_reviews < 5:
        return 0.5
    if rating == 5.0 and num_reviews < 50:
        return 0.4
    if rating < 2.5:
        return 0.2
    rating_norm = (rating - 1) / 4
    review_norm = min(1.0, num_reviews / 500)
    return round(rating_norm * 0.6 + review_norm * 0.4, 3)


def _score_review_count(num_reviews: int | None) -> float:
    if not num_reviews:
        return 0.3
    if num_reviews >= 1000:
        return 1.0
    if num_reviews >= 100:
        return 0.8
    if num_reviews >= 20:
        return 0.6
    return 0.4


def _fairness_explanation(verdict: str, deviation_pct: float, predicted: float) -> str:
    pred_fmt = f"\u20a6{predicted:,.0f}"
    dev = abs(deviation_pct)
    if verdict == "great_deal":
        return f"Priced {dev:.0f}% below the estimated fair price of {pred_fmt}. Excellent value!"
    elif verdict == "fair":
        return f"Price is within {dev:.0f}% of the estimated fair market price ({pred_fmt})."
    elif verdict == "slightly_overpriced":
        return f"About {dev:.0f}% above the estimated fair price of {pred_fmt}."
    elif verdict == "overpriced":
        return f"Significantly overpriced — {dev:.0f}% above fair market estimate of {pred_fmt}."
    elif verdict == "suspicious":
        return f"Price is {dev:.0f}% above what similar products typically cost ({pred_fmt})."
    return f"Estimated fair price: {pred_fmt}."
