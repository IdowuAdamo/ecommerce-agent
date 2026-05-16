"""
Agent 5: Recommendation & Ranking Agent.

Combines multiple signals into a composite ranking score:
  - Semantic similarity (sentence-transformer cosine similarity)
  - Behavioral match (user persona affinity)
  - Price fairness (DeBERTa model score)
  - Trust score (seller + rating + discount legitimacy)
  - Contextual relevance (urgency, location, use-case)

Uses MMR (Maximal Marginal Relevance) for diversity in top-K results.
Optimized for NDCG@10 and Hit Rate.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from app.config import get_settings
from app.models.embeddings import EmbeddingModel
from app.schemas.agent import AgentState
from app.schemas.product import Product, RankedProduct
from app.schemas.user import UserPersona

logger = logging.getLogger(__name__)


class RecommendationAgent:
    """Multi-signal product ranking with MMR diversity."""

    def __init__(self):
        self.embedder = EmbeddingModel.get_instance()
        s = get_settings()
        self.w_semantic = s.rec_semantic_weight
        self.w_behavioral = s.rec_behavioral_weight
        self.w_fairness = s.rec_price_fairness_weight
        self.w_trust = s.rec_trust_weight
        self.w_context = s.rec_contextual_weight
        self.top_k = s.rec_top_k
        self.mmr_lambda = s.rec_diversity_lambda

    async def run(self, state: AgentState) -> AgentState:
        state.agent_trace.append("🎯 Recommendation Agent: Ranking products...")
        logger.info(f"Ranking {len(state.scored_products)} products")

        products = state.scored_products or state.raw_products
        if not products:
            state.agent_trace.append("  → No products to rank")
            return state

        try:
            # Compute query embedding for semantic scoring
            query_emb = self.embedder.embed(state.user_query)

            # Score each product
            scored = [
                self._compute_composite_score(p, query_emb, state.user_persona, state)
                for p in products
            ]

            # Sort by composite score
            scored.sort(key=lambda x: x["composite"], reverse=True)

            # Apply MMR for diversity
            selected = self._mmr_rerank(scored, query_emb, k=self.top_k)

            # Build RankedProduct objects
            state.ranked_products = [
                RankedProduct(
                    product=item["product"],
                    rank=i + 1,
                    composite_score=round(item["composite"], 4),
                    semantic_score=round(item["semantic"], 4),
                    behavioral_score=round(item["behavioral"], 4),
                    price_fairness_score=round(item["fairness"], 4),
                    trust_score_val=round(item["trust"], 4),
                    contextual_score=round(item["contextual"], 4),
                )
                for i, item in enumerate(selected)
            ]

            state.agent_trace.append(
                f"  → Ranked {len(state.ranked_products)} products. "
                f"Top: '{state.ranked_products[0].product.name[:40]}...'"
                if state.ranked_products else "  → No ranked products"
            )

        except Exception as e:
            logger.error(f"Recommendation Agent failed: {e}")
            state.errors.append(f"Recommendation: {e}")
            # Fallback: sort by trust score
            state.ranked_products = [
                RankedProduct(product=p, rank=i+1, composite_score=0.5)
                for i, p in enumerate(products[:self.top_k])
            ]

        return state

    def _compute_composite_score(
        self,
        product: Product,
        query_emb: list[float],
        persona: Optional[UserPersona],
        state: AgentState,
    ) -> dict:
        # 1. Semantic similarity
        product_text = f"{product.name} {product.category} {product.description}"[:256]
        product_emb = self.embedder.embed(product_text)
        semantic = self.embedder.cosine_similarity(query_emb, product_emb)
        semantic = max(0.0, semantic)

        # 2. Behavioral match
        behavioral = self._behavioral_score(product, persona)

        # 3. Price fairness (from DeBERTa model)
        fairness = 0.5  # default if not scored
        if product.price_fairness:
            fairness = product.price_fairness.fairness_score

        # 4. Trust score
        trust = 0.5  # default
        if product.trust_score:
            trust = product.trust_score.overall

        # 5. Contextual relevance
        contextual = self._contextual_score(product, state)

        composite = (
            self.w_semantic * semantic
            + self.w_behavioral * behavioral
            + self.w_fairness * fairness
            + self.w_trust * trust
            + self.w_context * contextual
        )

        return {
            "product": product,
            "composite": composite,
            "semantic": semantic,
            "behavioral": behavioral,
            "fairness": fairness,
            "trust": trust,
            "contextual": contextual,
            "embedding": product_emb,
        }

    def _behavioral_score(self, product: Product, persona: Optional[UserPersona]) -> float:
        if not persona:
            return 0.5

        score = 0.5

        # Category affinity
        cat_affinity = persona.category_affinity_scores.get(product.category, 0)
        score += min(0.3, cat_affinity)

        # Budget alignment
        budget_min, budget_max = persona.inferred_budget_range
        if budget_min <= product.price <= budget_max:
            score += 0.2
        elif product.price < budget_min:
            score += 0.1  # under budget — still ok

        # Brand tier alignment
        tier = persona.preferred_brand_tier
        if tier == "premium" and product.brand in ("Apple", "Samsung", "Sony", "HP", "Dell"):
            score += 0.15
        elif tier == "budget" and product.brand in ("Tecno", "Infinix", "Itel"):
            score += 0.15

        return min(1.0, max(0.0, score))

    def _contextual_score(self, product: Product, state: AgentState) -> float:
        score = 0.5

        # Urgency — prefer in-stock items
        if state.urgency in ("medium", "high") and product.stock_info == "In stock":
            score += 0.3

        # Use-case match (keyword overlap)
        use_case = (state.use_case or "").lower()
        if use_case and use_case in product.description.lower():
            score += 0.2

        return min(1.0, score)

    def _mmr_rerank(self, scored: list[dict], query_emb: list[float], k: int) -> list[dict]:
        """
        Maximal Marginal Relevance reranking for result diversity.
        Balances relevance with dissimilarity to already-selected items.
        """
        if len(scored) <= k:
            return scored

        selected: list[dict] = []
        remaining = list(scored)
        λ = self.mmr_lambda

        while len(selected) < k and remaining:
            if not selected:
                # First item: just pick the top-scored
                best = max(remaining, key=lambda x: x["composite"])
            else:
                # MMR score = λ * relevance - (1-λ) * max_similarity_to_selected
                best = max(
                    remaining,
                    key=lambda x: λ * x["composite"] - (1 - λ) * self._max_sim_to_selected(
                        x["embedding"], selected
                    ),
                )
            selected.append(best)
            remaining.remove(best)

        return selected

    def _max_sim_to_selected(self, emb: list[float], selected: list[dict]) -> float:
        """Max cosine similarity between a candidate and all selected items."""
        if not selected:
            return 0.0
        sims = [
            self.embedder.cosine_similarity(emb, s["embedding"])
            for s in selected
        ]
        return max(sims)
