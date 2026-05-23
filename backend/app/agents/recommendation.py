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

import asyncio
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
        self.w_budget_proximity = s.rec_budget_proximity_weight
        self.w_category = getattr(s, "rec_category_match_weight", 0.25)
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
            # Compute query embedding (CPU-bound: run in thread to keep event loop free)
            query_emb = await asyncio.to_thread(self.embedder.embed, state.user_query)

            # Batch-embed all products in one model call (critical optimization)
            product_texts = [
                f"{p.name} {p.category} {p.description}"[:256] for p in products
            ]
            product_embs = await asyncio.to_thread(self.embedder.embed_batch, product_texts)

            # Score each product using precomputed embeddings
            scored = [
                self._compute_composite_score(p, emb, query_emb, state.user_persona, state)
                for p, emb in zip(products, product_embs)
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

            if state.ranked_products:
                state.agent_trace.append(
                    f"  → Ranked {len(state.ranked_products)} products. "
                    f"Top: '{state.ranked_products[0].product.name[:40]}'"
                )
            else:
                state.agent_trace.append("  → No ranked products")

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
        product_emb: list[float],
        query_emb: list[float],
        persona,
        state: AgentState,
    ) -> dict:
        # 1. Semantic similarity (uses precomputed embedding)
        semantic = self.embedder.cosine_similarity(query_emb, product_emb)
        semantic = max(0.0, semantic)

        # 2. Category match — hard gate against category leakage
        category_match = self._category_match_score(product, state)

        # 3. Behavioral match
        behavioral = self._behavioral_score(product, persona)

        # 4. Price fairness (from DeBERTa model)
        fairness = 0.5  # default if not scored
        if product.price_fairness:
            fairness = product.price_fairness.fairness_score

        # 5. Trust score
        trust = 0.5  # default
        if product.trust_score:
            trust = product.trust_score.overall

        # 6. Contextual relevance
        contextual = self._contextual_score(product, state)

        # 7. Budget proximity (rewards products closer to max budget if provided)
        budget_prox = self._budget_proximity_score(product, state)

        # Normalize weights so they still sum to 1 with category_match added
        total_w = (
            self.w_semantic + self.w_category + self.w_behavioral
            + self.w_fairness + self.w_trust + self.w_context + self.w_budget_proximity
        )
        composite = (
            self.w_semantic * semantic
            + self.w_category * category_match
            + self.w_behavioral * behavioral
            + self.w_fairness * fairness
            + self.w_trust * trust
            + self.w_context * contextual
            + self.w_budget_proximity * budget_prox
        ) / total_w

        return {
            "product": product,
            "composite": composite,
            "semantic": semantic,
            "category_match": category_match,
            "behavioral": behavioral,
            "fairness": fairness,
            "trust": trust,
            "contextual": contextual,
            "budget_prox": budget_prox,
            "embedding": product_emb,
        }

    # ── Category Match \u2014 Anti-Leakage Scoring ───────────────────────────────
    # Subcategory relationships: defines which product categories are related.
    # An exact match = 1.0; a related sibling = 0.6; totally unrelated = 0.0.
    _CATEGORY_SIBLINGS: dict[str, set[str]] = {
        "freezers":         {"refrigerators"},
        "refrigerators":    {"freezers"},
        "washing_machines": {"appliances"},
        "air_conditioners": {"fans", "appliances"},
        "fans":             {"air_conditioners", "appliances"},
        "microwaves":       {"gas_cookers", "appliances"},
        "gas_cookers":      {"microwaves", "appliances"},
        "blenders":         {"appliances"},
        "irons":            {"appliances"},
        "gaming_laptops":   {"laptops"},
        "laptops":          {"gaming_laptops", "desktops"},
        "smartphones":      {"phones", "tablets"},
        "tablets":          {"smartphones", "phones"},
        "speakers":         {"audio", "headphones"},
        "headphones":       {"audio", "speakers"},
        "gaming_consoles":  {"gaming", "gaming_accessories"},
        "shoes_men":        {"fashion_men"},
        "shoes_women":      {"fashion_women"},
    }

    def _category_match_score(self, product: Product, state: AgentState) -> float:
        """
        Scores how well a product's category matches the user's requested category.
        - Exact subcategory match \u2192 1.0
        - Known sibling/related subcategory \u2192 0.6
        - Parent/child match \u2192 0.4
        - Completely unrelated \u2192 0.0

        This is the primary guard against 'category leakage' (e.g., irons
        appearing in freezer results because both are 'appliances').
        """
        resolved_category = getattr(state, "_resolved_category", None) or state.category
        if not resolved_category:
            return 0.5  # No category constraint \u2014 neutral

        wanted = resolved_category.lower().replace(" ", "_")
        have = (product.category or "").lower().replace(" ", "_")

        if have == wanted:
            return 1.0  # Exact match

        siblings = self._CATEGORY_SIBLINGS.get(wanted, set())
        if have in siblings:
            return 0.6  # Sibling/related

        # Check parent-child: e.g. "appliances" is parent of "freezers"
        if wanted in have or have in wanted:
            return 0.4  # Partial hierarchical match

        return 0.0  # No relationship \u2014 strongly penalise

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

    # Keywords that signal commercial/industrial product grade
    _COMMERCIAL_SIGNALS = {
        "commercial", "industrial", "business", "professional", "heavy duty",
        "heavy-duty", "large capacity", "chest", "deep", "upright", "display",
    }

    def _contextual_score(self, product: Product, state: AgentState) -> float:
        score = 0.5

        # Urgency — prefer in-stock items
        if state.urgency in ("medium", "high") and product.stock_info == "In stock":
            score += 0.2

        # Use-case match (keyword overlap in name + description)
        use_case = (state.use_case or "").lower()
        product_text = (product.name + " " + (product.description or "")).lower()
        if use_case and use_case in product_text:
            score += 0.15

        # Business context: boost products matching commercial-grade signals
        business_ctx = getattr(state, "business_context", None)
        if business_ctx == "commercial":
            if any(signal in product_text for signal in self._COMMERCIAL_SIGNALS):
                score += 0.25  # Strong boost for commercial intent

        # Feature keyword match: if discovery extracted specific features, reward matches
        features = state.extracted_intent.get("features", []) if state.extracted_intent else []
        matched_features = sum(
            1 for feat in features
            if feat.lower() in product_text
        )
        if features:
            score += 0.1 * (matched_features / len(features))

        return min(1.0, score)


    def _budget_proximity_score(self, product: Product, state: AgentState) -> float:
        """
        Rewards products that reasonably match the user's budget ceiling.
        High budget implies high quality expectation; don't just return cheap items.
        """
        if not state.budget_max:
            return 0.5  # Neutral if no budget specified

        ratio = product.price / state.budget_max

        if 0.7 <= ratio <= 1.0:
            return 1.0   # Perfect sweet spot (70-100% of budget)
        elif 0.4 <= ratio < 0.7:
            return 0.7   # Mid-range (40-70% of budget)
        elif 1.0 < ratio <= 1.2:
            return 0.6   # Slightly over budget stretch
        elif ratio < 0.4:
            return 0.3   # Suspiciously cheap for this budget (low quality proxy)
        else:
            return 0.0   # Way over budget

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
            best_idx = 0
            if not selected:
                # First item: just pick the top-scored
                best_score = remaining[0]["composite"]
                for i, item in enumerate(remaining):
                    if item["composite"] > best_score:
                        best_score = item["composite"]
                        best_idx = i
            else:
                # MMR score = λ * relevance - (1-λ) * max_similarity_to_selected
                best_mmr = -float("inf")
                for i, item in enumerate(remaining):
                    sim = self._max_sim_to_selected(item["embedding"], selected)
                    mmr_score = λ * item["composite"] - (1 - λ) * sim
                    if mmr_score > best_mmr:
                        best_mmr = mmr_score
                        best_idx = i
            
            selected.append(remaining.pop(best_idx))

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
