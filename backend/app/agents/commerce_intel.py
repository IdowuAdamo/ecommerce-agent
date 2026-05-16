"""
Agent 3: Commerce Intelligence Agent.

Fetches products from Jumia (live scraping) with mock catalog fallback.
Handles deduplication, normalization, and vector indexing.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional
import uuid

from app.schemas.agent import AgentState
from app.schemas.product import Product
from app.services.jumia_client import JumiaScraper, JUMIA_CATEGORIES
from app.models.embeddings import EmbeddingModel
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

# Category aliases — map extracted intent to Jumia categories
CATEGORY_ALIASES: dict[str, str] = {
    "laptop": "laptops", "laptops": "laptops", "computer": "laptops",
    "phone": "phones", "phones": "phones", "smartphone": "phones", "mobile": "phones",
    "tv": "televisions", "television": "televisions", "monitor": "televisions",
    "appliance": "appliances", "blender": "appliances", "fridge": "appliances",
    "generator": "generators", "gen": "generators",
    "camera": "cameras",
    "speaker": "audio", "headphone": "audio", "earphone": "audio",
    "gaming": "gaming", "game": "gaming", "console": "gaming",
    "health": "health", "beauty": "health",
    "shoe": "shoes_men", "shoes": "shoes_men", "sneaker": "shoes_men", "office shoes": "shoes_men", "footwear": "shoes_men",
    "clothing": "fashion_women", "clothes": "fashion_women", "dress": "fashion_women",
}

MOCK_CATALOG_PATH = Path(__file__).parent.parent / "data" / "mock_products.json"


class CommerceIntelAgent:
    """Fetches, normalizes, and indexes products from Jumia or mock catalog."""

    def __init__(self):
        self.scraper = JumiaScraper()
        self.embedder = EmbeddingModel.get_instance()
        self.vector_store = VectorStoreService.get_instance()
        self._mock_catalog: Optional[list[Product]] = None

    async def run(self, state: AgentState) -> AgentState:
        state.agent_trace.append("🛒 Commerce Agent: Fetching products...")
        logger.info(f"Commerce Agent for category: {state.category}")

        try:
            products = await self._fetch_products(state)
            products = self._deduplicate(products)
            products = self._filter_by_budget(products, state.budget_max)

            state.raw_products = products[:50]  # cap at 50 for processing
            state.retrieval_source = products[0].source if products else "mock"

            # Index into Pinecone for semantic search
            await self._index_products(products)

            state.agent_trace.append(
                f"  → Found {len(products)} products from '{state.retrieval_source}'"
            )
            logger.info(f"Fetched {len(products)} products")

        except Exception as e:
            logger.error(f"Commerce Agent failed: {e}")
            state.errors.append(f"Commerce: {e}")
            # Always fall back to mock
            state.raw_products = await self._load_mock_catalog(state.category)
            state.retrieval_source = "mock"

        return state

    async def _fetch_products(self, state: AgentState) -> list[Product]:
        """Try live Jumia scraping first, fall back to Pinecone, then mock."""
        category = self._resolve_category(state.category)

        # 1. Live scrape from Jumia (prioritized)
        try:
            state.agent_trace.append(f"  → Live scraping Jumia for '{category}'...")
            products = await self.scraper.scrape_category(
                category or "phones",
                max_pages=1,  # Keep it fast for real-time chat
                max_products=20,
            )
            if products:
                return products
        except Exception as e:
            logger.warning(f"Live scraping failed: {e}")

        # 2. Semantic vector search on previously cached products
        existing = await self._vector_search(state.user_query, category, state.budget_max)
        if existing and len(existing) >= 3:
            state.agent_trace.append(f"  → Using {len(existing)} cached vector results")
            return existing

        # 3. Fall back to mock catalog
        return await self._load_mock_catalog(category)

    async def _vector_search(
        self, query: str, category: Optional[str], budget_max: Optional[int]
    ) -> list[Product]:
        """Search Pinecone for semantically similar products."""
        try:
            query_vec = self.embedder.embed(query)
            matches = self.vector_store.search(
                query_embedding=query_vec,
                top_k=20,
                category_filter=category,
                price_max=budget_max,
            )
            return [_pinecone_match_to_product(m) for m in matches if m["score"] > 0.3]
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

    async def _index_products(self, products: list[Product]) -> None:
        """Embed and upsert products into Pinecone (background task)."""
        try:
            texts = [f"{p.name} {p.category} {p.description}"[:256] for p in products]
            embeddings = self.embedder.embed_batch(texts)
            self.vector_store.upsert_products(products, embeddings)
        except Exception as e:
            logger.warning(f"Product indexing failed: {e}")

    async def _load_mock_catalog(self, category: Optional[str] = None) -> list[Product]:
        """Load fallback mock catalog from JSON file."""
        if self._mock_catalog is None:
            try:
                with open(MOCK_CATALOG_PATH) as f:
                    data = json.load(f)
                self._mock_catalog = [Product(**p) for p in data]
            except Exception as e:
                logger.error(f"Failed to load mock catalog: {e}")
                self._mock_catalog = _generate_minimal_mock()

        catalog = self._mock_catalog
        if category:
            filtered = [p for p in catalog if p.category == category]
            return filtered
        return catalog

    def _resolve_category(self, category: Optional[str]) -> Optional[str]:
        if not category:
            return None
        return CATEGORY_ALIASES.get(category.lower(), category.lower())

    def _deduplicate(self, products: list[Product]) -> list[Product]:
        seen_names: set[str] = set()
        unique: list[Product] = []
        for p in products:
            key = p.name.lower()[:50]
            if key not in seen_names:
                seen_names.add(key)
                unique.append(p)
        return unique

    def _filter_by_budget(self, products: list[Product], budget_max: Optional[int]) -> list[Product]:
        if not budget_max:
            return products
        # Include products within 20% over budget (user might stretch)
        threshold = budget_max * 1.2
        within_budget = [p for p in products if p.price <= budget_max]
        slightly_over = [p for p in products if budget_max < p.price <= threshold]
        return within_budget + slightly_over[:3]  # add max 3 stretch options


def _pinecone_match_to_product(match: dict) -> Product:
    meta = match.get("metadata", {})
    return Product(
        id=match["id"],
        source=meta.get("source", "jumia"),
        name=meta.get("name", "Unknown Product"),
        category=meta.get("category", "general"),
        price=int(meta.get("price", 0)),
        rating=meta.get("rating") or None,
        discount_pct=meta.get("discount_pct") or None,
        brand=meta.get("brand") or None,
    )


def _generate_minimal_mock() -> list[Product]:
    """Minimal hardcoded fallback products."""
    return [
        Product(
            id="mock001", source="mock", name="Tecno Spark 20 Pro 256GB",
            category="phones", price=195_000, rating=4.2, num_reviews=340,
            seller="Jumia", brand="Tecno",
            description="6.78\" display, 108MP camera, 5000mAh battery",
        ),
        Product(
            id="mock002", source="mock", name="HP 255 G9 Laptop 16GB RAM 512GB SSD",
            category="laptops", price=450_000, rating=4.5, num_reviews=127,
            seller="HP Official Store", brand="HP",
            description="AMD Ryzen 5, Windows 11, 15.6\" FHD display",
        ),
        Product(
            id="mock003", source="mock", name="Samsung 65\" 4K Smart TV",
            category="televisions", price=780_000, rating=4.7, num_reviews=89,
            seller="Samsung Nigeria", brand="Samsung",
            description="Crystal UHD, HDR, Tizen OS, Built-in WiFi",
        ),
        Product(
            id="mock004", source="mock", name="Binatone 1.5L Blender BLG-415",
            category="appliances", price=31_750, old_price=36_830,
            discount_pct=14, rating=4.0, num_reviews=283,
            seller="Jumia", brand="Binatone",
            description="1.5 litre capacity, stainless steel blades",
        ),
        Product(
            id="mock005", source="mock", name="Thermocool 7.5KVA Generator TEC-GEN-75",
            category="generators", price=950_000, rating=4.3, num_reviews=56,
            seller="Thermocool Nigeria", brand="Thermocool",
            description="7500W peak power, electric start, fuel gauge",
        ),
    ]
