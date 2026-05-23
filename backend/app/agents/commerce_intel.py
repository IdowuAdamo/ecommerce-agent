"""
Agent 3: Commerce Intelligence Agent.

Fetches products from Jumia (live scraping) with mock catalog fallback.
Handles deduplication, normalization, and vector indexing.
"""
from __future__ import annotations

import json
import logging
import asyncio
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

# ── Primary Synonym Map: user intent token → canonical Jumia category key ───────────────────────
CATEGORY_ALIASES: dict[str, str] = {
    # Computing
    "laptop": "laptops", "laptops": "laptops", "notebook": "laptops",
    "macbook": "laptops", "chromebook": "laptops", "computer": "laptops",
    "gaming laptop": "gaming_laptops", "gaming pc": "gaming_laptops",
    "workstation": "gaming_laptops",  # high-end
    "desktop": "desktops", "pc": "desktops", "all-in-one": "desktops",
    "monitor": "monitors", "screen": "monitors",
    "printer": "printers", "scanner": "printers",
    "router": "networking", "modem": "networking", "wifi": "networking",

    # Phones
    "phone": "smartphones", "phones": "smartphones",
    "smartphone": "smartphones", "mobile": "smartphones", "android": "smartphones",
    "iphone": "smartphones", "feature phone": "phones",
    "tablet": "tablets", "ipad": "tablets",
    "smartwatch": "smartwatches", "wristwatch": "watches",
    "power bank": "power_banks", "powerbank": "power_banks",

    # Electronics
    "tv": "televisions", "television": "televisions", "smart tv": "televisions",
    "generator": "generators", "gen": "generators", "petrol generator": "generators",
    "solar panel": "solar", "inverter": "solar", "solar": "solar",
    "camera": "cameras", "dslr": "cameras", "camcorder": "cameras",
    "speaker": "speakers", "bluetooth speaker": "speakers", "soundbar": "speakers",
    "headphone": "headphones", "earphone": "headphones", "earbuds": "headphones",
    "cctv": "security", "security camera": "security",
    "audio": "audio", "home theatre": "audio",

    # Appliances — SUBCATEGORY-SPECIFIC MAPPINGS
    "fridge": "refrigerators", "refrigerator": "refrigerators",
    "double door fridge": "refrigerators", "side by side fridge": "refrigerators",
    "freezer": "freezers", "chest freezer": "freezers", "deep freezer": "freezers",
    "commercial freezer": "freezers", "industrial freezer": "freezers",
    "upright freezer": "freezers", "display freezer": "freezers",
    "washing machine": "washing_machines", "washer": "washing_machines",
    "dryer": "washing_machines",
    "air conditioner": "air_conditioners", "ac": "air_conditioners",
    "split ac": "air_conditioners", "window ac": "air_conditioners",
    "fan": "fans", "ceiling fan": "fans", "standing fan": "fans",
    "microwave": "microwaves", "microwave oven": "microwaves",
    "gas cooker": "gas_cookers", "cooker": "gas_cookers", "oven": "gas_cookers",
    "blender": "blenders", "juicer": "blenders", "food processor": "blenders",
    "iron": "irons", "dry iron": "irons", "steam iron": "irons",
    "water dispenser": "water_dispensers", "dispenser": "water_dispensers",
    "vacuum cleaner": "vacuum_cleaners", "vacuum": "vacuum_cleaners",
    "appliance": "appliances", "home appliance": "appliances",

    # Fashion
    "shoe": "shoes_men", "shoes": "shoes_men", "sneaker": "shoes_men",
    "office shoes": "shoes_men", "footwear": "shoes_men",
    "sandal": "shoes_women", "heels": "shoes_women", "ladies shoes": "shoes_women",
    "clothing": "fashion_men", "clothes": "fashion_men",
    "dress": "fashion_women", "gown": "fashion_women", "skirt": "fashion_women",
    "bag": "bags", "handbag": "bags", "backpack": "bags",
    "watch": "watches",
    "jewelry": "jewelry", "necklace": "jewelry", "ring": "jewelry",

    # Health
    "skincare": "skincare", "cream": "skincare", "lotion": "skincare",
    "hair care": "haircare", "shampoo": "haircare", "conditioner": "haircare",
    "makeup": "makeup", "lipstick": "makeup", "foundation": "makeup",
    "vitamins": "vitamins", "supplement": "vitamins", "health": "health",

    # Home & Office
    "furniture": "furniture", "sofa": "furniture", "bed": "furniture",
    "mattress": "furniture", "chair": "furniture", "table": "furniture",
    "cookware": "cookware", "pot": "cookware", "pan": "cookware",
    "bedding": "bedding", "duvet": "bedding", "pillow": "bedding",
    "office supply": "office_supplies", "stationery": "office_supplies",

    # Gaming
    "gaming": "gaming", "game": "gaming", "console": "gaming_consoles",
    "playstation": "gaming_consoles", "xbox": "gaming_consoles",
    "nintendo": "gaming_consoles", "ps5": "gaming_consoles",
    "gaming accessories": "gaming_accessories", "gaming keyboard": "gaming_accessories",

    # Baby
    "baby": "baby", "diaper": "baby", "stroller": "baby",

    # Sports
    "sports": "sports", "gym equipment": "sports", "bicycle": "sports",
    "treadmill": "sports",

    # Supermarket
    "groceries": "groceries", "food": "groceries", "beverage": "groceries",
}

# ── Fallback map: if results are too few, broaden to these parent categories ────────────────
CATEGORY_FALLBACKS: dict[str, list[str]] = {
    "freezers":         ["refrigerators", "appliances"],
    "refrigerators":    ["freezers", "appliances"],
    "air_conditioners": ["fans", "appliances"],
    "blenders":         ["appliances"],
    "irons":            ["appliances"],
    "gas_cookers":      ["appliances"],
    "microwaves":       ["appliances"],
    "washing_machines": ["appliances"],
    "gaming_laptops":   ["laptops"],
    "smartphones":      ["phones"],
    "tablets":          ["phones"],
    "speakers":         ["audio", "headphones"],
    "headphones":       ["audio", "speakers"],
    "gaming_consoles":  ["gaming"],
    "shoes_men":        ["fashion_men"],
    "shoes_women":      ["fashion_women"],
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
            resolved_cat = self._resolve_category(state.category)
            state.resolved_category = resolved_cat  # propagate to ranking agent
            products = await self._fetch_products(state)
            products = self._deduplicate(products)
            
            # Apply budget filter with soft fallback
            filtered_products = self._filter_by_budget(products, state.budget_max)
            if not filtered_products and state.budget_max:
                state.agent_trace.append("  → Strict budget yielded 0 results, relaxing budget by 40%...")
                filtered_products = self._filter_by_budget(products, int(state.budget_max * 1.4))
                
            products = filtered_products or products # If still empty, just return everything to avoid "No Results" error

            state.raw_products = products[:80]  # Increased cap for better diversity in ranking
            state.retrieval_source = "hybrid"

            # Index into Pinecone for semantic search (in background)
            asyncio.create_task(self._index_products(products))

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
        """Hybrid Retrieval: Fetch from Jumia, Pinecone, and Mock with category intelligence."""
        category = self._resolve_category(state.category)
        SCRAPE_TIMEOUT = 12  # Hard wall-clock cap per scrape (prevents pipeline hang)
        MIN_THRESHOLD = 8    # Min results before activating category fallback

        async def safe_scrape(cat: str) -> list[Product]:
            if not cat:
                return []
            try:
                state.agent_trace.append(f"  → Live scraping Jumia: '{cat}'...")
                return await asyncio.wait_for(
                    self.scraper.scrape_category(cat, max_pages=1, max_products=20),
                    timeout=SCRAPE_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning(f"Scrape timeout for '{cat}' after {SCRAPE_TIMEOUT}s")
                return []
            except Exception as e:
                logger.warning(f"Live scraping failed for '{cat}': {e}")
                return []

        async def safe_vector() -> list[Product]:
            state.agent_trace.append("  → Querying Vector DB...")
            return await self._vector_search(state.user_query, category, state.budget_max)

        async def safe_mock(cat: str) -> list[Product]:
            return await self._load_mock_catalog(cat)

        # Step 1: Concurrent fetch from primary category + Vector DB
        primary_scrape, vector_results, mock_results = await asyncio.gather(
            safe_scrape(category),
            safe_vector(),
            safe_mock(category),
        )

        combined: list[Product] = []
        combined.extend(primary_scrape)
        combined.extend(vector_results)
        combined.extend(mock_results)

        # Step 2: If primary category yields too few products, apply category fallback
        primary_count = len(primary_scrape)
        if primary_count < MIN_THRESHOLD and category and category in CATEGORY_FALLBACKS:
            fallback_cats = CATEGORY_FALLBACKS[category]
            state.agent_trace.append(
                f"  → Primary '{category}' returned {primary_count} items. "
                f"Expanding to: {fallback_cats}"
            )
            fallback_tasks = [safe_scrape(fc) for fc in fallback_cats]
            fallback_results = await asyncio.gather(*fallback_tasks)
            for batch in fallback_results:
                combined.extend(batch)

        # Step 3: Normalize category tags so products carry correct metadata
        for p in combined:
            if category and not p.category:
                p.category = category

        logger.info(
            f"Hybrid retrieval pool: {len(combined)} products "
            f"(scrape={len(primary_scrape)}, vector={len(vector_results)}, mock={len(mock_results)})"
        )
        return combined

    async def _vector_search(
        self, query: str, category: Optional[str], budget_max: Optional[int]
    ) -> list[Product]:
        """Search Pinecone for semantically similar products."""
        try:
            query_vec = self.embedder.embed(query)
            matches = self.vector_store.search(
                query_embedding=query_vec,
                top_k=40,  # Increased from 20 for better recall
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
            # Exact match first
            filtered = [p for p in catalog if p.category == category]
            if not filtered:
                # Fuzzy fallback: partial string match (handles legacy names)
                # e.g. canonical 'smartphones' matches mock 'phones'
                filtered = [
                    p for p in catalog
                    if category in (p.category or "") or (p.category or "") in category
                ]
            if filtered:
                return filtered
            # No match at all — return full catalog (ranked agent will filter via category_match_score)
            return catalog
        return catalog

    def _resolve_category(self, category: Optional[str]) -> Optional[str]:
        """Resolve user-provided category string to canonical Jumia category key.
        Uses longest-match to handle multi-word phrases like 'chest freezer'.
        """
        if not category:
            return None
        text = category.lower().strip()
        # Try longest-match first (2-word phrases before 1-word)
        for phrase_len in (3, 2, 1):
            tokens = text.split()
            if len(tokens) >= phrase_len:
                for i in range(len(tokens) - phrase_len + 1):
                    phrase = " ".join(tokens[i:i + phrase_len])
                    if phrase in CATEGORY_ALIASES:
                        return CATEGORY_ALIASES[phrase]
        # Direct match on full text
        if text in CATEGORY_ALIASES:
            return CATEGORY_ALIASES[text]
        # If still unknown but the key exists in JUMIA_CATEGORIES directly, use it
        from app.services.jumia_client import JUMIA_CATEGORIES
        if text in JUMIA_CATEGORIES:
            return text
        return None  # Unknown category — vector search will be unconstrained

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
