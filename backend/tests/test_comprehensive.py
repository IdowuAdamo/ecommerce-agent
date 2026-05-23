"""
NaijaShop AI — Comprehensive Unit & Integration Tests
Covers: category resolution, budget parsing, scoring, agent state, mock catalog.
Run with: pytest tests/test_comprehensive.py -v
"""
import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

# ── Category Resolution Tests ─────────────────────────────────────────────────

class TestCategoryResolution:
    """Tests for CommerceIntelAgent._resolve_category"""

    def setup_method(self):
        from app.agents.commerce_intel import CommerceIntelAgent
        self.agent = CommerceIntelAgent.__new__(CommerceIntelAgent)

    def test_freezer_exact_match(self):
        assert self.agent._resolve_category("freezer") == "freezers"

    def test_chest_freezer_multi_word(self):
        assert self.agent._resolve_category("chest freezer") == "freezers"

    def test_deep_freezer_multi_word(self):
        assert self.agent._resolve_category("deep freezer") == "freezers"

    def test_commercial_freezer(self):
        assert self.agent._resolve_category("commercial freezer") == "freezers"

    def test_fridge_maps_to_refrigerators(self):
        assert self.agent._resolve_category("fridge") == "refrigerators"

    def test_gaming_laptop_two_word(self):
        assert self.agent._resolve_category("gaming laptop") == "gaming_laptops"

    def test_office_shoes_two_word(self):
        assert self.agent._resolve_category("office shoes") == "shoes_men"

    def test_laptop_single(self):
        assert self.agent._resolve_category("laptop") == "laptops"

    def test_smartphone_maps_correctly(self):
        assert self.agent._resolve_category("smartphone") == "smartphones"

    def test_case_insensitive(self):
        assert self.agent._resolve_category("LAPTOP") == "laptops"
        assert self.agent._resolve_category("Freezer") == "freezers"

    def test_none_input(self):
        assert self.agent._resolve_category(None) is None

    def test_empty_string(self):
        # Empty string is falsy -> returns None
        assert self.agent._resolve_category("") is None

    def test_unknown_category_returns_none(self):
        result = self.agent._resolve_category("xxxxxxunknownxxxxxx")
        assert result is None

    def test_air_conditioner(self):
        assert self.agent._resolve_category("air conditioner") == "air_conditioners"

    def test_washing_machine(self):
        assert self.agent._resolve_category("washing machine") == "washing_machines"

    def test_gas_cooker(self):
        assert self.agent._resolve_category("gas cooker") == "gas_cookers"

    def test_water_dispenser(self):
        assert self.agent._resolve_category("water dispenser") == "water_dispensers"


# ── Naira Amount Parser Tests ─────────────────────────────────────────────────

class TestNairaParser:
    """Tests for discovery._parse_naira_amount"""

    def setup_method(self):
        from app.agents.discovery import _parse_naira_amount
        self.parse = _parse_naira_amount

    def test_plain_number(self):
        assert self.parse("50000") == 50000

    def test_k_suffix(self):
        assert self.parse("50k") == 50_000

    def test_m_suffix(self):
        assert self.parse("1m") == 1_000_000

    def test_float_k(self):
        assert self.parse("1.5k") == 1_500

    def test_float_m(self):
        assert self.parse("1.5m") == 1_500_000

    def test_naira_symbol_ascii(self):
        assert self.parse("N50,000") == 50_000

    def test_naira_symbol_unicode(self):
        # ₦ is U+20A6
        assert self.parse("\u20a6400,000") == 400_000

    def test_naira_with_commas(self):
        assert self.parse("1,000,000") == 1_000_000

    def test_million_literal(self):
        # "1 million" — regex captures 'm' from 'million' as the M suffix -> 1 * 1,000,000
        # This is correct and useful: "1 million" correctly resolves to 1,000,000 naira
        assert self.parse("1 million") == 1_000_000

    def test_none_input(self):
        assert self.parse(None) is None

    def test_empty_string(self):
        assert self.parse("") is None

    def test_budget_1m(self):
        assert self.parse("1m") == 1_000_000

    def test_budget_900k(self):
        assert self.parse("900k") == 900_000


# ── Category Match Score Tests ────────────────────────────────────────────────

class TestCategoryMatchScore:
    """Tests for RecommendationAgent._category_match_score"""

    def setup_method(self):
        from app.agents.recommendation import RecommendationAgent
        from app.schemas.agent import AgentState
        self.agent = RecommendationAgent.__new__(RecommendationAgent)
        self.AgentState = AgentState

    def _state(self, category: str) -> object:
        s = self.AgentState(
            session_id="test",
            user_query="test query",
            category=category,
            resolved_category=category,
        )
        return s

    def _product(self, category: str) -> object:
        from app.schemas.product import Product
        return Product(id="x", source="mock", name="Test", category=category, price=100000)

    def test_exact_match_returns_1(self):
        state = self._state("freezers")
        product = self._product("freezers")
        assert self.agent._category_match_score(product, state) == 1.0

    def test_sibling_returns_06(self):
        state = self._state("freezers")
        product = self._product("refrigerators")
        assert self.agent._category_match_score(product, state) == 0.6

    def test_unrelated_returns_0(self):
        state = self._state("freezers")
        product = self._product("irons")
        assert self.agent._category_match_score(product, state) == 0.0

    def test_no_category_state_returns_05(self):
        state = self._state("")
        state.resolved_category = None
        state.category = None
        product = self._product("freezers")
        assert self.agent._category_match_score(product, state) == 0.5

    def test_shoes_exact(self):
        state = self._state("shoes_men")
        product = self._product("shoes_men")
        assert self.agent._category_match_score(product, state) == 1.0

    def test_phones_vs_tablets_sibling(self):
        state = self._state("smartphones")
        product = self._product("tablets")
        assert self.agent._category_match_score(product, state) == 0.6


# ── Budget Proximity Score Tests ──────────────────────────────────────────────

class TestBudgetProximityScore:
    """Tests for RecommendationAgent._budget_proximity_score"""

    def setup_method(self):
        from app.agents.recommendation import RecommendationAgent
        from app.schemas.agent import AgentState
        from app.schemas.product import Product

        self.agent = RecommendationAgent.__new__(RecommendationAgent)
        self.AgentState = AgentState
        self.Product = Product

    def _state(self, budget_max: int) -> object:
        s = self.AgentState(session_id="t", user_query="q", budget_max=budget_max)
        return s

    def _product(self, price: int) -> object:
        return self.Product(id="x", source="mock", name="T", category="laptops", price=price)

    def test_no_budget_returns_neutral(self):
        state = self._state(None)
        assert self.agent._budget_proximity_score(self._product(100000), state) == 0.5

    def test_perfect_sweet_spot(self):
        state = self._state(500000)
        assert self.agent._budget_proximity_score(self._product(400000), state) == 1.0  # 80%

    def test_too_cheap_penalised(self):
        state = self._state(500000)
        score = self.agent._budget_proximity_score(self._product(100000), state)  # 20%
        assert score == 0.3

    def test_slightly_over_budget(self):
        state = self._state(500000)
        score = self.agent._budget_proximity_score(self._product(550000), state)  # 110%
        assert score == 0.6

    def test_way_over_budget(self):
        state = self._state(500000)
        score = self.agent._budget_proximity_score(self._product(700000), state)  # 140%
        assert score == 0.0


# ── Mock Catalog Tests ────────────────────────────────────────────────────────

class TestMockCatalog:
    """Tests for mock_products.json presence and validity"""

    def test_mock_catalog_file_exists(self):
        path = Path("app/data/mock_products.json")
        assert path.exists(), f"mock_products.json missing at {path.absolute()}"

    def test_mock_catalog_valid_json(self):
        path = Path("app/data/mock_products.json")
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_mock_catalog_has_freezers(self):
        path = Path("app/data/mock_products.json")
        with open(path) as f:
            data = json.load(f)
        freezers = [p for p in data if p.get("category") == "freezers"]
        assert len(freezers) >= 3, f"Expected >=3 freezers in mock catalog, got {len(freezers)}"

    def test_mock_catalog_products_valid(self):
        from app.schemas.product import Product
        path = Path("app/data/mock_products.json")
        with open(path) as f:
            data = json.load(f)
        for item in data:
            p = Product(**item)
            assert p.price > 0
            assert p.name
            assert p.category

    def test_mock_catalog_has_key_categories(self):
        path = Path("app/data/mock_products.json")
        with open(path) as f:
            data = json.load(f)
        categories = {p["category"] for p in data}
        required = {"laptops", "smartphones", "freezers", "televisions", "shoes_men"}
        missing = required - categories
        assert not missing, f"Missing categories in mock catalog: {missing}"


# ── CATEGORY_FALLBACKS Tests ──────────────────────────────────────────────────

class TestCategoryFallbacks:

    def test_freezers_fallback_to_refrigerators(self):
        from app.agents.commerce_intel import CATEGORY_FALLBACKS
        assert "refrigerators" in CATEGORY_FALLBACKS["freezers"]

    def test_gaming_laptops_fallback_to_laptops(self):
        from app.agents.commerce_intel import CATEGORY_FALLBACKS
        assert "laptops" in CATEGORY_FALLBACKS["gaming_laptops"]

    def test_smartphones_fallback_to_phones(self):
        from app.agents.commerce_intel import CATEGORY_FALLBACKS
        assert "phones" in CATEGORY_FALLBACKS["smartphones"]


# ── Product Schema Validation ─────────────────────────────────────────────────

class TestProductSchema:

    def test_product_to_model_input_text(self):
        from app.schemas.product import Product
        p = Product(id="t1", source="mock", name="HP Laptop", category="laptops",
                    description="Core i5, 8GB RAM", price=350000, seller="HP Store")
        text = p.to_model_input_text()
        assert "category: laptops" in text
        assert "name: HP Laptop" in text
        assert "seller: HP Store" in text

    def test_price_fairness_properties(self):
        from app.schemas.product import PriceFairness
        f = PriceFairness(actual_price=200000, predicted_fair_price=150000,
                          price_deviation_pct=33.3, verdict="overpriced",
                          fairness_score=0.4, confidence=0.85)
        assert f.is_overpriced is True
        assert f.is_great_deal is False

    def test_great_deal_detection(self):
        from app.schemas.product import PriceFairness
        f = PriceFairness(actual_price=80000, predicted_fair_price=100000,
                          price_deviation_pct=-20.0, verdict="great_deal",
                          fairness_score=0.9, confidence=0.85)
        assert f.is_great_deal is True
        assert f.is_overpriced is False


# ── Trust Scoring Tests ───────────────────────────────────────────────────────

class TestTrustScoring:

    def test_score_seller_jumia(self):
        from app.agents.trust_value import _score_seller
        assert _score_seller("Jumia") == 0.95

    def test_score_seller_official(self):
        from app.agents.trust_value import _score_seller
        assert _score_seller("Samsung Official Store") == 0.85

    def test_score_seller_none(self):
        from app.agents.trust_value import _score_seller
        assert _score_seller(None) == 0.4

    def test_score_rating_low_reviews(self):
        from app.agents.trust_value import _score_rating_authenticity
        score = _score_rating_authenticity(4.5, 3)
        assert score == 0.5  # insufficient data

    def test_score_rating_suspicious_perfect(self):
        from app.agents.trust_value import _score_rating_authenticity
        score = _score_rating_authenticity(5.0, 10)
        assert score == 0.4  # suspiciously perfect

    def test_score_review_count_high(self):
        from app.agents.trust_value import _score_review_count
        assert _score_review_count(1200) == 1.0

    def test_score_review_count_none(self):
        from app.agents.trust_value import _score_review_count
        assert _score_review_count(None) == 0.3
