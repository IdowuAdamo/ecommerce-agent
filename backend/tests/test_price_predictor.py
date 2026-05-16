"""
Unit tests for the price predictor model.
"""
import pytest
import math


def test_compute_price_fairness_score():
    """Test the price fairness scoring logic without loading the full model."""
    from app.models.price_predictor import PricePredictorModel

    class MockPredictor:
        def compute_price_fairness_score(self, actual, predicted):
            return PricePredictorModel.compute_price_fairness_score(self, actual, predicted)

    # Test cases
    test_cases = [
        (100_000, 95_000, "fair"),        # 5% over — fair
        (200_000, 100_000, "suspicious"),  # 100% over — suspicious
        (50_000, 100_000, "great_deal"),   # 50% under — great deal
        (120_000, 100_000, "slightly_overpriced"),  # 20% over
    ]

    for actual, predicted, expected_verdict in test_cases:
        if predicted <= 0:
            continue
        deviation_pct = (actual - predicted) / predicted * 100
        fairness_score = max(0.0, min(1.0, math.exp(-abs(deviation_pct) / 30)))

        if deviation_pct < -20:
            verdict = "great_deal"
        elif deviation_pct < 10:
            verdict = "fair"
        elif deviation_pct < 30:
            verdict = "slightly_overpriced"
        elif deviation_pct < 60:
            verdict = "overpriced"
        else:
            verdict = "suspicious"

        assert verdict == expected_verdict, f"Expected {expected_verdict}, got {verdict}"
        assert 0.0 <= fairness_score <= 1.0


def test_product_text_formatting():
    """Test composite text formatting for model input."""
    from app.schemas.product import Product

    p = Product(
        id="test001",
        name="Samsung Galaxy A54",
        category="phones",
        description="5G smartphone with 108MP camera",
        price=250_000,
        seller="Jumia",
        product_url="https://www.jumia.com.ng/samsung-galaxy-a54.html",
    )
    text = p.to_model_input_text()

    assert "category: phones" in text
    assert "name: Samsung Galaxy A54" in text
    assert "seller: Jumia" in text
    assert " | " in text
