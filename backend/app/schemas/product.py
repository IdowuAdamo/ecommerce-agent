"""
Pydantic schemas for products, trust scores, and price fairness.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, HttpUrl


class PriceFairness(BaseModel):
    """Output of the Trust & Value Agent (DeBERTa price predictor)."""
    actual_price: int                          # in Naira
    predicted_fair_price: int                  # from DeBERTa model
    price_deviation_pct: float                 # (actual - predicted) / predicted * 100
    verdict: str                               # "fair" | "overpriced" | "great_deal" | "suspicious"
    fairness_score: float = Field(ge=0.0, le=1.0)  # 1.0 = perfectly fair
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = ""

    @property
    def is_overpriced(self) -> bool:
        return self.price_deviation_pct > 20

    @property
    def is_great_deal(self) -> bool:
        return self.price_deviation_pct < -15


class TrustScore(BaseModel):
    """Composite trust score for a product listing."""
    overall: float = Field(ge=0.0, le=1.0)
    seller_score: float = Field(ge=0.0, le=1.0)
    rating_authenticity: float = Field(ge=0.0, le=1.0)
    discount_legitimacy: float = Field(ge=0.0, le=1.0)
    review_count_score: float = Field(ge=0.0, le=1.0)
    stock_reliability: float = Field(ge=0.0, le=1.0)
    flags: list[str] = Field(default_factory=list)  # ["fake_discount", "suspicious_seller"]


class Product(BaseModel):
    """Normalized product schema from any source (Jumia, Konga, mock)."""
    id: str
    external_id: Optional[str] = None
    source: str = "jumia"                      # jumia | konga | mock
    name: str
    category: str
    description: str = ""
    price: int                                 # in Naira
    old_price: Optional[int] = None
    discount_pct: Optional[float] = None
    rating: Optional[float] = None
    num_reviews: Optional[int] = None
    seller: Optional[str] = None
    product_url: Optional[str] = None
    image_url: Optional[str] = None
    stock_info: str = "In stock"
    brand: Optional[str] = None
    tags: list[str] = Field(default_factory=list)

    # Computed fields (filled by Trust & Value Agent)
    price_fairness: Optional[PriceFairness] = None
    trust_score: Optional[TrustScore] = None

    def to_model_input_text(self) -> str:
        """Format product data as composite text for DeBERTa price predictor."""
        parts = [
            f"category: {self.category}",
            f"name: {self.name}",
        ]
        if self.description:
            parts.append(f"description: {self.description[:300]}")
        if self.product_url:
            # Clean URL for model input (as in training)
            clean_url = self.product_url.replace("https://", "").replace("http://", "")
            parts.append(f"product_url: {clean_url}")
        if self.seller:
            parts.append(f"seller: {self.seller}")
        return " | ".join(parts)


class RankedProduct(BaseModel):
    """A product with its recommendation ranking metadata."""
    product: Product
    rank: int
    composite_score: float = Field(ge=0.0, le=1.0)
    semantic_score: float = 0.0
    behavioral_score: float = 0.0
    price_fairness_score: float = 0.0
    trust_score_val: float = 0.0
    contextual_score: float = 0.0
    explanation: Optional[str] = None
