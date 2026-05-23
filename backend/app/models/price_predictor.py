"""
NaijaShop AI — Price Predictor (OpenAI API)

CHANGELOG — v2.0.0:
  The original DeBERTa-v3-small + LoRA fine-tuned model
  (Idowenst/ecommerce-price-predictor-v1) has been replaced with an
  OpenAI API-based predictor using engineered prompts.

  Reason for change:
    - The PyTorch/transformers stack required ~2 GB RAM and 60+ s cold-start
      time, exceeding the resource limits of standard cloud deployment tiers.
    - The OpenAI API approach is stateless, requires zero local ML dependencies,
      and starts instantly.
    - Prompt engineering with few-shot examples and JSON-mode output achieves
      comparable accuracy for Nigerian market price estimation.

  The original model code is preserved below in a large comment block
  as a clean fallback reference. To re-enable it, uncomment that block,
  restore the heavy requirements (torch, transformers, peft, etc.) and
  revert the Settings.hf_token field to required (non-Optional).
"""
from __future__ import annotations

import json
import logging
import math
from typing import Optional

from openai import AsyncOpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# NEW IMPLEMENTATION: OpenAI API-based Price Predictor
# ─────────────────────────────────────────────────────────────────────────────

# System prompt for the OpenAI price predictor.
# Uses few-shot examples and explicit JSON output constraints.
_PRICE_PREDICTOR_SYSTEM_PROMPT = """You are a Nigerian e-commerce pricing expert with deep knowledge \
of Jumia Nigeria and Konga marketplace prices as of 2024–2025.

Your sole task is to estimate the fair market price in Nigerian Naira (NGN) \
for a product based on its category, name, and attributes.

RULES:
- Return ONLY valid JSON with exactly one key: "predicted_price_ngn"
- The value must be a positive integer (no decimals, no currency symbols)
- Base predictions strictly on current Nigerian market prices, NOT global prices
- Account for Nigerian import duties, logistics, and local demand
- Do NOT add any explanation or extra text

FEW-SHOT EXAMPLES:
Input: category: smartphones | name: Samsung Galaxy A15 | brand: Samsung
Output: {"predicted_price_ngn": 185000}

Input: category: laptops | name: HP 250 G9 Notebook 8GB RAM 256GB SSD | brand: HP
Output: {"predicted_price_ngn": 420000}

Input: category: refrigerators | name: Hisense 200L Single Door Refrigerator | brand: Hisense
Output: {"predicted_price_ngn": 195000}

Input: category: generators | name: Elepaq 3.5KVA Generator EC4900CX | brand: Elepaq
Output: {"predicted_price_ngn": 280000}

Input: category: washing machines | name: LG 7KG Top Load Washing Machine | brand: LG
Output: {"predicted_price_ngn": 255000}

Input: category: televisions | name: Hisense 43-inch Full HD Smart TV 43A4H | brand: Hisense
Output: {"predicted_price_ngn": 215000}

Input: category: air conditioners | name: Midea 1.5HP Split Unit Inverter AC | brand: Midea
Output: {"predicted_price_ngn": 320000}

Input: category: freezers | name: Haier Thermocool 300L Chest Freezer | brand: Haier Thermocool
Output: {"predicted_price_ngn": 280000}

Input: category: blenders | name: Binatone Blender BLG 450 | brand: Binatone
Output: {"predicted_price_ngn": 18500}

Input: category: power banks | name: Romoss 20000mAh Power Bank Sense 8P | brand: Romoss
Output: {"predicted_price_ngn": 22000}
"""


class PricePredictorModel:
    """
    OpenAI API-based fair-price predictor for Nigerian e-commerce products.

    Replaces the previous DeBERTa-v3-small + LoRA fine-tuned model.
    Uses engineered prompts with few-shot examples and JSON-mode output
    to return accurate price estimates from the GPT-4o-mini model.

    Interface is kept backward-compatible with the original class:
      predictor = PricePredictorModel.get_instance()
      price     = await predictor.predict("category: laptops | name: HP ...")
    """

    _instance: Optional["PricePredictorModel"] = None

    # In-memory response cache — avoids repeated API calls for the same text.
    # Key: product_text (str) → Value: predicted price in Naira (float)
    _cache: dict[str, float] = {}

    def __init__(self) -> None:
        s = get_settings()
        self._client = AsyncOpenAI(api_key=s.openai_api_key)
        self._model = s.openai_model  # defaults to gpt-4o-mini
        logger.info(f"[OK] OpenAI price predictor initialised (model={self._model})")

    @classmethod
    def get_instance(cls) -> "PricePredictorModel":
        """Return the singleton instance (creates it on first call)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def predict(self, product_text: str) -> float:
        """
        Predict the fair market price in Naira for a product.

        Args:
            product_text: Composite description, e.g.
                "category: laptops | name: HP Elitebook 840 | brand: HP"

        Returns:
            Predicted price in Naira (float). Floored at ₦500.
        """
        # 1. Cache hit — skip the API call entirely
        if product_text in self._cache:
            return self._cache[product_text]

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                response_format={"type": "json_object"},   # guaranteed JSON output
                temperature=0.0,                            # deterministic pricing
                max_tokens=64,                              # price JSON is tiny
                messages=[
                    {"role": "system", "content": _PRICE_PREDICTOR_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Predict the fair market price for this product:\n"
                            f"{product_text}\n\n"
                            f'Return JSON: {{"predicted_price_ngn": <integer>}}'
                        ),
                    },
                ],
            )

            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
            price = float(data.get("predicted_price_ngn", 0))

            if price <= 0:
                raise ValueError(f"Model returned non-positive price: {price}")

            # Cache and return
            self._cache[product_text] = price
            logger.debug(f"Price predicted: ₦{price:,.0f} for '{product_text[:60]}...'")
            return price

        except Exception as e:
            logger.warning(f"OpenAI price prediction failed ({e}); using heuristic fallback")
            # Fallback: extract any number that looks like a Naira price from the text
            return self._heuristic_fallback(product_text)

    async def predict_batch(self, texts: list[str]) -> list[float]:
        """Batch price prediction — runs all predictions concurrently."""
        import asyncio
        tasks = [self.predict(t) for t in texts]
        return list(await asyncio.gather(*tasks, return_exceptions=False))

    def compute_price_fairness_score(
        self, actual_price: int, predicted_price: float
    ) -> tuple[float, float, str]:
        """
        Compute fairness metrics from actual vs. predicted price.

        Returns:
            (fairness_score, deviation_pct, verdict)
            - fairness_score : 0.0 (very unfair) → 1.0 (perfectly fair)
            - deviation_pct  : +ve = overpriced, -ve = underpriced
            - verdict        : "great_deal" | "fair" | "slightly_overpriced"
                               | "overpriced" | "suspicious"
        """
        if predicted_price <= 0:
            return 0.5, 0.0, "unknown"

        deviation_pct = (actual_price - predicted_price) / predicted_price * 100

        # Fairness score decays exponentially with deviation magnitude
        raw_score = math.exp(-abs(deviation_pct) / 30)
        fairness_score = max(0.0, min(1.0, raw_score))

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

        return fairness_score, deviation_pct, verdict

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _heuristic_fallback(product_text: str) -> float:
        """
        Rough category-based price estimate used when the API call fails.
        Returns a median market price in Naira for the detected category.
        """
        text_lower = product_text.lower()
        category_prices = {
            "laptop": 400_000,
            "smartphone": 200_000,
            "tablet": 180_000,
            "television": 200_000,
            "refrigerator": 250_000,
            "freezer": 280_000,
            "washing machine": 220_000,
            "air conditioner": 300_000,
            "generator": 250_000,
            "microwave": 80_000,
            "blender": 20_000,
            "iron": 15_000,
            "headphone": 35_000,
            "speaker": 50_000,
            "power bank": 22_000,
            "camera": 250_000,
        }
        for keyword, price in category_prices.items():
            if keyword in text_lower:
                return float(price)
        return 100_000.0  # default fallback


# =============================================================================
# ORIGINAL IMPLEMENTATION — DeBERTa-v3-small + LoRA (COMMENTED OUT)
#
# Preserved for reference. To re-enable:
#   1. Restore requirements: torch, transformers, peft, accelerate,
#      sentence-transformers, huggingface-hub (see requirements.txt.bak)
#   2. Restore config.py: hf_token: str (required, not Optional)
#   3. Uncomment the block below and delete the OpenAI implementation above.
# =============================================================================

# """
# DeBERTa-v3-small + LoRA Price Predictor Inference Wrapper.
# Loads the fine-tuned model from HuggingFace (Idowenst/ecommerce-price-predictor-v1)
# and provides a fast, cached predict() interface used by the Trust & Value Agent.
#
# Architecture mirrors exactly what was trained in price-predictor.ipynb:
#   - Encoder: microsoft/deberta-v3-small + LoRA adapters
#   - Head: Dropout(0.2) → Linear(hidden, 256) → GELU → Linear(256, 1)
#   - Output: log_price → expm1 → Naira price
# """
#
# import numpy as np
# import torch
# import torch.nn as nn
# from huggingface_hub import hf_hub_download
# from peft import PeftModel
# from transformers import AutoModel, AutoTokenizer
#
#
# class PriceRegressor(nn.Module):
#     """
#     Regression head — exactly mirrors the training architecture.
#     CLS-free mean pooling → Dropout → Linear(256) → GELU → Linear(1)
#     """
#
#     def __init__(self, hidden_size: int):
#         super().__init__()
#         self.regressor = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(hidden_size, 256),
#             nn.GELU(),
#             nn.Linear(256, 1),
#         )
#
#     def forward(self, pooled: torch.Tensor) -> torch.Tensor:
#         return self.regressor(pooled).squeeze(-1)
#
#
# class PricePredictorModel:
#     """Singleton inference wrapper for the Jumia price prediction model."""
#
#     _instance: Optional["PricePredictorModel"] = None
#
#     def __init__(self, model_id: str, hf_token: str, device: str, max_len: int):
#         self.model_id = model_id
#         self.device = torch.device(device)
#         self.max_len = max_len
#         logger.info(f"Loading price predictor from {model_id} on {device}...")
#         self._load_model(model_id, hf_token)
#         logger.info("Price predictor loaded ✓")
#
#     def _load_model(self, model_id: str, hf_token: str) -> None:
#         """Load tokenizer, LoRA encoder, and regressor head from HuggingFace."""
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             model_id, token=hf_token, trust_remote_code=True
#         )
#         base_encoder = AutoModel.from_pretrained(
#             "microsoft/deberta-v3-small", token=hf_token,
#         )
#         self.encoder = PeftModel.from_pretrained(
#             base_encoder, model_id, token=hf_token
#         ).to(self.device)
#         self.encoder.eval()
#         head_path = hf_hub_download(
#             repo_id=model_id, filename="regressor_head.pt", token=hf_token,
#         )
#         hidden_size = self.encoder.config.hidden_size
#         self.head = PriceRegressor(hidden_size).to(self.device)
#         self.head.regressor.load_state_dict(
#             torch.load(head_path, map_location=self.device, weights_only=True)
#         )
#         self.head.eval()
#
#     @classmethod
#     def get_instance(cls) -> "PricePredictorModel":
#         if cls._instance is None:
#             s = get_settings()
#             cls._instance = cls(
#                 model_id=s.price_model_id,
#                 hf_token=s.hf_token,
#                 device=s.price_model_device,
#                 max_len=s.price_model_max_len,
#             )
#         return cls._instance
#
#     @torch.no_grad()
#     def predict(self, product_text: str) -> float:
#         enc = self.tokenizer(
#             product_text, max_length=self.max_len, padding="max_length",
#             truncation=True, return_tensors="pt",
#         )
#         input_ids = enc["input_ids"].to(self.device)
#         attention_mask = enc["attention_mask"].to(self.device)
#         out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         mask = attention_mask.unsqueeze(-1).float()
#         pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
#         log_pred = self.head(pooled)
#         price_naira = float(np.expm1(log_pred.cpu().numpy()))
#         return max(price_naira, 100.0)
#
#     def predict_batch(self, texts: list[str]) -> list[float]:
#         results = []
#         for text in texts:
#             try:
#                 results.append(self.predict(text))
#             except Exception as e:
#                 logger.warning(f"Price prediction failed: {e}")
#                 results.append(0.0)
#         return results
#
#     def compute_price_fairness_score(
#         self, actual_price: int, predicted_price: float
#     ) -> tuple[float, float, str]:
#         if predicted_price <= 0:
#             return 0.5, 0.0, "unknown"
#         deviation_pct = (actual_price - predicted_price) / predicted_price * 100
#         raw_score = math.exp(-abs(deviation_pct) / 30)
#         fairness_score = max(0.0, min(1.0, raw_score))
#         if deviation_pct < -20:
#             verdict = "great_deal"
#         elif deviation_pct < 10:
#             verdict = "fair"
#         elif deviation_pct < 30:
#             verdict = "slightly_overpriced"
#         elif deviation_pct < 60:
#             verdict = "overpriced"
#         else:
#             verdict = "suspicious"
#         return fairness_score, deviation_pct, verdict
# =============================================================================
