"""
DeBERTa-v3-small + LoRA Price Predictor Inference Wrapper.

Loads the fine-tuned model from HuggingFace (Idowenst/ecommerce-price-predictor-v1)
and provides a fast, cached predict() interface used by the Trust & Value Agent.

Architecture mirrors exactly what was trained in price-predictor.ipynb:
  - Encoder: microsoft/deberta-v3-small + LoRA adapters
  - Head: Dropout(0.2) → Linear(hidden, 256) → GELU → Linear(256, 1)
  - Output: log_price → expm1 → Naira price
"""
from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

from app.config import get_settings

logger = logging.getLogger(__name__)


class PriceRegressor(nn.Module):
    """
    Regression head — exactly mirrors the training architecture.
    CLS-free mean pooling → Dropout → Linear(256) → GELU → Linear(1)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.regressor(pooled).squeeze(-1)


class PricePredictorModel:
    """
    Singleton inference wrapper for the Jumia price prediction model.

    Usage:
        predictor = PricePredictorModel.get_instance()
        price_naira = predictor.predict("category: laptops | name: HP Elitebook ...")
    """

    _instance: Optional["PricePredictorModel"] = None

    def __init__(self, model_id: str, hf_token: str, device: str, max_len: int):
        self.model_id = model_id
        self.device = torch.device(device)
        self.max_len = max_len

        logger.info(f"Loading price predictor from {model_id} on {device}...")
        self._load_model(model_id, hf_token)
        logger.info("Price predictor loaded ✓")

    def _load_model(self, model_id: str, hf_token: str) -> None:
        """Load tokenizer, LoRA encoder, and regressor head from HuggingFace."""
        # 1. Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=hf_token, trust_remote_code=True
        )

        # 2. Load base encoder + LoRA adapters
        base_encoder = AutoModel.from_pretrained(
            "microsoft/deberta-v3-small",
            token=hf_token,
        )
        self.encoder = PeftModel.from_pretrained(
            base_encoder, model_id, token=hf_token
        ).to(self.device)
        self.encoder.eval()

        # 3. Load regressor head weights
        head_path = hf_hub_download(
            repo_id=model_id,
            filename="regressor_head.pt",
            token=hf_token,
        )
        hidden_size = self.encoder.config.hidden_size
        self.head = PriceRegressor(hidden_size).to(self.device)
        self.head.regressor.load_state_dict(
            torch.load(head_path, map_location=self.device, weights_only=True)
        )
        self.head.eval()

    @classmethod
    def get_instance(cls) -> "PricePredictorModel":
        if cls._instance is None:
            s = get_settings()
            cls._instance = cls(
                model_id=s.price_model_id,
                hf_token=s.hf_token,
                device=s.price_model_device,
                max_len=s.price_model_max_len,
            )
        return cls._instance

    @torch.no_grad()
    def predict(self, product_text: str) -> float:
        """
        Predict fair market price in Naira for a product described by text.

        Args:
            product_text: Composite text in the format used during training:
                "category: {cat} | name: {name} | description: {desc} | ..."

        Returns:
            Predicted price in Naira (float).
        """
        enc = self.tokenizer(
            product_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # Mean pooling (matches training)
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        log_pred = self.head(pooled)
        price_naira = float(np.expm1(log_pred.cpu().numpy()))
        return max(price_naira, 100.0)  # floor at ₦100

    def predict_batch(self, texts: list[str]) -> list[float]:
        """Batch prediction for multiple products."""
        results = []
        for text in texts:
            try:
                results.append(self.predict(text))
            except Exception as e:
                logger.warning(f"Price prediction failed for text: {e}")
                results.append(0.0)
        return results

    def compute_price_fairness_score(
        self, actual_price: int, predicted_price: float
    ) -> tuple[float, float, str]:
        """
        Compute fairness metrics.

        Returns:
            (fairness_score, deviation_pct, verdict)
            - fairness_score: 0.0 (very unfair) to 1.0 (perfectly fair)
            - deviation_pct: +ve = overpriced, -ve = underpriced
            - verdict: "great_deal" | "fair" | "slightly_overpriced" | "overpriced" | "suspicious"
        """
        if predicted_price <= 0:
            return 0.5, 0.0, "unknown"

        deviation_pct = (actual_price - predicted_price) / predicted_price * 100

        # Fairness score: 1.0 when price matches predicted, decays with deviation
        raw_score = math.exp(-abs(deviation_pct) / 30)
        fairness_score = max(0.0, min(1.0, raw_score))

        # Verdict thresholds
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
