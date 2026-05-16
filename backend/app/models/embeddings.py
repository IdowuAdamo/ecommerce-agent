"""
Sentence-Transformer embedding wrapper for semantic search and product indexing.
Uses all-MiniLM-L6-v2 (384 dims) — lightweight, fast, matches Pinecone index dimension.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """Singleton sentence embedding model."""

    _instance: Optional["EmbeddingModel"] = None

    def __init__(self):
        logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        logger.info("Embedding model loaded ✓")

    @classmethod
    def get_instance(cls) -> "EmbeddingModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns list[float] of length 384."""
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embed texts. Returns list of 384-dim vectors."""
        vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=32)
        return vecs.tolist()

    def cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two already-normalized vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        return float(np.dot(a, b))  # normalized → dot product = cosine sim
