"""
NaijaShop AI — Embedding Model (OpenAI API)

CHANGELOG — v2.0.0:
  The original sentence-transformers/all-MiniLM-L6-v2 local model has been
  replaced with the OpenAI text-embedding-3-small API.

  Reason for change:
    - sentence-transformers transitively requires PyTorch (~700 MB), which
      exceeded cloud deployment memory limits.
    - OpenAI text-embedding-3-small supports a `dimensions` parameter that
      allows output to be reduced to 384 dims — exactly matching the existing
      Pinecone index. No index migration required.
    - The API-based approach is stateless and starts instantly.

  The original sentence-transformers code is preserved below in a comment
  block. To re-enable it, restore the sentence-transformers dependency and
  uncomment that section.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from openai import AsyncOpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

# Pinecone index was created with 384 dims (all-MiniLM-L6-v2).
# OpenAI text-embedding-3-small natively supports dimension reduction,
# so we keep 384 dims for full backward compatibility.
EMBED_DIM = 384


class EmbeddingModel:
    """
    OpenAI text-embedding-3-small wrapper (replaces sentence-transformers).

    Outputs 384-dimensional vectors — identical shape to the previous
    all-MiniLM-L6-v2 model, so the Pinecone index requires no changes.

    Usage (unchanged from v1):
        embedder = EmbeddingModel.get_instance()
        vec  = await embedder.embed("some text")
        vecs = await embedder.embed_batch(["text1", "text2"])
    """

    _instance: Optional["EmbeddingModel"] = None

    def __init__(self) -> None:
        s = get_settings()
        self._client = AsyncOpenAI(api_key=s.openai_api_key)
        self._model = s.openai_embedding_model   # "text-embedding-3-small"
        self._dims = EMBED_DIM
        logger.info(
            f"[OK] OpenAI embedding model initialised "
            f"(model={self._model}, dims={self._dims})"
        )

    @classmethod
    def get_instance(cls) -> "EmbeddingModel":
        """Return the singleton instance (creates it on first call)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def embed(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Returns:
            list[float] of length 384 (L2-normalised).
        """
        text = text.strip().replace("\n", " ") or "."
        response = await self._client.embeddings.create(
            input=text,
            model=self._model,
            dimensions=self._dims,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Batch embed a list of texts in a single API call (most efficient).

        Returns:
            list of 384-dim vectors, in the same order as `texts`.
        """
        cleaned = [t.strip().replace("\n", " ") or "." for t in texts]
        response = await self._client.embeddings.create(
            input=cleaned,
            model=self._model,
            dimensions=self._dims,
        )
        # API guarantees items are returned in index order, but sort explicitly
        items = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in items]

    def cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """
        Cosine similarity between two vectors.
        OpenAI embeddings are L2-normalised, so this equals the dot product.
        """
        a = np.array(vec_a)
        b = np.array(vec_b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)


# =============================================================================
# ORIGINAL IMPLEMENTATION — sentence-transformers/all-MiniLM-L6-v2 (COMMENTED OUT)
#
# Preserved for reference. To re-enable:
#   1. Restore requirements: sentence-transformers (and transitively: torch)
#   2. Uncomment the block below and delete the OpenAI implementation above.
#   3. Revert embed() / embed_batch() callers to sync (remove await).
# =============================================================================

# """
# Sentence-Transformer embedding wrapper for semantic search and product indexing.
# Uses all-MiniLM-L6-v2 (384 dims) — lightweight, fast, matches Pinecone index dimension.
# """
#
# import numpy as np
# from sentence_transformers import SentenceTransformer
#
# EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
#
#
# class EmbeddingModel:
#     """Singleton sentence embedding model."""
#
#     _instance: Optional["EmbeddingModel"] = None
#
#     def __init__(self):
#         logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
#         self.model = SentenceTransformer(EMBED_MODEL_NAME)
#         logger.info("Embedding model loaded ✓")
#
#     @classmethod
#     def get_instance(cls) -> "EmbeddingModel":
#         if cls._instance is None:
#             cls._instance = cls()
#         return cls._instance
#
#     def embed(self, text: str) -> list[float]:
#         """Embed a single text string. Returns list[float] of length 384."""
#         vec = self.model.encode(text, normalize_embeddings=True)
#         return vec.tolist()
#
#     def embed_batch(self, texts: list[str]) -> list[list[float]]:
#         """Batch embed texts. Returns list of 384-dim vectors."""
#         vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=32)
#         return vecs.tolist()
#
#     def cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
#         """Compute cosine similarity between two already-normalized vectors."""
#         a = np.array(vec_a)
#         b = np.array(vec_b)
#         return float(np.dot(a, b))  # normalized → dot product = cosine sim
# =============================================================================
