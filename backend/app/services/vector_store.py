"""
Vector store service using Pinecone for semantic product search.
Index dimension: 384 (all-MiniLM-L6-v2).
"""
from __future__ import annotations

import logging
from typing import Optional

from pinecone import Pinecone, ServerlessSpec

from app.config import get_settings
from app.schemas.product import Product

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Manages Pinecone index for semantic product search."""

    _instance: Optional["VectorStoreService"] = None

    def __init__(self):
        s = get_settings()
        self.pc = Pinecone(api_key=s.pinecone_api_key)
        self.index_name = s.pinecone_index_name
        self.dimension = s.pinecone_dimension
        self._ensure_index()
        self.index = self.pc.Index(self.index_name)
        logger.info(f"Pinecone index '{self.index_name}' ready ✓")

    @classmethod
    def get_instance(cls) -> "VectorStoreService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_index(self) -> None:
        existing = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing:
            logger.info(f"Creating Pinecone index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

    def upsert_products(self, products: list[Product], embeddings: list[list[float]]) -> None:
        """Upsert product embeddings into Pinecone."""
        vectors = []
        for product, embedding in zip(products, embeddings):
            vectors.append({
                "id": product.id,
                "values": embedding,
                "metadata": {
                    "name": product.name,
                    "category": product.category,
                    "price": product.price,
                    "rating": product.rating or 0,
                    "source": product.source,
                    "brand": product.brand or "",
                    "discount_pct": product.discount_pct or 0,
                },
            })
        if vectors:
            self.index.upsert(vectors=vectors)
            logger.info(f"Upserted {len(vectors)} products to Pinecone")

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        category_filter: Optional[str] = None,
        price_max: Optional[int] = None,
    ) -> list[dict]:
        """Semantic search with optional metadata filters."""
        filter_dict: dict = {}
        if category_filter:
            filter_dict["category"] = {"$eq": category_filter}
        if price_max:
            filter_dict["price"] = {"$lte": price_max}

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None,
        )
        return results.get("matches", [])

    def delete_all(self) -> None:
        """Clear all vectors from the index (for testing)."""
        self.index.delete(delete_all=True)
