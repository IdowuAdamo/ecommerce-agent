"""
PostgreSQL / Supabase user store for profile persistence.
Uses asyncpg for async database operations.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Optional

import asyncpg

from app.config import get_settings
from app.schemas.user import UserProfile, NigerianPersonaType, NigerianLocation

logger = logging.getLogger(__name__)


class UserStoreService:
    """Manages user profiles and interaction history in PostgreSQL."""

    _pool: Optional[asyncpg.Pool] = None
    _memory_store: dict[str, UserProfile] = {}

    @classmethod
    async def get_pool(cls) -> asyncpg.Pool:
        if cls._pool is None:
            s = get_settings()
            db_url = s.database_url or _supabase_to_asyncpg(s.supabase_url)
            cls._pool = await asyncpg.create_pool(db_url, min_size=2, max_size=10)
            await cls._ensure_tables(cls._pool)
            logger.info("PostgreSQL pool ready ✓")
        return cls._pool

    @classmethod
    async def _ensure_tables(cls, pool: asyncpg.Pool) -> None:
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    session_id VARCHAR(255) UNIQUE NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS user_profiles (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                    session_id VARCHAR(255) UNIQUE NOT NULL,
                    budget_min BIGINT,
                    budget_max BIGINT,
                    preferred_categories TEXT[],
                    location VARCHAR(100) DEFAULT 'Unknown',
                    persona_type VARCHAR(50) DEFAULT 'unknown',
                    price_sensitivity FLOAT DEFAULT 0.5,
                    brand_affinity JSONB DEFAULT '{}',
                    shopping_intent VARCHAR(50) DEFAULT 'browsing',
                    cold_start BOOLEAN DEFAULT TRUE,
                    interaction_count INTEGER DEFAULT 0,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS interaction_history (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    session_id VARCHAR(255) NOT NULL,
                    product_id VARCHAR(255),
                    interaction_type VARCHAR(50),
                    context JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS recommendation_logs (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    session_id VARCHAR(255),
                    query TEXT,
                    ranked_products JSONB,
                    ndcg_10 FLOAT,
                    hit_rate FLOAT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)

    async def get_or_create_profile(self, session_id: str) -> UserProfile:
        try:
            pool = await self.get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM user_profiles WHERE session_id = $1", session_id
                )
                if row:
                    return _row_to_profile(row)

                # New session — create user + profile
                user_id = str(uuid.uuid4())
                await conn.execute(
                    "INSERT INTO users (id, session_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                    user_id, session_id,
                )
                await conn.execute(
                    """INSERT INTO user_profiles
                       (user_id, session_id, preferred_categories, brand_affinity)
                       VALUES ($1, $2, $3, $4)
                       ON CONFLICT (session_id) DO NOTHING""",
                    user_id, session_id, [], json.dumps({}),
                )
                row = await conn.fetchrow(
                    "SELECT * FROM user_profiles WHERE session_id = $1", session_id
                )
                return _row_to_profile(row)

        except Exception as e:
            logger.warning(f"DB fallback for get_or_create_profile: {e}")
            if session_id not in self._memory_store:
                self._memory_store[session_id] = UserProfile(
                    user_id=str(uuid.uuid4()),
                    session_id=session_id,
                    persona_type=NigerianPersonaType.UNKNOWN,
                )
            return self._memory_store[session_id]

    async def update_profile(self, session_id: str, updates: dict) -> None:
        try:
            pool = await self.get_pool()
        except Exception as e:
            if session_id in self._memory_store:
                p = self._memory_store[session_id]
                for k, v in updates.items():
                    if hasattr(p, k):
                        setattr(p, k, v)
            return
        allowed = {
            "budget_min", "budget_max", "preferred_categories", "location",
            "persona_type", "price_sensitivity", "brand_affinity",
            "shopping_intent", "cold_start", "interaction_count",
        }
        filtered = {k: v for k, v in updates.items() if k in allowed}
        if not filtered:
            return

        set_clause = ", ".join(
            f"{k} = ${i+2}" for i, k in enumerate(filtered)
        )
        values = [session_id] + [
            json.dumps(v) if isinstance(v, dict) else v
            for v in filtered.values()
        ]
        async with pool.acquire() as conn:
            await conn.execute(
                f"UPDATE user_profiles SET {set_clause}, updated_at = NOW() "
                f"WHERE session_id = $1",
                *values,
            )

    async def log_interaction(
        self, session_id: str, product_id: str, interaction_type: str, context: dict
    ) -> None:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO interaction_history
                   (session_id, product_id, interaction_type, context)
                   VALUES ($1, $2, $3, $4)""",
                session_id, product_id, interaction_type, json.dumps(context),
            )

    async def get_interaction_history(self, session_id: str, limit: int = 50) -> list[dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM interaction_history WHERE session_id = $1 "
                "ORDER BY created_at DESC LIMIT $2",
                session_id, limit,
            )
            return [dict(r) for r in rows]


def _supabase_to_asyncpg(supabase_url: str) -> str:
    """Convert Supabase project URL to asyncpg connection string."""
    url = supabase_url.rstrip("/").strip()
    # Extract project ref from URL like https://xxxxx.supabase.co
    import re
    m = re.match(r"https://([^.]+)\.supabase\.co", url)
    if m:
        ref = m.group(1)
        return f"postgresql://postgres.{ref}@aws-0-us-east-1.pooler.supabase.com:5432/postgres"
    return url  # assume it's already a postgres URL


def _row_to_profile(row) -> UserProfile:
    return UserProfile(
        user_id=str(row["user_id"]),
        session_id=row["session_id"],
        budget_min=row["budget_min"],
        budget_max=row["budget_max"],
        preferred_categories=list(row["preferred_categories"] or []),
        location=NigerianLocation(row["location"] or "Unknown"),
        persona_type=NigerianPersonaType(row["persona_type"] or "unknown"),
        price_sensitivity=float(row["price_sensitivity"] or 0.5),
        brand_affinity=dict(row["brand_affinity"] or {}),
        shopping_intent=row["shopping_intent"] or "browsing",
        cold_start=bool(row["cold_start"]),
        interaction_count=int(row["interaction_count"] or 0),
    )
