"""
NaijaShop AI — Central Configuration
All settings are loaded from environment variables via Pydantic BaseSettings.
"""
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────────
    app_name: str = "NaijaShop AI"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    cors_origins: list[str] = [
        "http://localhost:3000",
        "https://naijashop.vercel.app",
    ]
    # Set this to your exact Vercel URL in the Render env vars dashboard
    # e.g. CORS_ORIGIN=https://naijashop-ai.vercel.app
    # If not set, CORS falls back to allow_origins=["*"] (open for hackathon)
    cors_origin: Optional[str] = None

    # ── OpenAI (Primary LLM Provider) ────────────────────────────────────────
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"          # cost-efficient for hackathon
    openai_embedding_model: str = "text-embedding-3-small"

    # ── Gemini (Fallback LLM Provider) ───────────────────────────────────────
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-1.5-flash"     # fast, cost-efficient fallback

    # ── LLM Provider Configuration ────────────────────────────────────────────
    llm_provider_priority: list[str] = ["openai", "gemini"]
    llm_request_timeout: int = 25              # seconds per provider attempt
    llm_max_retries: int = 1                   # 1 retry = 2 total attempts before giving up

    # ── HuggingFace (legacy — only needed if reverting to DeBERTa model) ───────
    # hf_token is now Optional: the OpenAI-based price predictor does not
    # require HuggingFace access. Set it only if re-enabling the DeBERTa model.
    hf_token: Optional[str] = None
    price_model_id: str = "Idowenst/ecommerce-price-predictor-v1"   # legacy ref
    price_model_max_len: int = 256                                    # legacy ref
    price_model_device: str = "cpu"                                   # legacy ref

    # ── Supabase / PostgreSQL ────────────────────────────────────────────────
    supabase_url: str
    supabase_key: Optional[str] = None         # service role key if needed
    database_url: Optional[str] = None         # overrides supabase if set

    # ── Pinecone ─────────────────────────────────────────────────────────────
    pinecone_api_key: str
    pinecone_index_name: str = "naijashop-products"
    pinecone_dimension: int = 384              # text-embedding-3-small @ 384 dims

    # ── Redis (optional — falls back to in-memory) ───────────────────────────
    redis_url: Optional[str] = None

    # ── Scraper ──────────────────────────────────────────────────────────────
    scraper_delay_seconds: float = 2.0         # polite delay between requests
    scraper_max_retries: int = 3
    scraper_user_agent: str = (
        "ClaudeBot/1.0 (NaijaShop AI Research; +https://github.com/naijashop)"
    )
    scraper_timeout_seconds: int = 15

    # ── Recommendation ───────────────────────────────────────────────────────
    rec_semantic_weight: float = 0.20
    rec_behavioral_weight: float = 0.15
    rec_price_fairness_weight: float = 0.20
    rec_trust_weight: float = 0.15
    rec_contextual_weight: float = 0.10
    rec_budget_proximity_weight: float = 0.20
    rec_top_k: int = 10
    rec_diversity_lambda: float = 0.5          # MMR diversity parameter


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
