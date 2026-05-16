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
    cors_origins: list[str] = ["http://localhost:3000", "https://naijashop.vercel.app"]

    # ── OpenAI ───────────────────────────────────────────────────────────────
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"          # cost-efficient for hackathon
    openai_embedding_model: str = "text-embedding-3-small"

    # ── HuggingFace (Price Predictor) ─────────────────────────────────────────
    hf_token: str
    price_model_id: str = "Idowenst/ecommerce-price-predictor-v1"
    price_model_max_len: int = 256
    price_model_device: str = "cpu"            # set to "cuda" if GPU available

    # ── Supabase / PostgreSQL ────────────────────────────────────────────────
    supabase_url: str
    supabase_key: Optional[str] = None         # service role key if needed
    database_url: Optional[str] = None         # overrides supabase if set

    # ── Pinecone ─────────────────────────────────────────────────────────────
    pinecone_api_key: str
    pinecone_index_name: str = "naijashop-products"
    pinecone_dimension: int = 384              # all-MiniLM-L6-v2

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
    rec_semantic_weight: float = 0.25
    rec_behavioral_weight: float = 0.20
    rec_price_fairness_weight: float = 0.25
    rec_trust_weight: float = 0.20
    rec_contextual_weight: float = 0.10
    rec_top_k: int = 10
    rec_diversity_lambda: float = 0.5          # MMR diversity parameter

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "naijashop-ai"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
