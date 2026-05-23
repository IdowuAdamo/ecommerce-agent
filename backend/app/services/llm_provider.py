"""
NaijaShop AI — Unified LLM Provider Service.

Provides a centralized abstraction over multiple LLM providers.
OpenAI is the primary provider; Gemini is the automatic fallback.

Failover triggers on:
  - openai.RateLimitError
  - openai.APITimeoutError
  - openai.APIConnectionError
  - openai.AuthenticationError
  - Any unexpected exception after max retries

Usage:
    provider = LLMProvider.get_instance()
    response = await provider.chat(messages=[...], temperature=0.3)
    print(response.content)   # normalized string content
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""
    content: str
    provider: str          # "openai" | "gemini"
    model: str
    was_fallback: bool = False


class LLMProvider:
    """
    Singleton LLM provider manager.

    Attempts OpenAI first. On rate-limit, timeout, or connection failure,
    automatically retries with Gemini (if configured).
    All provider switches are logged with structlog for observability.
    """

    _instance: Optional["LLMProvider"] = None

    def __init__(self) -> None:
        from app.config import get_settings
        s = get_settings()

        self._timeout = s.llm_request_timeout
        self._max_retries = s.llm_max_retries
        self._priority = s.llm_provider_priority

        # ── OpenAI client ─────────────────────────────────────────────────────
        self._openai_model = s.openai_model
        try:
            from openai import AsyncOpenAI
            self._openai = AsyncOpenAI(
                api_key=s.openai_api_key,
                timeout=float(self._timeout),
                max_retries=0,  # We handle retries ourselves
            )
            logger.info("[LLM] OpenAI provider initialized", model=self._openai_model)
        except Exception as e:
            logger.warning("[LLM] OpenAI provider init failed", error=str(e))
            self._openai = None

        # ── Gemini client ─────────────────────────────────────────────────────
        self._gemini_model = s.gemini_model
        self._gemini = None
        if s.gemini_api_key:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    import google.generativeai as genai
                    genai.configure(api_key=s.gemini_api_key)
                    self._gemini = genai.GenerativeModel(self._gemini_model)
                logger.info("[LLM] Gemini fallback provider initialized", model=self._gemini_model)
            except Exception as e:
                logger.warning("[LLM] Gemini provider init failed", error=str(e))
        else:
            logger.warning("[LLM] GEMINI_API_KEY not set — fallback provider disabled")

    @classmethod
    def get_instance(cls) -> "LLMProvider":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 512,
        response_format: Optional[dict] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Execute a chat completion with automatic provider failover.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            response_format: e.g. {"type": "json_object"} for structured output.

        Returns:
            LLMResponse with normalized content string and provider metadata.

        Raises:
            RuntimeError: If all providers fail.
        """
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            # Try OpenAI
            if self._openai and "openai" in self._priority:
                try:
                    return await self._call_openai(
                        messages, temperature, max_tokens, response_format
                    )
                except Exception as e:
                    if self._is_fatal_openai_error(e):
                        raise  # Don't retry fatal errors (bad request, etc.)
                    last_error = e
                    logger.warning(
                        "[LLM] OpenAI request failed — attempting Gemini fallback",
                        attempt=attempt + 1,
                        error=type(e).__name__,
                        detail=str(e)[:120],
                    )

            # Try Gemini fallback
            if self._gemini and "gemini" in self._priority:
                try:
                    result = await self._call_gemini(messages, temperature, max_tokens)
                    result.was_fallback = True
                    logger.info(
                        "[LLM] Gemini fallback successful",
                        attempt=attempt + 1,
                    )
                    return result
                except Exception as e:
                    last_error = e
                    logger.error(
                        "[LLM] Gemini fallback also failed",
                        attempt=attempt + 1,
                        error=type(e).__name__,
                        detail=str(e)[:120],
                    )

            # Back-off between retries (not after last attempt)
            if attempt < self._max_retries:
                await asyncio.sleep(2 ** attempt)  # 1s, 2s, 4s...

        raise RuntimeError(
            f"All LLM providers exhausted after {self._max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )

    # ── Private: OpenAI ───────────────────────────────────────────────────────

    async def _call_openai(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        response_format: Optional[dict],
    ) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self._openai_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response_format:
            kwargs["response_format"] = response_format

        resp = await self._openai.chat.completions.create(**kwargs)
        return LLMResponse(
            content=resp.choices[0].message.content,
            provider="openai",
            model=self._openai_model,
        )

    @staticmethod
    def _is_fatal_openai_error(e: Exception) -> bool:
        """Return True for errors that should NOT trigger fallback (client mistakes)."""
        try:
            from openai import BadRequestError, UnprocessableEntityError
            return isinstance(e, (BadRequestError, UnprocessableEntityError))
        except ImportError:
            return False

    # ── Private: Gemini ───────────────────────────────────────────────────────

    async def _call_gemini(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """
        Call Gemini. Converts OpenAI message format to Gemini format.
        System messages are prepended to the first user turn.
        Enforces a hard timeout equal to llm_request_timeout.
        """
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            import google.generativeai as genai

        # Merge system prompt into first user message (Gemini API style)
        system_parts: list[str] = []
        user_parts: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                user_parts.append(msg)

        gemini_history = []
        for i, msg in enumerate(user_parts):
            role = "user" if msg["role"] == "user" else "model"
            content = msg["content"]
            # Prepend system context to the first user message
            if i == 0 and system_parts:
                content = "\n\n".join(system_parts) + "\n\n---\n\n" + content
            gemini_history.append({"role": role, "parts": [content]})

        # Extract the final user turn as the actual prompt
        final_prompt = gemini_history[-1]["parts"][0] if gemini_history else ""
        history = gemini_history[:-1]

        chat = self._gemini.start_chat(history=history)
        config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        # Enforce hard timeout so Gemini DNS/network failures don't hang the pipeline
        response = await asyncio.wait_for(
            asyncio.to_thread(chat.send_message, final_prompt, generation_config=config),
            timeout=float(self._timeout),
        )
        return LLMResponse(
            content=response.text,
            provider="gemini",
            model=self._gemini_model,
        )
