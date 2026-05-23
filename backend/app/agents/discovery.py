"""
Agent 2: Conversational Discovery Agent.

Extracts structured shopping intent from natural language queries.
Handles Nigerian English, pidgin, slang, and cultural context.
Uses the centralized LLMProvider (OpenAI primary, Gemini fallback).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.config import get_settings
from app.schemas.agent import AgentState
from app.services.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

DISCOVERY_SYSTEM_PROMPT = """You are a Nigerian e-commerce shopping assistant that understands Nigerian English, Pidgin, and local shopping culture.

Extract structured shopping intent from the user's message. Understand Nigerian context:
- "NYSC" = National Youth Service Corps (young person, tight budget ~₦50k-200k)
- "January blues" = post-holiday financial squeeze
- "Detty December" = festive season, higher budget
- "Abuja salary" = government worker, stable income
- Budget in "k" = thousands (₦400k = ₦400,000)
- "Am I doing well" / "E go dey alright" = positive confirmation
- High performance use-cases (e.g. "gaming", "machine learning", "video editing") IMPLY a high minimum budget (e.g. at least ₦400k-₦600k minimum for laptops). DO NOT infer a low budget for these!
- Detect commercial/business usage signals: "business", "shop", "supermarket", "restaurant", "store", "office", "commercial", "industrial"

Return a JSON object with these fields:
{
  "category": string or null,           // product category. Be SPECIFIC: e.g. 'freezer', 'chest freezer', 'smartphone', not just 'appliance'
  "use_case": string or null,           // what they want to do with it
  "budget_min": number or null,         // minimum budget in Naira (infer reasonably high if use_case demands it!)
  "budget_max": number or null,         // maximum budget in Naira
  "location": string or null,           // Lagos, Abuja, PH, etc.
  "urgency": "low" | "medium" | "high", // how urgently they need it
  "brands": list[string],               // preferred brands if mentioned
  "features": list[string],             // key features requested e.g. ["large capacity", "commercial grade", "energy efficient"]
  "nigerian_signals": list[string],     // detected context signals like "nysc", "student", etc.
  "shopping_intent": "browsing" | "comparing" | "ready_to_buy" | "out_of_context",
  "business_context": "commercial" | "personal" | null, // is the user buying for a business or personal use?
  "clarification_needed": boolean,      // do we need more info?
  "clarification_question": string or null  // what to ask if needed
}"""


class DiscoveryAgent:
    """Extracts structured intent from conversational shopping queries."""

    def __init__(self):
        self.provider = LLMProvider.get_instance()

    async def run(self, state: AgentState) -> AgentState:
        """Extract intent from user query and update agent state."""
        state.agent_trace.append("Discovery Agent: Extracting shopping intent...")
        logger.info(f"Discovery Agent processing: '{state.user_query[:80]}...'")

        try:
            intent = await self._extract_intent(state.user_query, state.conversation_history)
            state.extracted_intent = intent

            if intent.get("shopping_intent") == "out_of_context":
                state.clarification_needed = True
                state.clarification_question = (
                    "Abeg, I be shopping assistant! I only help with finding and buying products. "
                    "How can I help you shop today?"
                )
                state.agent_trace.append("  -> Out of context query detected.")
                return state

            state.category = intent.get("category")
            state.use_case = intent.get("use_case")
            state.business_context = intent.get("business_context")
            state.budget_min = intent.get("budget_min")
            state.budget_max = intent.get("budget_max")
            state.location_hint = intent.get("location")
            state.urgency = intent.get("urgency", "low")
            state.clarification_needed = intent.get("clarification_needed", False)
            state.clarification_question = intent.get("clarification_question")

            trace = (
                f"  -> Category: {state.category}, Budget: N{state.budget_min or 0:,} - N{state.budget_max:,}"
                if state.budget_max else f"  -> Category: {state.category}, Budget: not specified"
            )
            state.agent_trace.append(trace)
            logger.info(trace)

        except Exception as e:
            logger.error(f"Discovery Agent failed: {e}")
            state.errors.append(f"Discovery: {e}")
            state.extracted_intent = {"category": None, "clarification_needed": True}
            state.clarification_needed = True
            state.clarification_question = "What kind of product are you looking for, and what's your budget?"

        return state

    async def _extract_intent(
        self, query: str, history: list
    ) -> dict[str, Any]:
        messages = [{"role": "system", "content": DISCOVERY_SYSTEM_PROMPT}]

        for msg in history[-6:]:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": query})

        response = await self.provider.chat(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=512,
        )
        intent = json.loads(response.content)

        for key in ("budget_min", "budget_max"):
            if isinstance(intent.get(key), str):
                intent[key] = _parse_naira_amount(intent[key])

        return intent


def _parse_naira_amount(text: str) -> int | None:
    """Parse '400k', '1.5m', 'N400,000', '\u20a6400,000' -> int Naira."""
    if not text:
        return None
    # Strip currency symbols (both ASCII N and unicode \u20a6 Naira sign)
    text = str(text).replace("\u20a6", "").replace("N", "").replace(",", "").strip().lower()
    text = text.replace("naira", "").strip()
    m = re.match(r"([\d.]+)\s*([kmb]?)", text)
    if not m:
        return None
    val = float(m.group(1))
    suffix = m.group(2)
    if suffix == "k":
        val *= 1_000
    elif suffix == "m":
        val *= 1_000_000
    elif suffix == "b":
        val *= 1_000_000_000
    return int(val)
