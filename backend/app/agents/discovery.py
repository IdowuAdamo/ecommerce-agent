"""
Agent 2: Conversational Discovery Agent.

Extracts structured shopping intent from natural language queries using GPT-4o.
Handles Nigerian English, pidgin, slang, and cultural context.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

from app.config import get_settings
from app.schemas.agent import AgentState

logger = logging.getLogger(__name__)

DISCOVERY_SYSTEM_PROMPT = """You are a Nigerian e-commerce shopping assistant that understands Nigerian English, Pidgin, and local shopping culture.

Extract structured shopping intent from the user's message. Understand Nigerian context:
- "NYSC" = National Youth Service Corps (young person, tight budget ~₦50k-200k)
- "January blues" = post-holiday financial squeeze
- "Detty December" = festive season, higher budget
- "Abuja salary" = government worker, stable income
- Budget in "k" = thousands (₦400k = ₦400,000)
- "Am I doing well" / "E go dey alright" = positive confirmation

Return a JSON object with these fields:
{
  "category": string or null,           // product category (laptops, phones, appliances, etc.)
  "use_case": string or null,           // what they want to do with it
  "budget_min": number or null,         // minimum budget in Naira
  "budget_max": number or null,         // maximum budget in Naira
  "location": string or null,           // Lagos, Abuja, PH, etc.
  "urgency": "low" | "medium" | "high", // how urgently they need it
  "brands": list[string],               // preferred brands if mentioned
  "features": list[string],             // key features requested
  "nigerian_signals": list[string],     // detected context signals like "nysc", "student", etc.
  "shopping_intent": "browsing" | "comparing" | "ready_to_buy" | "out_of_context",
  "clarification_needed": boolean,      // do we need more info?
  "clarification_question": string or null  // what to ask if needed
}"""


class DiscoveryAgent:
    """Extracts structured intent from conversational shopping queries."""

    def __init__(self):
        s = get_settings()
        self.client = AsyncOpenAI(api_key=s.openai_api_key)
        self.model = s.openai_model

    async def run(self, state: AgentState) -> AgentState:
        """Extract intent from user query and update agent state."""
        state.agent_trace.append("🔍 Discovery Agent: Extracting shopping intent...")
        logger.info(f"Discovery Agent processing: '{state.user_query[:80]}...'")

        try:
            intent = await self._extract_intent(state.user_query, state.conversation_history)
            state.extracted_intent = intent
            
            if intent.get("shopping_intent") == "out_of_context":
                state.clarification_needed = True
                state.clarification_question = "Abeg, I be shopping assistant! I only help with finding and buying products. How can I help you shop today?"
                state.agent_trace.append("  → Out of context query detected.")
                return state

            state.category = intent.get("category")
            state.use_case = intent.get("use_case")
            state.budget_min = intent.get("budget_min")
            state.budget_max = intent.get("budget_max")
            state.location_hint = intent.get("location")
            state.urgency = intent.get("urgency", "low")
            state.clarification_needed = intent.get("clarification_needed", False)
            state.clarification_question = intent.get("clarification_question")

            trace = (
                f"  → Category: {state.category}, Budget: ₦{state.budget_min or 0:,} – ₦{state.budget_max:,}"
                if state.budget_max else f"  → Category: {state.category}, Budget: not specified"
            )
            state.agent_trace.append(trace)
            logger.info(trace)

        except Exception as e:
            logger.error(f"Discovery Agent failed: {e}")
            state.errors.append(f"Discovery: {e}")
            # Graceful fallback — still pass through
            state.extracted_intent = {"category": None, "clarification_needed": True}
            state.clarification_needed = True
            state.clarification_question = "What kind of product are you looking for, and what's your budget?"

        return state

    async def _extract_intent(
        self, query: str, history: list
    ) -> dict[str, Any]:
        messages = [{"role": "system", "content": DISCOVERY_SYSTEM_PROMPT}]

        # Include recent conversation history for context
        for msg in history[-6:]:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": query})

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content
        intent = json.loads(raw)

        # Post-process budget — handle "400k" style inputs
        for key in ("budget_min", "budget_max"):
            if isinstance(intent.get(key), str):
                intent[key] = _parse_naira_amount(intent[key])

        return intent


def _parse_naira_amount(text: str) -> int | None:
    """Parse '400k', '1.5m', '₦400,000' → int Naira."""
    if not text:
        return None
    text = str(text).lower().replace("₦", "").replace(",", "").strip()
    m = re.match(r"([\d.]+)\s*([km]?)", text)
    if not m:
        return None
    val = float(m.group(1))
    suffix = m.group(2)
    if suffix == "k":
        val *= 1_000
    elif suffix == "m":
        val *= 1_000_000
    return int(val)
