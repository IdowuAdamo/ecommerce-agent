"""
LangGraph Agent Orchestrator.

Defines the 7-agent DAG:
  Discovery → UserModeling → CommerceIntel → TrustValue → Recommendation → Explanation → ReviewSim

Uses LangGraph StateGraph for checkpointed, observable agent execution.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from langgraph.graph import StateGraph, END

from app.agents.discovery import DiscoveryAgent
from app.agents.user_modeling import UserModelingAgent
from app.agents.commerce_intel import CommerceIntelAgent
from app.agents.trust_value import TrustValueAgent
from app.agents.recommendation import RecommendationAgent
from app.agents.explanation import ExplanationAgent
from app.agents.review_simulation import ReviewSimulationAgent
from app.schemas.agent import AgentState, ChatMessage

logger = logging.getLogger(__name__)


def _state_to_dict(state: AgentState) -> dict:
    return state.model_dump()


def _dict_to_state(d: dict) -> AgentState:
    return AgentState(**d)


class AgentOrchestrator:
    """
    LangGraph DAG orchestrating all 7 NaijaShop AI agents.

    Graph topology:
      discovery → user_modeling → commerce_intel → trust_value
               → recommendation → explanation → [END]

    Review simulation runs in-pipeline for top products.
    """

    def __init__(self):
        self.discovery = DiscoveryAgent()
        self.user_modeling = UserModelingAgent()
        self.commerce_intel = CommerceIntelAgent()
        self.trust_value = TrustValueAgent()
        self.recommendation = RecommendationAgent()
        self.explanation = ExplanationAgent()
        self.review_sim = ReviewSimulationAgent()
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Build the LangGraph StateGraph DAG."""

        async def run_discovery(state: dict) -> dict:
            s = _dict_to_state(state)
            s = await self.discovery.run(s)
            # If clarification needed, skip to explanation directly
            if s.clarification_needed:
                s.final_response = s.clarification_question
            return _state_to_dict(s)

        async def run_user_modeling(state: dict) -> dict:
            s = _dict_to_state(state)
            s = await self.user_modeling.run(s)
            return _state_to_dict(s)

        async def run_commerce_intel(state: dict) -> dict:
            s = _dict_to_state(state)
            s = await self.commerce_intel.run(s)
            return _state_to_dict(s)

        async def run_trust_value(state: dict) -> dict:
            s = _dict_to_state(state)
            s = await self.trust_value.run(s)
            return _state_to_dict(s)

        async def run_recommendation(state: dict) -> dict:
            s = _dict_to_state(state)
            s = await self.recommendation.run(s)
            return _state_to_dict(s)

        async def run_explanation(state: dict) -> dict:
            s = _dict_to_state(state)
            s = await self.explanation.run(s)
            return _state_to_dict(s)

        async def run_review_sim(state: dict) -> dict:
            s = _dict_to_state(state)
            s = await self.review_sim.run(s)
            return _state_to_dict(s)

        def needs_clarification(state: dict) -> str:
            """Conditional edge: if discovery needs more info, skip to end."""
            if state.get("clarification_needed", False):
                return "end"
            return "continue"

        builder = StateGraph(dict)

        # Add all nodes
        builder.add_node("discovery", run_discovery)
        builder.add_node("user_modeling", run_user_modeling)
        builder.add_node("commerce_intel", run_commerce_intel)
        builder.add_node("trust_value", run_trust_value)
        builder.add_node("recommendation", run_recommendation)
        builder.add_node("explanation", run_explanation)
        builder.add_node("review_sim", run_review_sim)

        # Entry point
        builder.set_entry_point("discovery")

        # Conditional: skip pipeline if clarification needed
        builder.add_conditional_edges(
            "discovery",
            needs_clarification,
            {"continue": "user_modeling", "end": END},
        )

        # Main pipeline edges
        builder.add_edge("user_modeling", "commerce_intel")
        builder.add_edge("commerce_intel", "trust_value")
        builder.add_edge("trust_value", "recommendation")
        builder.add_edge("recommendation", "explanation")
        builder.add_edge("explanation", "review_sim")
        builder.add_edge("review_sim", END)

        return builder.compile()

    async def run(self, session_id: str, query: str, history: list[ChatMessage]) -> AgentState:
        """Execute the full agent pipeline for a user query."""
        start_time = time.time()

        initial_state = AgentState(
            session_id=session_id,
            user_query=query,
            conversation_history=history,
        )

        try:
            result_dict = await self.graph.ainvoke(_state_to_dict(initial_state))
            result = _dict_to_state(result_dict)
        except Exception as e:
            logger.error(f"Orchestrator failed: {e}")
            result = initial_state
            result.final_response = (
                "E don happen small issue. Please try again — "
                "I dey here to help you find the best products!"
            )
            result.errors.append(str(e))

        result.latency_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"Pipeline complete | Session: {session_id} | "
            f"Products: {len(result.ranked_products)} | "
            f"Latency: {result.latency_ms}ms"
        )
        return result
