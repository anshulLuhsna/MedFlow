"""Routing utilities for conditional edges"""

from agents.state import MedFlowState
import logging

logger = logging.getLogger(__name__)


def route_after_analysis(state: MedFlowState) -> str:
    """Route after data analysis"""
    shortage_count = state.get("shortage_count", 0)
    return "forecasting" if shortage_count > 0 else "END"


def route_after_optimization(state: MedFlowState) -> str:
    """Route after optimization"""
    strategies = state.get("allocation_strategies", [])
    if not strategies:
        return "END"
    feasible = [s for s in strategies if s.get("status") in ["optimal", "feasible"]]
    return "preference" if feasible else "END"


def route_after_human_review(state: MedFlowState) -> str:
    """Route after human review"""
    user_decision = state.get("user_decision")
    return "feedback" if user_decision is not None else "END"
