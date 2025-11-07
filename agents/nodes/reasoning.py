"""Reasoning Agent Node"""

from typing import Dict
from agents.state import MedFlowState
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
import logging
import os

# LangSmith tracing
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    LANGSMITH_AVAILABLE = False

logger = logging.getLogger(__name__)

_langsmith_project = os.getenv("LANGSMITH_PROJECT", "medflow")


@traceable(name="reasoning_node", run_type="llm", project_name=_langsmith_project)
def reasoning_node(state: MedFlowState) -> Dict:
    """
    Reasoning Agent - Generate LLM explanation of recommendations.

    Uses Groq/Llama 3.3 70B to create natural language explanation.
    """
    logger.info("[Reasoning] Generating LLM explanation")

    import time
    reasoning_start = time.time()

    # Get top recommendation
    top_strategy = state.get("ranked_strategies", [{}])[0] if state.get("ranked_strategies") else {}
    
    # Safely extract strategy summary with defaults
    summary = top_strategy.get('summary', {})
    strategy_name = top_strategy.get('strategy_name', 'Recommended')
    hospitals_helped = summary.get('hospitals_helped', 0)
    total_cost = summary.get('total_cost', 0)
    # Handle both 'shortage_reduction' and 'shortage_reduction_percent' keys
    shortage_reduction = summary.get('shortage_reduction', 
                                     summary.get('shortage_reduction_percent', 0))
    resource_type = state.get('resource_type', 'resources')
    shortage_count = state.get('shortage_count', 0)
    
    # Check if GROQ_API_KEY is set
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        # Fallback: Generate a template-based explanation if Groq API key is not set
        logger.warning("[Reasoning] GROQ_API_KEY not set. Using fallback template-based explanation.")
        
        explanation = (
            f"Based on the analysis of {shortage_count} hospitals with {resource_type} shortages, "
            f"the {strategy_name} strategy is recommended. This approach will help {hospitals_helped} hospitals "
            f"at a total transfer cost of ${total_cost:,.0f}, reducing shortages by {shortage_reduction:.1f}%. "
            f"The strategy has been optimized using linear programming to balance cost efficiency, coverage, "
            f"and urgency while respecting distance constraints and hospital capacities."
        )
        reasoning_elapsed = time.time() - reasoning_start
        logger.info(f"[Reasoning] Template-based explanation generated in {reasoning_elapsed:.2f}s")
    else:
        # Initialize LLM (Groq/Llama 3.3 70B)
        try:
            llm = ChatGroq(
                model=os.getenv("DEFAULT_LLM_MODEL", "llama-3.3-70b-versatile"),
                temperature=float(os.getenv("DEFAULT_LLM_TEMPERATURE", "0.3")),
                groq_api_key=groq_api_key
            )

            # Build prompt
            system_prompt = """You are a healthcare operations analyst explaining resource allocation recommendations.
Be concise, specific, and data-driven. Use 200-300 words."""

            human_prompt = f"""
Explain this resource allocation recommendation:

Situation:
- {shortage_count} hospitals need {resource_type}
- {len(state.get('active_outbreaks', []))} active outbreak(s)

Top Recommended Strategy: {strategy_name}
- Hospitals Helped: {hospitals_helped}
- Total Cost: ${total_cost:,.0f}
- Shortage Reduction: {shortage_reduction:.1f}%

User Profile: {state.get('preference_profile', {}).get('preference_type', 'unknown')}

Generate a clear, actionable explanation in 3-4 sentences.
"""

            # Generate explanation
            llm_start = time.time()
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ])
            llm_elapsed = time.time() - llm_start
            logger.info(f"[Reasoning] LLM call completed in {llm_elapsed:.2f}s")

            explanation = response.content
        except Exception as e:
            llm_elapsed = time.time() - llm_start if 'llm_start' in locals() else 0
            logger.error(f"[Reasoning] LLM call failed after {llm_elapsed:.2f}s: {e}. Using fallback explanation.")
            # Fallback explanation
            explanation = (
                f"The {strategy_name} strategy is recommended to address "
                f"{shortage_count} hospitals with {resource_type} shortages. "
                f"This approach will help {hospitals_helped} hospitals "
                f"at a cost of ${total_cost:,.0f}, reducing shortages by "
                f"{shortage_reduction:.1f}%."
            )
    
    reasoning_elapsed = time.time() - reasoning_start
    logger.info(f"[Reasoning] Generated {len(explanation)} character explanation in {reasoning_elapsed:.2f}s total")

    return {
        "final_recommendation": top_strategy,
        "explanation": explanation,
        "reasoning_chain": explanation,
        "messages": [AIMessage(content=explanation)],
        "current_node": "reasoning"
    }
