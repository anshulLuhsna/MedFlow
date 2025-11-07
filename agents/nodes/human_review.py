"""Human Review Node (HITL)"""

from typing import Dict
from agents.state import MedFlowState
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import logging
import os
import sys

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
console = Console()

_langsmith_project = os.getenv("LANGSMITH_PROJECT", "medflow")


@traceable(name="human_review_node", project_name=_langsmith_project)
def human_review_node(state: MedFlowState) -> Dict:
    """
    Human-in-the-Loop Review Node.

    Displays recommendations and prompts user to select a strategy.
    """
    logger.info("[Human Review] Awaiting user decision")

    # Display recommendations
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]🤖 AI Recommendations Ready for Review[/bold cyan]",
        border_style="cyan"
    ))

    # Show top 3 strategies in a table
    ranked = state["ranked_strategies"][:3]
    
    table = Table(title="Allocation Strategies", show_header=True)
    table.add_column("Index", style="cyan")
    table.add_column("Strategy", style="green")
    table.add_column("Cost", justify="right")
    table.add_column("Hospitals", justify="right")
    table.add_column("Reduction", justify="right")
    table.add_column("Score", justify="right")

    for i, strategy in enumerate(ranked):
        summary = strategy.get('summary', {})
        # Handle both 'shortage_reduction' and 'shortage_reduction_percent' keys
        shortage_reduction = summary.get('shortage_reduction', 
                                         summary.get('shortage_reduction_percent', 0))
        total_cost = summary.get('total_cost', 0)
        hospitals_helped = summary.get('hospitals_helped', 0)
        
        table.add_row(
            str(i),
            strategy.get("strategy_name", "Unknown"),
            f"${total_cost:,.0f}",
            str(hospitals_helped),
            f"{shortage_reduction:.1f}%",
            f"{strategy.get('preference_score', 0):.3f}"
        )

    console.print(table)

    # Show explanation
    console.print("\n")
    console.print(Panel(
        state["explanation"],
        title="[bold green]💡 Recommendation Explanation[/bold green]",
        border_style="green"
    ))

    # Check if we're running in automated/testing mode (auto-select first strategy)
    automated_mode = os.getenv("AUTOMATED_TESTING", "false").lower() == "true"
    
    # Check if we're running in Streamlit (workaround for blocking input)
    in_streamlit = "streamlit" in sys.modules or os.getenv("STREAMLIT_RUNNING", "false").lower() == "true"
    
    if automated_mode:
        # Automated mode: Simulate realistic user behavior with preferences
        # Default to biased selection (70% cost-efficient, 30% random) to enable learning
        # Set DEMO_LEARNING_MODE=true to always select top-ranked (for testing ranking consistency)
        # Otherwise, use biased selection to simulate user preferences and enable learning
        learning_mode = os.getenv("DEMO_LEARNING_MODE", "false").lower() == "true"
        
        if learning_mode:
            # Always select top-ranked strategy to demonstrate ranking consistency
            selected_index = 0  # Rank #1
            selection_reason = "learning mode (always select top-ranked)"
        else:
            # Biased selection mode: simulate user preferences (enables learning)
            import random
            
            # Simulate a user who prefers cost-efficient strategies (70% of the time)
            # This creates a learning signal: agent should learn this preference over time
            preference_bias = 0.7  # 70% prefer cost-efficient
            
            # Find cost-efficient strategy (usually has "cost" or "efficient" in name)
            cost_efficient_idx = None
            for idx, strategy in enumerate(ranked):
                strategy_name = strategy.get("strategy_name", "").lower()
                if "cost" in strategy_name or "efficient" in strategy_name:
                    cost_efficient_idx = idx
                    break
            
            # Biased selection: 70% cost-efficient, 30% random
            if cost_efficient_idx is not None and random.random() < preference_bias:
                selected_index = cost_efficient_idx
                selection_reason = "biased (prefers cost-efficient)"
            else:
                selected_index = random.randint(0, len(ranked) - 1)
                selection_reason = "random (exploring)"
        
        selected_strategy = ranked[selected_index]
        
        # Calculate rank (1-indexed for logging)
        rank = selected_index + 1
        preference_score = selected_strategy.get('preference_score', 0)
        
        logger.info(
            f"[Human Review] Automated mode ({selection_reason}): "
            f"Selected '{selected_strategy['strategy_name']}' "
            f"(ranked #{rank} by agent, preference_score: {preference_score:.3f})"
        )
        
        return {
            "user_decision": selected_index,
            "user_feedback": f"Automated selection ({selection_reason}) for testing",
            "current_node": "human_review"
        }
    
    if in_streamlit:
        # In Streamlit mode, we don't block here - the dashboard will handle it
        # Return None for now - the dashboard will update this later
        logger.info("[Human Review] Running in Streamlit mode - dashboard will handle user input")
        return {
            "user_decision": None,
            "user_feedback": None,
            "current_node": "human_review",
            "_streamlit_mode": True  # Flag for Streamlit
        }
    
    # Normal CLI mode - prompt for input
    console.print("\n")
    selected_index = int(input("Select a strategy (enter index 0-2): "))

    selected_strategy = ranked[selected_index]
    console.print(f"\n✅ Selected: {selected_strategy['strategy_name']}")

    # Optional feedback
    feedback = input("\nOptional feedback (press Enter to skip): ")

    logger.info(
        f"[Human Review] User selected strategy {selected_index}: "
        f"{selected_strategy['strategy_name']}"
    )

    if feedback:
        logger.info(f"[Human Review] User feedback: {feedback}")

    return {
        "user_decision": selected_index,
        "user_feedback": feedback if feedback else None,
        "current_node": "human_review"
    }
