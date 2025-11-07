"""
MedFlow Streamlit Dashboard

Web-based UI for running the MedFlow agentic workflow.
"""

import sys
import os
from pathlib import Path

# Add project root to path BEFORE any other imports
# This must happen first to ensure all modules can be found

# Determine project root - try multiple methods
try:
    # Method 1: From __file__ (when running as script)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
except NameError:
    # Method 2: From current working directory
    project_root = Path.cwd()
    # If we're in dashboard/, go up one level
    if project_root.name == "dashboard":
        project_root = project_root.parent

project_root_str = str(project_root.resolve())

# Add to sys.path if not already there (add at beginning)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# Verify the path is correct
agents_path = project_root / "agents"
if not agents_path.exists():
    # Try to find agents module in sys.path
    found = False
    for path_str in sys.path:
        path_obj = Path(path_str)
        if (path_obj / "agents").exists():
            found = True
            break
    if not found:
        raise ImportError(
            f"Cannot find agents module.\n"
            f"Project root: {project_root}\n"
            f"Agents path: {agents_path}\n"
            f"sys.path: {sys.path[:5]}..."
        )

# Now import other modules
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
import uuid
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import MedFlow modules (after path is set)
from agents.graph import medflow_graph
from agents.state import MedFlowState
from agents.nodes.feedback import feedback_node
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    load_dotenv(override=True)

# Configure page
st.set_page_config(
    page_title="MedFlow AI - Resource Allocation",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def initialize_state(
    resource_type: str,
    user_id: str,
    hospital_ids: Optional[list] = None,
    outbreak_id: Optional[str] = None
) -> MedFlowState:
    """Initialize workflow state"""
    return {
        "resource_type": resource_type,
        "user_id": user_id,
        "session_id": str(uuid.uuid4()),
        "hospital_ids": hospital_ids,
        "outbreak_id": outbreak_id,
        "messages": [],
        "workflow_status": "pending",
        "timestamp": datetime.now().isoformat(),
        "current_node": None,
        "shortage_count": 0,
        "shortage_hospitals": [],
        "active_outbreaks": [],
        "affected_regions": None,
        "analysis_summary": "",
        "demand_forecasts": {},
        "forecast_summary": "",
        "allocation_strategies": [],
        "strategy_count": 0,
        "ranked_strategies": [],
        "preference_profile": {},
        "final_recommendation": {},
        "explanation": "",
        "reasoning_chain": "",
        "user_decision": None,
        "user_feedback": None,
        "feedback_stored": False,
        "error": None,
        "execution_time_seconds": None
    }


def run_workflow_with_streamlit_interrupt(state: MedFlowState) -> tuple[MedFlowState, str]:
    """
    Run workflow until human review node, then pause for user input.
    Uses invoke() which is synchronous, and the human_review node detects Streamlit mode.
    
    Returns:
        (state, next_node) - Updated state and next node name
    """
    config = {
        "configurable": {
            "thread_id": state["user_id"]
        }
    }
    
    try:
        # Use invoke() which is synchronous (like CLI does)
        # The human_review node will detect Streamlit mode and return early
        result = medflow_graph.invoke(state, config=config)
        
        # Check if we're at human review (user_decision is None and we have ranked_strategies)
        if (result.get("user_decision") is None and 
            result.get("ranked_strategies") and 
            result.get("explanation") and
            result.get("current_node") == "human_review"):
            # We're at human review - return state for user input
            return result, "human_review"
        
        # Check if workflow is complete
        if result.get("workflow_status") == "completed":
            return result, "END"
        
        # Otherwise return the result
        return result, "END"
                    
    except Exception as e:
        logger.error(f"Error in workflow execution: {e}")
        state["error"] = str(e)
        state["workflow_status"] = "failed"
        return state, "END"


def run_workflow_sync(state: MedFlowState) -> MedFlowState:
    """
    Run workflow synchronously in a separate thread to avoid blocking.
    This is a workaround for Streamlit's synchronous nature.
    """
    config = {
        "configurable": {
            "thread_id": state["user_id"]
        }
    }
    
    try:
        # Use invoke for synchronous execution
        result = medflow_graph.invoke(state, config=config)
        return result
    except Exception as e:
        logger.error(f"Error in workflow execution: {e}")
        state["error"] = str(e)
        state["workflow_status"] = "failed"
        return state


def display_strategies_table(ranked_strategies: list) -> pd.DataFrame:
    """Display strategies in a DataFrame"""
    data = []
    for i, strategy in enumerate(ranked_strategies[:3]):
        summary = strategy.get('summary', {})
        shortage_reduction = summary.get('shortage_reduction', 
                                         summary.get('shortage_reduction_percent', 0))
        data.append({
            "Index": i,
            "Strategy": strategy.get("strategy_name", "Unknown"),
            "Cost": f"${summary.get('total_cost', 0):,.0f}",
            "Hospitals Helped": summary.get('hospitals_helped', 0),
            "Shortage Reduction": f"{shortage_reduction:.1f}%",
            "Preference Score": f"{strategy.get('preference_score', 0):.3f}"
        })
    return pd.DataFrame(data)


def main():
    """Main Streamlit app"""
    
    # Sidebar - Configuration
    with st.sidebar:
        st.title("🏥 MedFlow AI")
        st.markdown("**Resource Allocation Assistant**")
        st.markdown("Powered by LangGraph • Phase 5")
        
        st.divider()
        
        st.subheader("Configuration")
        
        resource_type = st.selectbox(
            "Resource Type",
            ["ventilators", "ppe", "o2_cylinders", "beds", "medications"],
            index=0
        )
        
        user_id = st.text_input(
            "User ID",
            value="default_user",
            help="User ID for preference learning"
        )
        
        hospital_ids_input = st.text_input(
            "Hospital IDs (optional)",
            value="",
            help="Comma-separated hospital IDs to process. Leave empty for all hospitals."
        )
        
        outbreak_id = st.text_input(
            "Outbreak ID (optional)",
            value="",
            help="Outbreak ID to use for context. Will process hospitals in affected regions."
        )
        
        # Parse hospital IDs
        hospital_ids = None
        if hospital_ids_input.strip():
            hospital_ids = [h.strip() for h in hospital_ids_input.split(",") if h.strip()]
        
        # Parse outbreak ID
        outbreak_id_parsed = None
        if outbreak_id.strip():
            outbreak_id_parsed = outbreak_id.strip()
    
    # Main content area
    st.title("🏥 MedFlow AI - Resource Allocation")
    st.markdown("Intelligent resource allocation using multi-agent AI workflow")
    
    # Initialize session state
    if "workflow_state" not in st.session_state:
        st.session_state.workflow_state = None
    if "workflow_running" not in st.session_state:
        st.session_state.workflow_running = False
    if "awaiting_review" not in st.session_state:
        st.session_state.awaiting_review = False
    if "human_review_state" not in st.session_state:
        st.session_state.human_review_state = None
    
    # Run workflow button
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        if st.button("🚀 Run Workflow", type="primary", disabled=st.session_state.workflow_running):
            st.session_state.workflow_running = True
            st.session_state.awaiting_review = False
            st.session_state.workflow_state = None
            st.session_state.human_review_state = None
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset"):
            st.session_state.workflow_state = None
            st.session_state.workflow_running = False
            st.session_state.awaiting_review = False
            st.session_state.human_review_state = None
            st.rerun()
    
    st.divider()
    
    # Display workflow status
    if st.session_state.workflow_running:
        st.info("⏳ Workflow is running... Please wait. This may take a few minutes.")
        
        # Initialize state
        initial_state = initialize_state(
            resource_type=resource_type,
            user_id=user_id,
            hospital_ids=hospital_ids,
            outbreak_id=outbreak_id_parsed
        )
        
        # Run workflow until human review
        with st.spinner("Running workflow... This may take a few minutes."):
            try:
                # Run workflow until we reach human review
                state, next_node = run_workflow_with_streamlit_interrupt(initial_state)
                
                if next_node == "human_review":
                    # Store state for human review
                    st.session_state.human_review_state = state
                    st.session_state.awaiting_review = True
                    st.session_state.workflow_running = False
                    st.rerun()
                else:
                    # Workflow completed without human review (shouldn't happen normally)
                    st.session_state.workflow_state = state
                    st.session_state.workflow_running = False
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error running workflow: {str(e)}")
                st.session_state.workflow_running = False
                logger.exception("Workflow execution error")
                st.rerun()
                
    elif st.session_state.awaiting_review:
        # Human Review Section
        st.header("🤖 AI Recommendations Ready for Review")
        
        state = st.session_state.human_review_state
        ranked_strategies = state.get("ranked_strategies", [])
        
        if ranked_strategies:
            # Display strategies table
            st.subheader("Allocation Strategies")
            df = display_strategies_table(ranked_strategies)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Display explanation
            explanation = state.get("explanation", "")
            if explanation:
                st.subheader("💡 Recommendation Explanation")
                st.info(explanation)
            
            # Strategy selection
            st.divider()
            st.subheader("Select Strategy")
            
            strategy_options = [f"{i}: {s.get('strategy_name', 'Unknown')}" 
                               for i, s in enumerate(ranked_strategies[:3])]
            
            selected_option = st.radio(
                "Choose a strategy:",
                strategy_options,
                index=0
            )
            
            selected_index = int(selected_option.split(":")[0])
            
            # Feedback
            feedback = st.text_area(
                "Optional Feedback",
                value="",
                help="Provide feedback to help improve future recommendations"
            )
            
            # Submit button
            if st.button("✅ Submit Selection", type="primary"):
                # Update state with user decision
                state["user_decision"] = selected_index
                state["user_feedback"] = feedback if feedback.strip() else None
                
                # Continue workflow from feedback node
                with st.spinner("Processing your selection..."):
                    try:
                        config = {
                            "configurable": {
                                "thread_id": state["user_id"]
                            }
                        }
                        
                        # Call feedback node manually
                        updated_state = feedback_node(state)
                        
                        # Continue workflow from feedback node
                        start_time = datetime.now()
                        
                        # Continue workflow execution using invoke (synchronous)
                        # The feedback node should lead to END, so this should complete quickly
                        final_state = medflow_graph.invoke(updated_state, config=config)
                        
                        # Calculate execution time
                        end_time = datetime.now()
                        execution_time = (end_time - start_time).total_seconds()
                        final_state["execution_time_seconds"] = execution_time
                        
                        st.session_state.workflow_state = final_state
                        st.session_state.awaiting_review = False
                        st.session_state.human_review_state = None
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing selection: {str(e)}")
                        logger.exception("Error in feedback processing")
        else:
            st.warning("No strategies available for review.")
            
    elif st.session_state.workflow_state:
        # Display final results
        state = st.session_state.workflow_state
        
        if state.get("error"):
            st.error(f"❌ Error: {state['error']}")
        else:
            st.success("✅ Workflow Completed!")
            
            # Display execution time
            exec_time = state.get("execution_time_seconds")
            if exec_time:
                st.metric("Execution Time", f"{exec_time:.1f}s")
            
            # Display final recommendation
            final_rec = state.get("final_recommendation")
            if final_rec:
                st.header("📊 Final Recommendation")
                
                summary = final_rec.get('summary', {})
                shortage_reduction = summary.get('shortage_reduction', 
                                                summary.get('shortage_reduction_percent', 0))
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Strategy", final_rec.get('strategy_name', 'Unknown'))
                with col2:
                    st.metric("Hospitals Helped", summary.get('hospitals_helped', 0))
                with col3:
                    st.metric("Total Cost", f"${summary.get('total_cost', 0):,.0f}")
                with col4:
                    st.metric("Shortage Reduction", f"{shortage_reduction:.1f}%")
                
                # Display explanation
                explanation = state.get("explanation", "")
                if explanation:
                    st.subheader("💡 Explanation")
                    st.info(explanation)
                
                # Display feedback confirmation
                if state.get("feedback_stored"):
                    st.success("💾 Preferences updated successfully!")
            
            # Display workflow summary
            with st.expander("📋 Workflow Summary"):
                st.write(f"**Resource Type:** {state.get('resource_type')}")
                st.write(f"**User ID:** {state.get('user_id')}")
                st.write(f"**Session ID:** {state.get('session_id')}")
                st.write(f"**Shortage Count:** {state.get('shortage_count', 0)}")
                st.write(f"**Strategies Generated:** {state.get('strategy_count', 0)}")
                
                if state.get('active_outbreaks'):
                    st.write(f"**Active Outbreaks:** {len(state.get('active_outbreaks', []))}")
                
                if state.get('affected_regions'):
                    st.write(f"**Affected Regions:** {', '.join(state.get('affected_regions', []))}")
    else:
        # Initial state
        st.info("👈 Configure settings in the sidebar and click 'Run Workflow' to start.")
        
        # Display help
        with st.expander("ℹ️ How to Use"):
            st.markdown("""
            ### Workflow Steps
            
            1. **Data Analyst** - Assesses current shortages and outbreaks
            2. **Forecasting** - Predicts 14-day demand for at-risk hospitals
            3. **Optimization** - Generates multiple allocation strategies
            4. **Preference** - Ranks strategies by your preferences
            5. **Reasoning** - Generates AI explanation
            6. **Human Review** - You review and select a strategy
            7. **Feedback** - System learns from your decision
            
            ### Tips
            
            - **Resource Type**: Choose the resource you want to allocate
            - **Hospital IDs**: Leave empty to process all hospitals, or specify comma-separated IDs
            - **Outbreak ID**: Use an outbreak ID to simulate realistic scenarios
            - **User ID**: Use consistent ID for preference learning
            """)


if __name__ == "__main__":
    main()
