"""
MedFlow Demo Dashboard - Time-Series Simulation

Demo-ready Streamlit interface for showcasing the MedFlow AI system with:
- Day-by-day time progression
- Normal vs Outbreak scenarios
- Learning progression tracking
- Interactive workflow execution
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add project root to path (robust like dashboard/app.py)
try:
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
except NameError:
    project_root = Path.cwd()
    if project_root.name == "dashboard":
        project_root = project_root.parent

project_root_str = str(project_root.resolve())
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# Verify `agents` module is reachable; try to discover if not directly present
agents_path = project_root / "agents"
if not agents_path.exists():
    found = False
    for path_str in list(sys.path):
        if (Path(path_str) / "agents").exists():
            found = True
            break
    if not found:
        raise ImportError(
            f"Cannot find agents module.\n"
            f"Project root: {project_root}\n"
            f"Agents path: {agents_path}\n"
            f"sys.path: {sys.path[:5]}..."
        )

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Dict, Any, List
import uuid
import logging

# Import MedFlow modules
from agents.graph import medflow_graph
from agents.state import MedFlowState
from agents.nodes.feedback import feedback_node
from agents.tools.api_client import MedFlowAPIClient
from dotenv import load_dotenv

# CrewAI import (optional)
try:
    from agents.crewai import MedFlowCrew
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    MedFlowCrew = None

# Load environment variables
load_dotenv(override=True)

# Configure page
st.set_page_config(
    page_title="MedFlow AI - Demo Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize API client
api_client = MedFlowAPIClient()

# Outbreak definitions
OUTBREAKS = {
    "TB Outbreak": {
        "id": "dd891681-7e1a-409c-81dd-96009394802c",
        "start_date": "2024-06-15",
        "end_date": "2024-07-30",
        "regions": ["Bihar", "Uttar Pradesh", "West Bengal", "Madhya Pradesh"],
        "severity": "high"
    }
}

# === HELPER FUNCTIONS ===

def initialize_session_state():
    """Initialize all session state variables"""
    if "demo_mode" not in st.session_state:
        st.session_state.demo_mode = "Normal"
    if "framework" not in st.session_state:
        st.session_state.framework = "langgraph"  # Default to LangGraph
    if "current_date" not in st.session_state:
        st.session_state.current_date = datetime(2024, 1, 15)
    if "date_range" not in st.session_state:
        st.session_state.date_range = [datetime(2024, 1, 15), datetime(2024, 2, 15)]
    if "prev_date_range" not in st.session_state:
        st.session_state.prev_date_range = list(st.session_state.date_range)
    if "resource_type" not in st.session_state:
        st.session_state.resource_type = "ventilators"
    if "hospital_limit" not in st.session_state:
        st.session_state.hospital_limit = 5
    if "user_id" not in st.session_state:
        st.session_state.user_id = "demo_user"
    if "selected_regions" not in st.session_state:
        st.session_state.selected_regions = []
    if "simulation_history" not in st.session_state:
        st.session_state.simulation_history = []
    if "workflow_running" not in st.session_state:
        st.session_state.workflow_running = False
    if "awaiting_review" not in st.session_state:
        st.session_state.awaiting_review = False
    if "current_workflow_state" not in st.session_state:
        st.session_state.current_workflow_state = None
    if "learning_metrics" not in st.session_state:
        st.session_state.learning_metrics = []
    if "inventory_overlays" not in st.session_state:
        # dict: date_str -> list of adjustments {hospital_id, resource_type, delta_available}
        st.session_state.inventory_overlays = {}
    if "hospital_cache" not in st.session_state:
        # Cache for hospital ID -> name mapping
        st.session_state.hospital_cache = {}


def get_outbreak_status(date: datetime) -> Optional[Dict]:
    """Check if date falls within any outbreak period"""
    for name, outbreak in OUTBREAKS.items():
        start = datetime.strptime(outbreak["start_date"], "%Y-%m-%d")
        end = datetime.strptime(outbreak["end_date"], "%Y-%m-%d")
        if start <= date <= end:
            return {"name": name, **outbreak}
    return None


def get_hospital_name(hospital_id: str) -> str:
    """Get hospital name from ID, using cache"""
    if not hospital_id:
        return "Unknown"
    
    # Check cache first
    if hospital_id in st.session_state.hospital_cache:
        return st.session_state.hospital_cache[hospital_id]
    
    # Fetch from API
    try:
        response = api_client.client.get(
            f"/api/v1/hospitals/{hospital_id}",
            headers={"X-API-Key": api_client.api_key}
        )
        if response.status_code == 200:
            data = response.json()
            hospital_name = data.get("hospital", {}).get("name", hospital_id[:8] + "...")
            st.session_state.hospital_cache[hospital_id] = hospital_name
            return hospital_name
    except Exception as e:
        logger.warning(f"Failed to fetch hospital {hospital_id}: {e}")
    
    # Fallback to truncated ID
    return hospital_id[:8] + "..."


def fetch_day_data(date: datetime, resource_type: str, regions: List[str] = None, limit: int = 5) -> Dict:
    """Fetch all data for a specific day via API"""
    date_str = date.strftime("%Y-%m-%d")
    
    try:
        # Fetch shortages
        shortages = api_client.detect_shortages(
            resource_type=resource_type,
            limit=limit,
            regions=regions,
            simulation_date=date_str
        )
        
        # Fetch active outbreaks
        outbreaks = api_client.get_active_outbreaks(simulation_date=date_str)
        
        # Generate strategies
        strategies = api_client.generate_strategies(
            resource_type=resource_type,
            n_strategies=3,
            limit=limit,
            regions=regions,
            simulation_date=date_str
        )
        
        return {
            "date": date_str,
            "shortages": shortages,
            "outbreaks": outbreaks,
            "strategies": strategies,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error fetching day data: {e}")
        return {
            "date": date_str,
            "error": str(e),
            "success": False
        }


def initialize_workflow_state(date: datetime, resource_type: str, regions: List[str] = None, limit: int = 5) -> MedFlowState:
    """Initialize MedFlow workflow state"""
    return {
        "resource_type": resource_type,
        "user_id": st.session_state.user_id,
        "session_id": str(uuid.uuid4()),
        "hospital_ids": None,
        "outbreak_id": None,
        "regions": regions,
        "hospital_limit": limit,
        "simulation_date": date.strftime("%Y-%m-%d"),
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


def run_workflow_until_review(state: MedFlowState) -> tuple[MedFlowState, str]:
    """Run workflow until human review"""
    framework = st.session_state.get("framework", "langgraph")
    
    try:
        if framework == "crewai" and CREWAI_AVAILABLE:
            # Use CrewAI implementation
            crew_inputs = {
                "resource_type": state.get("resource_type", "ventilators"),
                "user_id": state.get("user_id", "default_user"),
                "simulation_date": state.get("simulation_date"),
                "hospital_ids": state.get("hospital_ids"),
                "outbreak_id": state.get("outbreak_id"),
                "regions": state.get("regions"),
                "hospital_limit": state.get("hospital_limit", 5)
            }
            
            crew = MedFlowCrew().crew()
            result_obj = crew.kickoff(inputs=crew_inputs)
            
            # Convert CrewAI result to MedFlowState format
            result = state.copy()
            result["workflow_status"] = "completed"
            
            # Extract data from CrewAI output
            if result_obj.tasks:
                for task in result_obj.tasks:
                    if hasattr(task, 'output') and task.output:
                        try:
                            import json
                            if isinstance(task.output, str):
                                output_data = json.loads(task.output)
                            else:
                                output_data = task.output
                            
                            # Map task outputs to state
                            if isinstance(output_data, dict):
                                if 'shortage_count' in output_data:
                                    result["shortage_count"] = output_data.get("shortage_count", 0)
                                    result["shortage_hospitals"] = output_data.get("shortage_hospitals", [])
                                    result["active_outbreaks"] = output_data.get("active_outbreaks", [])
                                    result["analysis_summary"] = output_data.get("analysis_summary", "")
                                elif 'ranked_strategies' in output_data:
                                    result["ranked_strategies"] = output_data.get("ranked_strategies", [])
                                    result["preference_profile"] = output_data.get("preference_profile", {})
                                elif 'explanation' in output_data:
                                    result["explanation"] = output_data.get("explanation", "")
                                    result["final_recommendation"] = output_data.get("final_recommendation", {})
                                elif 'selected_strategy_index' in output_data:
                                    result["user_decision"] = output_data.get("selected_strategy_index")
                                    result["user_feedback"] = output_data.get("user_feedback")
                        except Exception as e:
                            logger.warning(f"Error parsing CrewAI task output: {e}")
            
            # Check if we need human review
            if (result.get("user_decision") is None and 
                result.get("ranked_strategies") and 
                len(result.get("ranked_strategies", [])) > 0):
                return result, "human_review"
            
            return result, "END"
        else:
            # Use LangGraph implementation (default)
            config = {"configurable": {"thread_id": state["user_id"]}}
            result = medflow_graph.invoke(state, config=config)
            
            if (result.get("user_decision") is None and 
                result.get("ranked_strategies") and 
                result.get("current_node") == "human_review"):
                return result, "human_review"
            
            return result, "END"
    except Exception as e:
        logger.error(f"Workflow error: {e}")
        state["error"] = str(e)
        state["workflow_status"] = "failed"
        return state, "END"


# === UI COMPONENTS ===

def render_setup_tab():
    """Render the Setup & Configuration tab"""
    st.header("⚙️ Demo Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Scenario Selection")
        
        # Framework selector (if CrewAI is available)
        if CREWAI_AVAILABLE:
            framework = st.selectbox(
                "Framework",
                ["langgraph", "crewai"],
                index=0 if st.session_state.framework == "langgraph" else 1,
                help="LangGraph: Original implementation | CrewAI: Alternative agent framework"
            )
            st.session_state.framework = framework
        else:
            st.session_state.framework = "langgraph"
            st.info("ℹ️ CrewAI not available. Using LangGraph.")
        
        # Mode selector
        mode = st.radio(
            "Demo Mode",
            ["Normal", "Outbreak"],
            index=0 if st.session_state.demo_mode == "Normal" else 1,
            help="Normal: Routine operations | Outbreak: TB outbreak simulation (Jun-Jul 2024)"
        )
        
        if mode != st.session_state.demo_mode:
            st.session_state.demo_mode = mode
            # Update date range based on mode
            if mode == "Normal":
                st.session_state.current_date = datetime(2024, 1, 15)
                st.session_state.date_range = [datetime(2024, 1, 15), datetime(2024, 2, 15)]
                st.session_state.selected_regions = []
            else:
                st.session_state.current_date = datetime(2024, 6, 20)
                st.session_state.date_range = [datetime(2024, 6, 15), datetime(2024, 7, 30)]
                st.session_state.selected_regions = ["Bihar", "West Bengal"]
        
        # Date controls
        if mode == "Outbreak":
            st.info("📅 TB Outbreak Period: June 15 - July 30, 2024")
            start_date = st.date_input(
                "Start Date",
                value=st.session_state.date_range[0],
                min_value=datetime(2024, 6, 15),
                max_value=datetime(2024, 7, 30)
            )
            end_date = st.date_input(
                "End Date",
                value=st.session_state.date_range[1],
                min_value=datetime(2024, 6, 15),
                max_value=datetime(2024, 7, 30)
            )
        else:
            start_date = st.date_input(
                "Start Date",
                value=st.session_state.date_range[0],
                min_value=datetime(2024, 1, 1),
                max_value=datetime(2024, 12, 31)
            )
            end_date = st.date_input(
                "End Date",
                value=st.session_state.date_range[1],
                min_value=datetime(2024, 1, 1),
                max_value=datetime(2024, 12, 31)
            )
        
        st.session_state.date_range = [
            datetime.combine(start_date, datetime.min.time()),
            datetime.combine(end_date, datetime.min.time())
        ]
        # Only update current_date if user changed the date range
        if (
            st.session_state.prev_date_range[0] != st.session_state.date_range[0]
            or st.session_state.prev_date_range[1] != st.session_state.date_range[1]
        ):
            st.session_state.current_date = st.session_state.date_range[0]
            st.session_state.prev_date_range = list(st.session_state.date_range)
    
    with col2:
        st.subheader("Resource Configuration")
        
        # Resource type
        resource_type = st.selectbox(
            "Resource Type",
            ["ventilators", "ppe", "o2_cylinders", "beds", "medications"],
            index=0
        )
        st.session_state.resource_type = resource_type
        
        # Hospital limit
        limit = st.slider(
            "Hospital Limit",
            min_value=3,
            max_value=20,
            value=5,
            help="Number of hospitals to process (demo uses smaller sets for speed)"
        )
        st.session_state.hospital_limit = limit
        
        # Region filter (Outbreak mode)
        if mode == "Outbreak":
            regions = st.multiselect(
                "Affected Regions",
                ["Bihar", "Uttar Pradesh", "West Bengal", "Madhya Pradesh"],
                default=["Bihar", "West Bengal"]
            )
            st.session_state.selected_regions = regions
        
        # User ID
        user_id = st.text_input(
            "User ID",
            value=st.session_state.user_id,
            help="Consistent ID for preference learning"
        )
        st.session_state.user_id = user_id
    
    st.divider()
    
    # Demo controls
    st.subheader("🎬 Demo Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("▶️ Start Simulation", type="primary", use_container_width=True):
            st.session_state.simulation_history = []
            st.session_state.learning_metrics = []
            st.session_state.current_date = st.session_state.date_range[0]
            st.success("Simulation initialized! Go to 'Daily Simulation' tab.")
    
    with col2:
        if st.button("🔄 Reset All", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            initialize_session_state()
            st.rerun()
    
    with col3:
        if st.button("📊 View Results", use_container_width=True, disabled=len(st.session_state.simulation_history) == 0):
            st.info("Go to 'Learning Analytics' tab to view results")
    
    with col4:
        days_simulated = len(st.session_state.simulation_history)
        st.metric("Days Simulated", days_simulated)


def render_daily_simulation_tab():
    """Render the Day-by-Day Simulation tab"""
    st.header("📅 Daily Simulation")
    
    # Current date display
    outbreak_status = get_outbreak_status(st.session_state.current_date)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader(f"📆 {st.session_state.current_date.strftime('%B %d, %Y')}")
        if outbreak_status:
            st.error(f"🚨 **{outbreak_status['name']}** - Severity: {outbreak_status['severity'].upper()}")
        else:
            st.success("✅ Normal Operations")
    
    with col2:
        day_num = (st.session_state.current_date - st.session_state.date_range[0]).days + 1
        total_days = (st.session_state.date_range[1] - st.session_state.date_range[0]).days + 1
        st.metric("Day", f"{day_num} / {total_days}")
    
    with col3:
        st.metric("Resource", st.session_state.resource_type.title())
    
    st.divider()
    
    # Navigation controls
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("⏮️ First Day", use_container_width=True):
            st.session_state.current_date = st.session_state.date_range[0]
            st.rerun()
    
    with col2:
        if st.button("◀️ Previous", use_container_width=True):
            new_date = st.session_state.current_date - timedelta(days=1)
            if new_date >= st.session_state.date_range[0]:
                st.session_state.current_date = new_date
                st.rerun()
    
    with col3:
        if st.button("🎯 Run Today", type="primary", use_container_width=True, 
                    disabled=st.session_state.workflow_running):
            st.session_state.workflow_running = True
            st.rerun()
    
    with col4:
        if st.button("▶️ Next", use_container_width=True):
            new_date = st.session_state.current_date + timedelta(days=1)
            if new_date <= st.session_state.date_range[1]:
                st.session_state.current_date = new_date
                st.rerun()
    
    with col5:
        if st.button("⏭️ Last Day", use_container_width=True):
            st.session_state.current_date = st.session_state.date_range[1]
            st.rerun()
    
    st.divider()
    
    # Workflow execution
    if st.session_state.workflow_running:
        st.info("⏳ Running workflow for current date... This may take 30-60 seconds.")
        
        with st.spinner("Executing MedFlow AI workflow..."):
            try:
                # Initialize workflow state
                initial_state = initialize_workflow_state(
                    date=st.session_state.current_date,
                    resource_type=st.session_state.resource_type,
                    regions=st.session_state.selected_regions if st.session_state.demo_mode == "Outbreak" else None,
                    limit=st.session_state.hospital_limit
                )
                
                # AGGRESSIVE DEBUG: Log simulation_date and state keys
                sim_date = initial_state.get('simulation_date')
                logger.error(f"[Streamlit] ⚠️ WORKFLOW STATE DEBUG:")
                logger.error(f"[Streamlit]   simulation_date = {sim_date} (type: {type(sim_date)})")
                logger.error(f"[Streamlit]   State keys: {list(initial_state.keys())}")
                logger.error(f"[Streamlit]   Current date: {st.session_state.current_date}")
                print(f"[Streamlit ERROR] simulation_date in state: {sim_date}")
                print(f"[Streamlit ERROR] State has simulation_date key: {'simulation_date' in initial_state}")
                if sim_date is None:
                    logger.error(f"[Streamlit] ❌ CRITICAL: simulation_date is None! Using fallback.")
                    initial_state['simulation_date'] = st.session_state.current_date.strftime("%Y-%m-%d")
                    logger.error(f"[Streamlit] ✅ Set fallback simulation_date: {initial_state['simulation_date']}")
                
                # Run workflow
                state, next_node = run_workflow_until_review(initial_state)
                
                if next_node == "human_review":
                    st.session_state.current_workflow_state = state
                    st.session_state.awaiting_review = True
                    st.session_state.workflow_running = False
                    st.rerun()
                else:
                    st.error(f"Unexpected workflow end: {state.get('error', 'Unknown error')}")
                    st.session_state.workflow_running = False
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.workflow_running = False
    
    elif st.session_state.awaiting_review:
        # Display strategies for review
        state = st.session_state.current_workflow_state
        ranked_strategies = state.get("ranked_strategies", [])
        
        if ranked_strategies:
            st.success("✅ Workflow Complete - Ready for Review")
            
            # Display strategies
            st.subheader("🎯 Recommended Strategies")
            
            for i, strategy in enumerate(ranked_strategies[:3]):
                summary = strategy.get('summary', {})
                with st.expander(f"**{i+1}. {strategy.get('strategy_name', 'Unknown')}**", expanded=(i==0)):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Cost", f"${summary.get('total_cost', 0):,.0f}")
                    with col2:
                        st.metric("Hospitals Helped", summary.get('hospitals_helped', 0))
                    with col3:
                        shortage_reduction = summary.get('shortage_reduction', summary.get('shortage_reduction_percent', 0))
                        st.metric("Shortage Reduction", f"{shortage_reduction:.1f}%")
                    with col4:
                        st.metric("Preference Score", f"{strategy.get('preference_score', 0):.3f}")
                    
                    # Show allocation details
                    allocations = strategy.get('allocations', [])
                    if allocations:
                        st.markdown("**📦 Resource Transfers:**")
                        allocation_data = []
                        for alloc in allocations:
                            from_id = alloc.get('from_hospital_id') or alloc.get('from_hospital') or alloc.get('from')
                            to_id = alloc.get('to_hospital_id') or alloc.get('to_hospital') or alloc.get('to')
                            quantity = alloc.get('quantity', 0)
                            cost = alloc.get('transfer_cost') or alloc.get('cost', 0)
                            time_hours = alloc.get('estimated_time_hours', 'N/A')
                            
                            allocation_data.append({
                                "From": get_hospital_name(str(from_id)) if from_id else "External Source",
                                "To": get_hospital_name(str(to_id)) if to_id else "Unknown",
                                "Quantity": quantity,
                                "Cost": f"${cost:,.2f}" if cost else "N/A",
                                "Time (hrs)": time_hours
                            })
                        
                        if allocation_data:
                            df_alloc = pd.DataFrame(allocation_data)
                            st.dataframe(df_alloc, use_container_width=True, hide_index=True)
                    else:
                        st.info("No transfers in this strategy")
            
            # LLM Explanation
            explanation = state.get("explanation", "")
            if explanation:
                st.subheader("💡 AI Reasoning")
                st.info(explanation)
            
            # Selection
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
            
            feedback = st.text_area(
                "Optional Feedback",
                value="",
                help="Provide feedback to improve future recommendations"
            )
            
            if st.button("✅ Submit & Continue", type="primary"):
                # Update state
                state["user_decision"] = selected_index
                state["user_feedback"] = feedback if feedback.strip() else None
                
                # Process feedback
                with st.spinner("Processing selection..."):
                    try:
                        updated_state = feedback_node(state)
                        config = {"configurable": {"thread_id": state["user_id"]}}
                        final_state = medflow_graph.invoke(updated_state, config=config)
                        
                        # Store in history
                        st.session_state.simulation_history.append({
                            "date": st.session_state.current_date.strftime("%Y-%m-%d"),
                            "outbreak_active": outbreak_status is not None,
                            "selected_strategy": selected_index,
                            "state": final_state
                        })
                        
                        # Build overlay adjustments for next day
                        try:
                            next_day = (st.session_state.current_date + timedelta(days=1)).strftime("%Y-%m-%d")
                            transfers = final_state.get('final_recommendation', {}).get('transfers', [])
                            adjustments = []
                            for t in transfers:
                                rtype = t.get('resource_type', st.session_state.resource_type)
                                qty = float(t.get('quantity', 0))
                                from_id = t.get('from_hospital_id') or t.get('from_hospital') or t.get('from')
                                to_id = t.get('to_hospital_id') or t.get('to_hospital') or t.get('to')
                                if from_id and qty:
                                    adjustments.append({
                                        'hospital_id': str(from_id),
                                        'resource_type': rtype,
                                        'delta_available': -qty
                                    })
                                if to_id and qty:
                                    adjustments.append({
                                        'hospital_id': str(to_id),
                                        'resource_type': rtype,
                                        'delta_available': qty
                                    })
                            if adjustments:
                                st.session_state.inventory_overlays[next_day] = adjustments
                                # Persist to file that backend reads (ml_core/utils/data_loader.py)
                                import json
                                overlay_path = project_root / 'simulation_overlay.json'
                                # Merge existing file if present
                                existing = {}
                                if overlay_path.exists():
                                    try:
                                        existing = json.loads(overlay_path.read_text())
                                    except Exception:
                                        existing = {}
                                existing[next_day] = adjustments
                                overlay_path.write_text(json.dumps(existing, indent=2))
                        except Exception as _:
                            pass

                        # Update learning metrics
                        if final_state.get("preference_profile"):
                            st.session_state.learning_metrics.append({
                                "date": st.session_state.current_date.strftime("%Y-%m-%d"),
                                "preferences": final_state["preference_profile"]
                            })
                        
                        st.session_state.awaiting_review = False
                        st.session_state.current_workflow_state = None
                        st.success("✅ Selection processed! Move to next day.")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.warning("No strategies available")
    
    else:
        # Display day summary if available
        history_entry = next((h for h in st.session_state.simulation_history 
                             if h["date"] == st.session_state.current_date.strftime("%Y-%m-%d")), None)
        
        if history_entry:
            st.success("✅ This day has been simulated")
            state = history_entry["state"]
            final_rec = state.get("final_recommendation", {})
            
            if final_rec:
                summary = final_rec.get('summary', {})
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Strategy", final_rec.get('strategy_name', 'Unknown'))
                with col2:
                    st.metric("Cost", f"${summary.get('total_cost', 0):,.0f}")
                with col3:
                    st.metric("Hospitals Helped", summary.get('hospitals_helped', 0))
                with col4:
                    shortage_reduction = summary.get('shortage_reduction', summary.get('shortage_reduction_percent', 0))
                    st.metric("Shortage Reduction", f"{shortage_reduction:.1f}%")
                
                # Show allocation details for completed day
                allocations = final_rec.get('allocations', [])
                if allocations:
                    with st.expander("📦 View Allocations Made This Day", expanded=False):
                        allocation_data = []
                        total_quantity = 0
                        total_cost = 0
                        
                        for alloc in allocations:
                            from_id = alloc.get('from_hospital_id') or alloc.get('from_hospital') or alloc.get('from')
                            to_id = alloc.get('to_hospital_id') or alloc.get('to_hospital') or alloc.get('to')
                            quantity = alloc.get('quantity', 0)
                            cost = alloc.get('transfer_cost') or alloc.get('cost', 0)
                            time_hours = alloc.get('estimated_time_hours', 'N/A')
                            
                            total_quantity += quantity
                            total_cost += cost if cost else 0
                            
                            allocation_data.append({
                                "From": get_hospital_name(str(from_id)) if from_id else "External Source",
                                "To": get_hospital_name(str(to_id)) if to_id else "Unknown",
                                "Quantity": quantity,
                                "Cost": f"${cost:,.2f}" if cost else "N/A",
                                "Time (hrs)": time_hours
                            })
                        
                        if allocation_data:
                            # Summary metrics
                            col_sum1, col_sum2, col_sum3 = st.columns(3)
                            with col_sum1:
                                st.metric("Total Transfers", len(allocation_data))
                            with col_sum2:
                                st.metric("Total Resources Moved", total_quantity)
                            with col_sum3:
                                st.metric("Total Transfer Cost", f"${total_cost:,.2f}")
                            
                            st.divider()
                            
                            # Detailed table
                            df_alloc = pd.DataFrame(allocation_data)
                            st.dataframe(df_alloc, use_container_width=True, hide_index=True)
                else:
                    st.info("No allocations were made on this day")
        else:
            st.info("👆 Click 'Run Today' to simulate this day")


def render_learning_analytics_tab():
    """Render the Learning Analytics tab"""
    st.header("📈 Learning Analytics")
    
    if len(st.session_state.simulation_history) == 0:
        st.info("No simulation data yet. Run some simulations first!")
        return
    
    # Extract metrics
    dates = [h["date"] for h in st.session_state.simulation_history]
    costs = [h["state"].get("final_recommendation", {}).get("summary", {}).get("total_cost", 0) 
             for h in st.session_state.simulation_history]
    hospitals_helped = [h["state"].get("final_recommendation", {}).get("summary", {}).get("hospitals_helped", 0) 
                        for h in st.session_state.simulation_history]
    selected_strategies = [h["selected_strategy"] for h in st.session_state.simulation_history]
    outbreak_active = [h["outbreak_active"] for h in st.session_state.simulation_history]
    
    # Create dataframe
    df = pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Cost": costs,
        "Hospitals Helped": hospitals_helped,
        "Selected Strategy": selected_strategies,
        "Outbreak Active": outbreak_active
    })
    
    # Cost trend
    st.subheader("💰 Cost Trend Over Time")
    fig_cost = go.Figure()
    fig_cost.add_trace(go.Scatter(
        x=df["Date"], y=df["Cost"],
        mode='lines+markers',
        name='Cost',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8)
    ))
    
    # Add outbreak shading
    for i, row in df.iterrows():
        if row["Outbreak Active"]:
            fig_cost.add_vrect(
                x0=row["Date"], x1=row["Date"] + pd.Timedelta(days=1),
                fillcolor="red", opacity=0.1, line_width=0
            )
    
    fig_cost.update_layout(
        xaxis_title="Date",
        yaxis_title="Total Cost ($)",
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig_cost, use_container_width=True)
    
    # Hospitals helped
    st.subheader("🏥 Hospitals Helped Over Time")
    fig_hospitals = go.Figure()
    fig_hospitals.add_trace(go.Scatter(
        x=df["Date"], y=df["Hospitals Helped"],
        mode='lines+markers',
        name='Hospitals Helped',
        line=dict(color='#2ca02c', width=2),
        marker=dict(size=8)
    ))
    
    # Add outbreak shading
    for i, row in df.iterrows():
        if row["Outbreak Active"]:
            fig_hospitals.add_vrect(
                x0=row["Date"], x1=row["Date"] + pd.Timedelta(days=1),
                fillcolor="red", opacity=0.1, line_width=0
            )
    
    fig_hospitals.update_layout(
        xaxis_title="Date",
        yaxis_title="Hospitals Helped",
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig_hospitals, use_container_width=True)
    
    # Strategy selection distribution
    st.subheader("🎯 Strategy Selection Distribution")
    strategy_names = ["Balanced", "Cost-Efficient", "Maximum Coverage"]
    strategy_counts = [selected_strategies.count(i) for i in range(3)]
    
    fig_strategies = go.Figure(data=[
        go.Bar(
            x=strategy_names,
            y=strategy_counts,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
        )
    ])
    fig_strategies.update_layout(
        xaxis_title="Strategy Type",
        yaxis_title="Times Selected",
        height=400
    )
    st.plotly_chart(fig_strategies, use_container_width=True)
    
    # Allocation History
    st.subheader("📦 Allocation History")
    
    # Collect all allocations across all days
    all_allocations = []
    for history_entry in st.session_state.simulation_history:
        date = history_entry["date"]
        state = history_entry["state"]
        final_rec = state.get("final_recommendation", {})
        allocations = final_rec.get('allocations', [])
        strategy_name = final_rec.get('strategy_name', 'Unknown')
        
        for alloc in allocations:
            from_id = alloc.get('from_hospital_id') or alloc.get('from_hospital') or alloc.get('from')
            to_id = alloc.get('to_hospital_id') or alloc.get('to_hospital') or alloc.get('to')
            quantity = alloc.get('quantity', 0)
            cost = alloc.get('cost') or alloc.get('transfer_cost', 0)
            time_hours = alloc.get('estimated_time_hours', 'N/A')
            
            all_allocations.append({
                "Date": date,
                "Strategy": strategy_name,
                "From": get_hospital_name(str(from_id)) if from_id else "External Source",
                "To": get_hospital_name(str(to_id)) if to_id else "Unknown",
                "Quantity": quantity,
                "Cost": cost if cost else 0,
                "Time (hrs)": time_hours
            })
    
    if all_allocations:
        # Filters
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            date_filter = st.multiselect(
                "Filter by Date",
                options=sorted(set(a["Date"] for a in all_allocations)),
                default=[]
            )
        with col_filter2:
            hospital_filter = st.multiselect(
                "Filter by Hospital (From/To)",
                options=sorted(set(a["From"] for a in all_allocations) | set(a["To"] for a in all_allocations)),
                default=[]
            )
        
        # Apply filters
        filtered_allocations = all_allocations
        if date_filter:
            filtered_allocations = [a for a in filtered_allocations if a["Date"] in date_filter]
        if hospital_filter:
            filtered_allocations = [a for a in filtered_allocations 
                                  if a["From"] in hospital_filter or a["To"] in hospital_filter]
        
        # Summary metrics
        total_transfers = len(filtered_allocations)
        total_quantity = sum(a["Quantity"] for a in filtered_allocations)
        total_cost = sum(a["Cost"] for a in filtered_allocations)
        
        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
        with col_sum1:
            st.metric("Total Transfers", total_transfers)
        with col_sum2:
            st.metric("Total Resources Moved", total_quantity)
        with col_sum3:
            st.metric("Total Cost", f"${total_cost:,.2f}")
        with col_sum4:
            unique_days = len(set(a["Date"] for a in filtered_allocations))
            st.metric("Days with Allocations", unique_days)
        
        st.divider()
        
        # Display table
        if filtered_allocations:
            df_allocations = pd.DataFrame(filtered_allocations)
            # Sort by date descending
            df_allocations["Date"] = pd.to_datetime(df_allocations["Date"])
            df_allocations = df_allocations.sort_values("Date", ascending=False)
            df_allocations["Date"] = df_allocations["Date"].dt.strftime("%Y-%m-%d")
            df_allocations["Cost"] = df_allocations["Cost"].apply(lambda x: f"${x:,.2f}" if x > 0 else "N/A")
            
            st.dataframe(df_allocations, use_container_width=True, hide_index=True)
        else:
            st.info("No allocations match the selected filters")
    else:
        st.info("No allocations recorded yet")
    
    # Summary stats
    st.divider()
    st.subheader("📊 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Days", len(dates))
    with col2:
        st.metric("Avg Cost", f"${sum(costs)/len(costs):,.0f}")
    with col3:
        st.metric("Total Hospitals Helped", sum(hospitals_helped))
    with col4:
        outbreak_days = sum(outbreak_active)
        st.metric("Outbreak Days", outbreak_days)


def render_traces_tab():
    """Render the LangSmith Traces tab"""
    st.header("🔍 LangSmith Traces")
    
    langsmith_project = os.getenv("LANGCHAIN_PROJECT", "medflow-ai")
    langsmith_endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    
    if not os.getenv("LANGCHAIN_API_KEY"):
        st.warning("⚠️ LangSmith API key not configured. Set LANGCHAIN_API_KEY in .env file.")
        return
    
    st.info(f"📊 **Project:** {langsmith_project}")
    st.markdown(f"🔗 [View in LangSmith]({langsmith_endpoint}/o/default/projects/p/{langsmith_project})")
    
    st.divider()
    
    # Recent runs
    st.subheader("Recent Workflow Runs")
    
    if len(st.session_state.simulation_history) > 0:
        for i, entry in enumerate(reversed(st.session_state.simulation_history[-5:])):
            with st.expander(f"Run {len(st.session_state.simulation_history) - i}: {entry['date']}"):
                state = entry["state"]
                st.write(f"**Session ID:** {state.get('session_id', 'N/A')}")
                st.write(f"**Status:** {state.get('workflow_status', 'N/A')}")
                st.write(f"**Execution Time:** {state.get('execution_time_seconds', 0):.1f}s")
                
                if state.get("error"):
                    st.error(f"Error: {state['error']}")
    else:
        st.info("No workflow runs yet")


# === MAIN APP ===

def main():
    """Main Streamlit app"""
    
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🏥 MedFlow AI")
        st.markdown("**Demo Dashboard**")
        framework_name = "CrewAI" if st.session_state.get("framework") == "crewai" else "LangGraph"
        st.markdown(f"Powered by {framework_name} • Phase 5")
        st.divider()
        
        st.info("👈 Use tabs to navigate:\n- **Setup**: Configure demo\n- **Daily**: Run simulations\n- **Analytics**: View learning\n- **Traces**: LangSmith logs")
    
    # Main content - Tabbed interface
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚙️ Setup",
        "📅 Daily Simulation",
        "📈 Learning Analytics",
        "🔍 Traces"
    ])
    
    with tab1:
        render_setup_tab()
    
    with tab2:
        render_daily_simulation_tab()
    
    with tab3:
        render_learning_analytics_tab()
    
    with tab4:
        render_traces_tab()


if __name__ == "__main__":
    main()

