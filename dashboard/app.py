"""
MedFlow Video Demo Dashboard

Comprehensive Streamlit interface for video recording showcasing:
- Hospital capacity dashboard
- Predictive intelligence (LSTM/RF)
- Agent reasoning (workflow, Qdrant, LP optimization)
- Adaptive learning
- Framework comparison (LangGraph vs CrewAI)
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta, date
import json
from typing import Optional, Dict, Any, List

# Add project root to path
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

# Verify agents module
agents_path = project_root / "agents"
if not agents_path.exists():
    found = False
    for path_str in sys.path:
        path_obj = Path(path_str)
        if (path_obj / "agents").exists():
            found = True
            break
    if not found:
        raise ImportError(f"Cannot find agents module. Project root: {project_root}")

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
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
    from agents.crewai.crew.medflow_crew import run_workflow_until_review as crewai_run_until_review
    from agents.crewai.crew.medflow_crew import continue_workflow_after_review as crewai_continue_workflow
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

# Qdrant import (optional)
try:
    from ml_core.utils.qdrant_client import InteractionVectorStore
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# Load environment variables
load_dotenv(override=True)

# Configure page
st.set_page_config(
    page_title="MedFlow",
    page_icon="",
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
    defaults = {
        "resource_type": "ventilators",
        "user_id": "demo_user",
        "framework": "langgraph",
        "simulation_date": None,
        "selected_outbreak": "TB Outbreak",
        "workflow_state": None,
        "workflow_running": False,
        "awaiting_review": False,
        "current_workflow_state": None,
        "hospital_cache": {},
        "capacity_data": None,
        "forecast_data": None,
        "workflow_results": {},
        "simulation_history": [],
        "current_date": date(2024, 6, 15)  # Default to outbreak start date
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_hospital_name(hospital_id: str) -> str:
    """Get hospital name from ID, using cache"""
    if not hospital_id:
        return "Unknown"
    
    if hospital_id in st.session_state.hospital_cache:
        return st.session_state.hospital_cache[hospital_id]
    
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
    
    return hospital_id[:8] + "..."

@st.cache_data(ttl=300)
def get_hospital_capacity_data(resource_type: str, simulation_date: Optional[str] = None) -> Dict:
    """Fetch and aggregate hospital capacity data"""
    try:
        # Get shortages and outbreaks
        shortages_result = api_client.get_shortages(
            resource_type=resource_type,
            simulation_date=simulation_date
        )
        
        outbreaks_result = api_client.get_active_outbreaks(simulation_date=simulation_date)
        
        shortage_hospitals = shortages_result.get("shortages", [])
        active_outbreaks = outbreaks_result.get("outbreaks", [])
        
        # Aggregate by region
        regional_data = {}
        for shortage in shortage_hospitals:
            region = shortage.get("region", "Unknown")
            if region not in regional_data:
                regional_data[region] = {
                    "shortages": 0,
                    "surplus": 0,
                    "hospitals": []
                }
            regional_data[region]["shortages"] += 1
            regional_data[region]["hospitals"].append(shortage)
        
        return {
            "shortage_hospitals": shortage_hospitals,
            "active_outbreaks": active_outbreaks,
            "regional_data": regional_data,
            "total_shortages": len(shortage_hospitals),
            "affected_regions": list(regional_data.keys())
        }
    except Exception as e:
        logger.error(f"Error fetching capacity data: {e}")
        return {
            "shortage_hospitals": [],
            "active_outbreaks": [],
            "regional_data": {},
            "total_shortages": 0,
            "affected_regions": []
        }

@st.cache_data(ttl=300)
def get_forecast_data(hospital_ids: List[str], resource_type: str, simulation_date: Optional[str] = None) -> pd.DataFrame:
    """Fetch LSTM forecasts for multiple hospitals"""
    forecasts = []
    
    for hospital_id in hospital_ids[:5]:  # Limit to 5 for performance
        try:
            result = api_client.predict_demand(
                hospital_id=hospital_id,
                resource_type=resource_type,
                days_ahead=14,
                simulation_date=simulation_date
            )
            
            # Updated to match actual API response structure
            predictions = result.get("predictions", {})
            if not predictions:
                logger.warning(f"No predictions data for {hospital_id}")
                continue
            
            predicted_demand = predictions.get("predicted_demand", [])
            forecast_dates = predictions.get("forecast_dates", [])
            
            # Process forecast data (14-day predictions)
            for i, pred in enumerate(predicted_demand):
                if i < len(forecast_dates):
                    forecasts.append({
                        "hospital_id": hospital_id,
                        "hospital_name": get_hospital_name(hospital_id),
                        "date": forecast_dates[i],
                        "consumption": pred,
                        "type": "forecast"
                    })
            
            # Add current stock as a data point
            current_stock = predictions.get("current_stock")
            if current_stock is not None and forecast_dates:
                forecasts.append({
                    "hospital_id": hospital_id,
                    "hospital_name": get_hospital_name(hospital_id),
                    "date": forecast_dates[0] if forecast_dates else None,
                    "consumption": current_stock,
                    "type": "historical"
                })
                
        except Exception as e:
            logger.warning(f"Error fetching forecast for {hospital_id}: {e}")
            logger.exception(e)
    
    return pd.DataFrame(forecasts)

def get_qdrant_similar_cases(interaction: Dict, user_id: Optional[str] = None, limit: int = 5) -> List[Dict]:
    """Query Qdrant for similar past cases"""
    if not QDRANT_AVAILABLE:
        return []
    
    try:
        store = InteractionVectorStore()
        similar = store.find_similar_interactions(interaction, user_id=user_id, limit=limit)
        return similar
    except Exception as e:
        logger.warning(f"Error querying Qdrant: {e}")
        return []

def safe_get_state(state: Any, key: str, default: Any = None) -> Any:
    """Safely get value from state (handles both dict and CrewOutput objects)"""
    if isinstance(state, dict):
        return state.get(key, default)
    else:
        # CrewAI CrewOutput object - use getattr
        return getattr(state, key, default)

def visualize_workflow_graph(state: Any) -> go.Figure:
    """Create workflow visualization (supports both LangGraph and CrewAI)"""
    # Define nodes and edges
    nodes = [
        "Data Analyst",
        "Forecasting",
        "Optimization",
        "Preference",
        "Reasoning",
        "Human Review",
        "Feedback"
    ]
    
    # Determine node states from workflow state
    node_states = {}
    current_node = safe_get_state(state, "current_node", None)
    workflow_status = safe_get_state(state, "workflow_status", "pending")
    
    # For CrewAI, check if we have tasks completed
    if hasattr(state, 'tasks') and state.tasks:
        # Count completed tasks
        completed_count = sum(1 for task in state.tasks if hasattr(task, 'status') and task.status == 'completed')
        workflow_status = "completed" if completed_count == len(state.tasks) else "in_progress"
    
    # Color mapping: Green (completed), Yellow (in progress), Gray (pending)
    if workflow_status == "completed":
        node_states = {node: "completed" for node in nodes}
    elif current_node:
        # Mark nodes up to current as completed
        node_index = {
            "data_analyst": 0,
            "forecasting": 1,
            "optimization": 2,
            "preference": 3,
            "reasoning": 4,
            "human_review": 5,
            "feedback": 6
        }.get(current_node, -1)
        
        for i, node in enumerate(nodes):
            if i <= node_index:
                node_states[node] = "completed"
            elif i == node_index + 1:
                node_states[node] = "in_progress"
            else:
                node_states[node] = "pending"
    else:
        node_states = {node: "pending" for node in nodes}
    
    # Create Sankey diagram
    colors = {
        "completed": "#2ca02c",
        "in_progress": "#ff7f0e",
        "pending": "#d3d3d3"
    }
    
    # Simple flow diagram using annotations
    fig = go.Figure()
    
    # Add nodes as scatter points
    y_positions = np.linspace(1, 0, len(nodes))
    for i, node in enumerate(nodes):
        state_color = colors.get(node_states.get(node, "pending"), "#d3d3d3")
        fig.add_trace(go.Scatter(
            x=[i % 2],
            y=[y_positions[i]],
            mode='markers+text',
            marker=dict(size=30, color=state_color),
            text=node,
            textposition="middle right" if i % 2 == 0 else "middle left",
            name=node,
            showlegend=False
        ))
    
    # Add edges (arrows)
    for i in range(len(nodes) - 1):
        x0 = (i % 2) + 0.1
        x1 = ((i + 1) % 2) - 0.1
        y0 = y_positions[i]
        y1 = y_positions[i + 1]
        
        fig.add_annotation(
            x=x1,
            y=y1,
            ax=x0,
            ay=y0,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#1f77b4"
        )
    
    # Determine title based on framework
    title = "Workflow Execution"
    if hasattr(state, 'tasks'):
        title = "CrewAI Workflow Execution"
    else:
        title = "LangGraph Workflow Execution"
    
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, showticklabels=False, range=[-0.5, 1.5]),
        yaxis=dict(showgrid=False, showticklabels=False, range=[-0.1, 1.1]),
        height=500,
        showlegend=False
    )
    
    return fig

# === TAB RENDER FUNCTIONS ===

def render_capacity_dashboard():
    """Tab 1: Hospital Capacity Dashboard"""
    st.header("🏥 Hospital Capacity Dashboard")
    
    resource_type = st.session_state.resource_type
    simulation_date = st.session_state.simulation_date
    
    with st.spinner("Loading capacity data..."):
        capacity_data = get_hospital_capacity_data(resource_type, simulation_date)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Shortages", capacity_data["total_shortages"])
    with col2:
        st.metric("Affected Regions", len(capacity_data["affected_regions"]))
    with col3:
        st.metric("Active Outbreaks", len(capacity_data["active_outbreaks"]))
    with col4:
        critical_count = sum(1 for h in capacity_data["shortage_hospitals"] 
                           if h.get("risk_level") == "critical")
        st.metric("Critical Alerts", critical_count)
    
    st.divider()
    
    # Hospital Capacity Distribution
    st.subheader("📈 Hospital Capacity Distribution")
    
    if capacity_data["shortage_hospitals"]:
        hospital_df = pd.DataFrame(capacity_data["shortage_hospitals"])
        hospital_df["hospital_name"] = hospital_df["hospital_id"].apply(get_hospital_name)
        
        # Calculate quantity_needed if not present (similar to ml_core logic)
        if "quantity_needed" not in hospital_df.columns:
            if "predicted_demand_7d" in hospital_df.columns and "current_stock" in hospital_df.columns:
                hospital_df["quantity_needed"] = (
                    hospital_df["predicted_demand_7d"] - hospital_df["current_stock"]
                ).clip(lower=1)
            elif "days_of_supply" in hospital_df.columns and "current_stock" in hospital_df.columns:
                hospital_df["quantity_needed"] = hospital_df.apply(
                    lambda row: max(1, int(row["current_stock"] * 
                                          (7 / max(row.get("days_of_supply", 7), 0.1)) - 
                                          row["current_stock"])),
                    axis=1
                )
            else:
                # Fallback: use predicted_demand_7d or current_stock as proxy
                if "predicted_demand_7d" in hospital_df.columns:
                    hospital_df["quantity_needed"] = hospital_df["predicted_demand_7d"]
                elif "current_stock" in hospital_df.columns:
                    # Estimate based on risk level
                    hospital_df["quantity_needed"] = hospital_df.apply(
                        lambda row: max(1, int(row["current_stock"] * 
                                              (0.5 if row.get("risk_level") == "critical" else 0.3))),
                        axis=1
                    )
                else:
                    hospital_df["quantity_needed"] = 1  # Default fallback
        
        # Sort by quantity_needed (or fallback to risk_level)
        sort_column = "quantity_needed" if "quantity_needed" in hospital_df.columns else "risk_level"
        hospital_df = hospital_df.sort_values(sort_column, ascending=False).head(20)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=hospital_df["hospital_name"],
            y=hospital_df["quantity_needed"] if "quantity_needed" in hospital_df.columns else [1] * len(hospital_df),
            marker_color=hospital_df["risk_level"].map({
                "critical": "#d62728",
                "high": "#ff7f0e",
                "medium": "#ffbb78",
                "low": "#2ca02c"
            }).fillna("#d3d3d3"),
            text=hospital_df["quantity_needed"] if "quantity_needed" in hospital_df.columns else hospital_df.get("risk_level", ""),
            textposition="outside"
        ))
        
        fig.update_layout(
            title="Top 20 Hospitals by Shortage Severity",
            xaxis_title="Hospital",
            yaxis_title="Quantity Needed" if "quantity_needed" in hospital_df.columns else "Risk Level",
            height=500,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No shortage data available")

def render_predictive_intelligence():
    """Tab 2: Predictive Intelligence"""
    st.header("🔮 Predictive Intelligence")
    
    resource_type = st.session_state.resource_type
    simulation_date = st.session_state.simulation_date
    
    # Get shortage hospitals for forecasting
    with st.spinner("Loading shortage data..."):
        capacity_data = get_hospital_capacity_data(resource_type, simulation_date)
    
    shortage_hospitals = capacity_data["shortage_hospitals"]
    
    if not shortage_hospitals:
        st.info("No shortages detected. Run analysis first.")
        return
    
    # Select hospitals to forecast (top 5 by severity)
    # Calculate quantity_needed for sorting if not present
    def get_sort_key(h):
        if "quantity_needed" in h:
            return h["quantity_needed"]
        elif "predicted_demand_7d" in h and "current_stock" in h:
            return max(1, h["predicted_demand_7d"] - h["current_stock"])
        elif "days_of_supply" in h and "current_stock" in h:
            return max(1, int(h["current_stock"] * (7 / max(h.get("days_of_supply", 7), 0.1)) - h["current_stock"]))
        else:
            # Fallback to risk level priority
            risk_priority = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            return risk_priority.get(h.get("risk_level", "low"), 0)
    
    hospital_ids = [h["hospital_id"] for h in sorted(
        shortage_hospitals,
        key=get_sort_key,
        reverse=True
    )[:5]]
    
    # LSTM Demand Forecasts
    st.subheader("📈 LSTM Demand Forecasts (14-Day)")
    
    with st.spinner("Fetching forecasts..."):
        forecast_df = get_forecast_data(hospital_ids, resource_type, simulation_date)
    
    if not forecast_df.empty:
        fig = go.Figure()
        
        for hospital_id in forecast_df["hospital_id"].unique():
            hospital_data = forecast_df[forecast_df["hospital_id"] == hospital_id]
            hospital_name = hospital_data["hospital_name"].iloc[0]
            
            historical = hospital_data[hospital_data["type"] == "historical"]
            forecast = hospital_data[hospital_data["type"] == "forecast"]
            
            if not historical.empty:
                fig.add_trace(go.Scatter(
                    x=historical["date"],
                    y=historical["consumption"],
                    mode='lines+markers',
                    name=f"{hospital_name} (Historical)",
                    line=dict(dash='solid', width=2)
                ))
            
            if not forecast.empty:
                fig.add_trace(go.Scatter(
                    x=forecast["date"],
                    y=forecast["consumption"],
                    mode='lines+markers',
                    name=f"{hospital_name} (Forecast)",
                    line=dict(dash='dash', width=2)
                ))
        
        fig.update_layout(
            title="14-Day Demand Forecast",
            xaxis_title="Date",
            yaxis_title="Consumption",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No forecast data available")
    
    # Forecast Accuracy Metrics
    st.subheader("📊 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    # MAE values from README
    mae_data = {
        "PPE": 4.65,
        "O2 Cylinders": 2.03,
        "Ventilators": 1.02,
        "Medications": 3.5,  # Estimated
        "Beds": 2.5  # Estimated
    }
    
    resource_display = {
        "ppe": "PPE",
        "o2_cylinders": "O2 Cylinders",
        "ventilators": "Ventilators",
        "medications": "Medications",
        "beds": "Beds"
    }
    
    with col1:
        current_mae = mae_data.get(resource_display.get(resource_type, "Ventilators"), 1.02)
        st.metric("LSTM MAE", f"{current_mae:.2f} units")
    
    with col2:
        st.metric("Forecast Coverage", "~75-80%")

    
    # MAE Bar Chart
    fig = go.Figure(data=go.Bar(
        x=list(mae_data.keys()),
        y=list(mae_data.values()),
        marker_color='#1f77b4',
        text=[f"{v:.2f}" for v in mae_data.values()],
        textposition="outside"
    ))
    
    fig.update_layout(
        title="Mean Absolute Error (MAE) by Resource Type",
        xaxis_title="Resource Type",
        yaxis_title="MAE (units)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Random Forest Shortage Predictions
    st.subheader("🌳 Random Forest Shortage Predictions")
    
    if shortage_hospitals:
        rf_df = pd.DataFrame(shortage_hospitals)
        rf_df["hospital_name"] = rf_df["hospital_id"].apply(get_hospital_name)
        
        # Calculate quantity_needed if not present
        if "quantity_needed" not in rf_df.columns:
            if "predicted_demand_7d" in rf_df.columns and "current_stock" in rf_df.columns:
                rf_df["quantity_needed"] = (rf_df["predicted_demand_7d"] - rf_df["current_stock"]).clip(lower=0)
            else:
                rf_df["quantity_needed"] = 0  # Fallback
        
        # Select available columns for display
        available_cols = ["hospital_name"]
        if "current_stock" in rf_df.columns:
            available_cols.append("current_stock")
        if "quantity_needed" in rf_df.columns:
            available_cols.append("quantity_needed")
        if "risk_level" in rf_df.columns:
            available_cols.append("risk_level")
        
        display_df = rf_df[available_cols].copy()
        column_mapping = {
            "hospital_name": "Hospital",
            "current_stock": "Current Stock",
            "quantity_needed": "Quantity Needed",
            "risk_level": "Risk Level"
        }
        display_df.columns = [column_mapping.get(col, col.replace("_", " ").title()) for col in available_cols]
        
        # Apply color coding if Risk Level column exists
        if "Risk Level" in display_df.columns:
            def color_risk_level(row):
                risk = str(row.get("Risk Level", "")).lower()
                if risk == "critical":
                    return ['background-color: #d62728'] * len(row)
                elif risk == "high":
                    return ['background-color: #ff7f0e'] * len(row)
                elif risk == "medium":
                    return ['background-color: #ffbb78'] * len(row)
                elif risk == "low":
                    return ['background-color: #2ca02c'] * len(row)
                else:
                    return [''] * len(row)
            
            st.dataframe(
                display_df.style.apply(color_risk_level, axis=1),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.dataframe(display_df, use_container_width=True, hide_index=True)

def render_agent_reasoning():
    """Tab 3: Agent Reasoning"""
    st.header("🤖 Agent Reasoning & Workflow")
    
    resource_type = st.session_state.resource_type
    user_id = st.session_state.user_id
    simulation_date = st.session_state.simulation_date
    framework = st.session_state.framework
    
    # Run workflow button
    if st.button("🚀 Run Workflow", type="primary"):
        st.session_state.workflow_running = True
        st.session_state.awaiting_review = False
        st.rerun()
    
    if st.session_state.workflow_running:
        with st.spinner("Running workflow... This may take 30-60 seconds."):
            # Initialize state
            initial_state = {
                "resource_type": resource_type,
                "user_id": user_id,
                "session_id": str(uuid.uuid4()),
                "simulation_date": simulation_date,
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
            
            try:
                if framework == "crewai" and CREWAI_AVAILABLE:
                    workflow_result = crewai_run_until_review(
                        resource_type=resource_type,
                        user_id=user_id,
                        simulation_date=simulation_date
                    )
                    
                    if workflow_result.get("success"):
                        crew_result = workflow_result.get("result")
                        tasks = workflow_result.get("tasks", [])
                        
                        # Extract data from CrewAI tasks to create a dict-like state
                        state = {}
                        if crew_result:
                            # Try to get data from CrewOutput object
                            if hasattr(crew_result, 'raw'):
                                try:
                                    if isinstance(crew_result.raw, str):
                                        state = json.loads(crew_result.raw)
                                    elif isinstance(crew_result.raw, dict):
                                        state = crew_result.raw
                                except:
                                    pass
                            
                            # Also try to extract from tasks
                            for task in tasks:
                                if hasattr(task, 'output') and task.output:
                                    try:
                                        task_output = task.output.raw if hasattr(task.output, 'raw') else str(task.output)
                                        if isinstance(task_output, str):
                                            try:
                                                parsed = json.loads(task_output)
                                                if isinstance(parsed, dict):
                                                    state.update(parsed)
                                            except:
                                                pass
                                    except:
                                        pass
                        
                        # Store both the extracted state and original result
                        st.session_state.current_workflow_state = state if state else crew_result
                        st.session_state.workflow_running = False
                        st.session_state.awaiting_review = True
                    else:
                        st.error(f"Workflow failed: {workflow_result.get('error')}")
                        st.session_state.workflow_running = False
                else:
                    # LangGraph
                    config = {"configurable": {"thread_id": user_id}}
                    state = medflow_graph.invoke(initial_state, config=config)
                    
                    if state.get("user_decision") is None and state.get("ranked_strategies"):
                        st.session_state.current_workflow_state = state
                        st.session_state.workflow_running = False
                        st.session_state.awaiting_review = True
                    else:
                        st.session_state.workflow_state = state
                        st.session_state.workflow_running = False
                
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.workflow_running = False
                logger.exception("Workflow error")
    
    # Display workflow state if available
    state = st.session_state.current_workflow_state or st.session_state.workflow_state
    
    if state:
        # LangGraph Workflow Visualization
        st.subheader("🔄 Workflow Execution Graph")
        workflow_fig = visualize_workflow_graph(state)
        st.plotly_chart(workflow_fig, use_container_width=True)
        
        # Agent Execution Logs
        st.subheader("📋 Agent Execution Logs")
        
        nodes_info = [
            ("Data Analyst", "shortage_count", "analysis_summary"),
            ("Forecasting", "demand_forecasts", "forecast_summary"),
            ("Optimization", "allocation_strategies", "strategy_count"),
            ("Preference", "ranked_strategies", "preference_profile"),
            ("Reasoning", "explanation", "reasoning_chain"),
        ]
        
        for node_name, data_key, summary_key in nodes_info:
            with st.expander(f"✅ {node_name}"):
                data_value = safe_get_state(state, data_key)
                summary_value = safe_get_state(state, summary_key)
                
                if data_value:
                    if isinstance(data_value, list):
                        st.write(f"**Output:** {len(data_value)} items")
                    else:
                        st.write(f"**Output:** {data_value}")
                
                if summary_value:
                    st.write(f"**Summary:** {summary_value}")
        
        # Qdrant Vector Retrieval
        st.subheader("🔍 Similar Past Cases (Qdrant)")
        
        ranked_strategies = safe_get_state(state, "ranked_strategies", [])
        if ranked_strategies:
            # Create interaction dict for Qdrant query
            interaction = {
                "selected_recommendation_index": 0,
                "recommendations": ranked_strategies[:3] if isinstance(ranked_strategies, list) else [],
                "timestamp": datetime.now().isoformat(),
                "context": {
                    "resource_type": resource_type,
                    "simulation_date": simulation_date
                }
            }
            
            similar_cases = get_qdrant_similar_cases(interaction, user_id=user_id, limit=5)
            
            if similar_cases:
                st.success(f"Found {len(similar_cases)} similar cases from history")
                
                similar_df = pd.DataFrame([
                    {
                        "Similarity": f"{s['score']:.3f}",
                        "Date": s['payload'].get('timestamp', 'Unknown')[:10],
                        "Strategy": s['payload'].get('strategy_name', 'Unknown'),
                        "Cost": f"${s['payload'].get('total_cost', 0):,.0f}",
                        "Hospitals Helped": s['payload'].get('hospitals_helped', 0)
                    }
                    for s in similar_cases
                ])
                
                st.dataframe(similar_df, use_container_width=True, hide_index=True)
            else:
                st.info("No similar cases found in Qdrant (may be empty or not configured)")
        
        # LP Optimization Details
        st.subheader("⚙️ Linear Programming Optimization")
        
        allocation_strategies = safe_get_state(state, "allocation_strategies", [])
        if allocation_strategies:
            strategy = allocation_strategies[0] if allocation_strategies else {}
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Status", strategy.get("status", "Unknown").title() if isinstance(strategy, dict) else "Completed")
            with col2:
                st.metric("Strategies Generated", safe_get_state(state, "strategy_count", len(allocation_strategies)))
            with col3:
                exec_time = safe_get_state(state, "execution_time_seconds")
                if exec_time:
                    st.metric("Solve Time", f"{exec_time:.2f}s")
            
            st.write("**Objective Function:** Minimize total cost + shortage penalty + maximize coverage")
            st.write("**Constraints:** Supply limits, demand requirements, transfer capacity")
        
        # Explainable Recommendations
        st.subheader("💡 Explainable Recommendations")
        
        ranked_strategies = safe_get_state(state, "ranked_strategies", [])
        
        if ranked_strategies:
            for i, strategy in enumerate(ranked_strategies[:3]):
                with st.expander(f"**{i+1}. {strategy.get('strategy_name', 'Unknown')}**", expanded=(i==0)):
                    summary = strategy.get('summary', {})
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Cost", f"${summary.get('total_cost', 0):,.0f}")
                    with col2:
                        st.metric("Hospitals Helped", summary.get('hospitals_helped', 0))
                    with col3:
                        shortage_reduction = summary.get('shortage_reduction_percent', 
                                                         summary.get('shortage_reduction', 0))
                        st.metric("Shortage Reduction", f"{shortage_reduction:.1f}%")
                    with col4:
                        st.metric("Preference Score", f"{strategy.get('preference_score', 0):.3f}")
                    
                    # Show allocations
                    allocations = strategy.get('allocations', [])
                    if allocations:
                        st.write("**Transfers:**")
                        alloc_df = pd.DataFrame([
                            {
                                "From": get_hospital_name(a.get('from_hospital_id', '')),
                                "To": get_hospital_name(a.get('to_hospital_id', '')),
                                "Quantity": a.get('quantity', 0),
                                "Cost": f"${a.get('transfer_cost', a.get('cost', 0)):,.2f}"
                            }
                            for a in allocations[:10]  # Show first 10
                        ])
                        st.dataframe(alloc_df, use_container_width=True, hide_index=True)
            
            # AI Explanation
            explanation = safe_get_state(state, "explanation", "")
            if explanation:
                st.subheader("🤖 AI Reasoning")
                st.info(explanation)
        
        # Transfer Network Visualization
        if ranked_strategies and ranked_strategies[0].get('allocations'):
            st.subheader("🕸️ Transfer Network")
            
            top_strategy = ranked_strategies[0]
            allocations = top_strategy.get('allocations', [])
            
            # Create network graph
            hospitals = set()
            for alloc in allocations:
                hospitals.add(alloc.get('from_hospital_id'))
                hospitals.add(alloc.get('to_hospital_id'))
            
            # Simple visualization using Sankey diagram
            source = []
            target = []
            value = []
            labels = []
            hospital_map = {}
            
            for i, hospital_id in enumerate(sorted(hospitals)):
                hospital_map[hospital_id] = i
                labels.append(get_hospital_name(hospital_id))
            
            for alloc in allocations[:20]:  # Limit to 20 for performance
                from_id = alloc.get('from_hospital_id')
                to_id = alloc.get('to_hospital_id')
                qty = alloc.get('quantity', 0)
                
                if from_id in hospital_map and to_id in hospital_map:
                    source.append(hospital_map[from_id])
                    target.append(hospital_map[to_id])
                    value.append(qty)
            
            if source and target:
                fig = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=labels,
                        color="#1f77b4"
                    ),
                    link=dict(
                        source=source,
                        target=target,
                        value=value,
                        color="rgba(31, 119, 180, 0.4)"
                    )
                )])
                
                fig.update_layout(title="Resource Transfer Network", height=600)
                st.plotly_chart(fig, use_container_width=True)
        
        # Interactive Strategy Selection
        if ranked_strategies and st.session_state.awaiting_review:
            st.divider()
            st.subheader("🎯 Select Strategy to Execute")
            
            # Create strategy options for radio buttons
            strategy_options = []
            for i, strategy in enumerate(ranked_strategies[:3]):
                summary = strategy.get('summary', {})
                shortage_reduction = summary.get('shortage_reduction_percent', 
                                                 summary.get('shortage_reduction', 0))
                option_text = (
                    f"{i}: {strategy.get('strategy_name', 'Unknown')} - "
                    f"${summary.get('total_cost', 0):,.0f} | "
                    f"{summary.get('hospitals_helped', 0)} hospitals | "
                    f"{shortage_reduction:.1f}% reduction"
                )
                strategy_options.append(option_text)
            
            selected_option = st.radio(
                "Choose a strategy:",
                strategy_options,
                index=0
            )
            
            selected_index = int(selected_option.split(":")[0])
            
            # Optional feedback
            feedback_text = st.text_area(
                "Optional Feedback",
                value="",
                help="Provide feedback to help improve future recommendations",
                height=100
            )
            
            # Execute button
            if st.button("✅ Execute Strategy & Continue", type="primary", use_container_width=True):
                with st.spinner("Executing strategy and updating preferences..."):
                    try:
                        success = execute_strategy_selection(
                            selected_index=selected_index,
                            feedback_text=feedback_text if feedback_text.strip() else None,
                            ranked_strategies=ranked_strategies,
                            state=state
                        )
                        
                        if success:
                            st.success("✅ Strategy executed! Preferences updated. Ready for next day.")
                            st.session_state.awaiting_review = False
                            st.session_state.workflow_running = False
                            # Optionally auto-advance to next day
                            st.info("💡 Click 'Next Day' in sidebar to continue simulation")
                            st.rerun()
                        else:
                            st.error("Failed to execute strategy. Please try again.")
                    except Exception as e:
                        st.error(f"Error executing strategy: {str(e)}")
                        logger.exception("Error in strategy execution")

def execute_strategy_selection(
    selected_index: int,
    feedback_text: Optional[str],
    ranked_strategies: List[Dict],
    state: MedFlowState
) -> bool:
    """Execute user's strategy selection and update preferences"""
    try:
        # Get selected strategy
        if selected_index < 0 or selected_index >= len(ranked_strategies):
            logger.error(f"Invalid strategy index: {selected_index}")
            return False
        
        selected_strategy = ranked_strategies[selected_index]
        
        # Build interaction for backend (matching feedback_node format)
        # Get session_id from state or generate new one
        session_id = safe_get_state(state, "session_id", None)
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        interaction = {
            "selected_recommendation_index": selected_index,
            "recommendations": ranked_strategies,
            "timestamp": datetime.now().isoformat(),
            "feedback_text": feedback_text,
            "context": {
                "resource_type": st.session_state.resource_type,
                "simulation_date": st.session_state.simulation_date,
                "shortage_count": safe_get_state(state, "shortage_count", 0),
                "session_id": session_id
            }
        }
        
        # Call feedback workflow using LangGraph feedback_node
        # This will call update_preferences API internally
        # Ensure state is a dict and has all required fields
        if isinstance(state, dict):
            updated_state = state.copy()
        else:
            # If state is not a dict (e.g., CrewAI CrewOutput), create a new dict
            updated_state = {}
            # Try to extract values using safe_get_state
            updated_state["resource_type"] = safe_get_state(state, "resource_type", st.session_state.resource_type)
            updated_state["user_id"] = safe_get_state(state, "user_id", st.session_state.user_id)
            updated_state["ranked_strategies"] = safe_get_state(state, "ranked_strategies", ranked_strategies)
            updated_state["shortage_count"] = safe_get_state(state, "shortage_count", 0)
        
        # Add/update required fields
        updated_state["user_decision"] = selected_index
        updated_state["user_feedback"] = feedback_text
        updated_state["ranked_strategies"] = ranked_strategies
        
        # Ensure session_id exists (required by feedback_node)
        if "session_id" not in updated_state:
            updated_state["session_id"] = str(uuid.uuid4())
        
        # Ensure all other required fields exist
        if "resource_type" not in updated_state:
            updated_state["resource_type"] = st.session_state.resource_type
        if "user_id" not in updated_state:
            updated_state["user_id"] = st.session_state.user_id
        if "shortage_count" not in updated_state:
            updated_state["shortage_count"] = 0
        
        # Call feedback node (this calls update_preferences API)
        from agents.nodes.feedback import feedback_node
        final_state = feedback_node(updated_state)
        
        # Get preference profile from current state (set by preference_node)
        # This represents the weights used for ranking BEFORE this selection
        # The updated weights will be reflected in the NEXT workflow run
        preference_profile = safe_get_state(state, "preference_profile", {})
        
        # Store in Qdrant (if available)
        if QDRANT_AVAILABLE:
            try:
                store = InteractionVectorStore()
                store.store_interaction(st.session_state.user_id, interaction)
            except Exception as e:
                logger.warning(f"Could not store in Qdrant: {e}")
        
        # Save to session history
        current_date_obj = st.session_state.current_date
        if isinstance(current_date_obj, str):
            date_str = current_date_obj
        elif current_date_obj is not None:
            date_str = current_date_obj.strftime("%Y-%m-%d")
        else:
            date_str = st.session_state.simulation_date or "2024-06-15"
        
        history_entry = {
            "date": date_str,
            "resource_type": st.session_state.resource_type,
            "user_id": st.session_state.user_id,
            "strategies": ranked_strategies,
            "ranked_strategies": ranked_strategies,
            "selected_strategy_index": selected_index,
            "selected_strategy_name": selected_strategy.get("strategy_name", "Unknown"),
            "selected_strategy": selected_strategy,
            "preference_profile": preference_profile,
            "feedback_text": feedback_text,
            "workflow_state": state,
            "timestamp": datetime.now().isoformat()
        }
        
        if "simulation_history" not in st.session_state:
            st.session_state.simulation_history = []
        
        st.session_state.simulation_history.append(history_entry)
        
        logger.info(f"Strategy {selected_index} executed and saved to history")
        return True
        
    except Exception as e:
        logger.error(f"Error executing strategy selection: {e}", exc_info=True)
        return False

def render_adaptive_learning():
    """Tab 4: Adaptive Learning - REAL Data"""
    st.header("🧠 Adaptive Learning")
    
    history = st.session_state.get("simulation_history", [])
    
    if not history:
        st.info("📊 Run workflows and select strategies to see learning progression!")
        st.write("The system will track your selections and show how it learns your preferences over time.")
        st.write("")
        st.write("**How to use:**")
        st.write("1. Go to 'Agent Reasoning' tab")
        st.write("2. Run a workflow")
        st.write("3. Select a strategy to execute")
        st.write("4. Return here to see the learning progression")
        return
    
    # Session Overview Metrics
    st.subheader("📈 Session Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Days Simulated", len(history))
    with col2:
        st.metric("Strategies Selected", len(history))
    with col3:
        strategy_names = [h.get("selected_strategy_name", "Unknown") for h in history]
        if strategy_names:
            most_common = max(set(strategy_names), key=strategy_names.count)
            st.metric("Most Selected", most_common)
        else:
            st.metric("Most Selected", "N/A")
    with col4:
        # Try to get Qdrant count
        if QDRANT_AVAILABLE:
            try:
                store = InteractionVectorStore()
                count = store.get_user_interaction_count(st.session_state.user_id)
                st.metric("Total Interactions", count)
            except:
                st.metric("Session Selections", len(history))
        else:
            st.metric("Session Selections", len(history))
    
    # Strategy Selection Timeline
    st.subheader("📊 Strategy Selection Timeline")
    
    timeline_data = []
    for h in history:
        timeline_data.append({
            "Date": h.get("date", "Unknown"),
            "Strategy Index": h.get("selected_strategy_index", 0),
            "Strategy Name": h.get("selected_strategy_name", "Unknown")
        })
    
    if timeline_data:
        timeline_df = pd.DataFrame(timeline_data)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timeline_df["Date"],
            y=timeline_df["Strategy Index"],
            mode='lines+markers',
            marker=dict(size=12, color=timeline_df["Strategy Index"], colorscale='Viridis'),
            text=timeline_df["Strategy Name"],
            hovertemplate='%{text}<br>Date: %{x}<br>Index: %{y}<extra></extra>',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title="User Strategy Selection Over Time",
            xaxis_title="Date",
            yaxis_title="Strategy Selected (0=Balanced, 1=Cost-Efficient, 2=Max Coverage)",
            height=400,
            yaxis=dict(tickmode='linear', tick0=0, dtick=1, range=[-0.5, 2.5])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Preference Weights Evolution
    st.subheader("⚖️ Preference Weights Evolution")
    
    # Extract weights over time
    weights_data = []
    for h in history:
        profile = h.get("preference_profile", {})
        weights = profile.get("weights", {}) if isinstance(profile, dict) else {}
        if weights:
            weights_data.append({
                "Date": h.get("date", "Unknown"),
                "Minimize Cost": weights.get("minimize_cost", 0.33),
                "Maximize Coverage": weights.get("maximize_coverage", 0.33),
                "Minimize Shortage": weights.get("minimize_shortage", 0.34)
            })
    
    if weights_data:
        weights_df = pd.DataFrame(weights_data)
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Minimize Cost', x=weights_df["Date"], y=weights_df["Minimize Cost"]))
        fig.add_trace(go.Bar(name='Maximize Coverage', x=weights_df["Date"], y=weights_df["Maximize Coverage"]))
        fig.add_trace(go.Bar(name='Minimize Shortage', x=weights_df["Date"], y=weights_df["Minimize Shortage"]))
        fig.update_layout(
            title="Preference Weight Evolution",
            xaxis_title="Date",
            yaxis_title="Weight",
            barmode='stack',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Preference weights will appear after strategy selections are made")
    
    # Learning Impact
    st.subheader("🎯 Learning Impact")
    
    if len(history) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**First Day Ranking:**")
            first_day = history[0]
            first_strategies = first_day.get("ranked_strategies", [])[:3]
            if first_strategies:
                first_df = pd.DataFrame([
                    {
                        "Strategy": s.get("strategy_name", "Unknown"),
                        "Preference Score": f"{s.get('preference_score', 0):.3f}"
                    }
                    for s in first_strategies
                ])
                st.dataframe(first_df, use_container_width=True, hide_index=True)
            else:
                st.info("No ranking data available")
        
        with col2:
            st.write("**Latest Day Ranking:**")
            last_day = history[-1]
            last_strategies = last_day.get("ranked_strategies", [])[:3]
            if last_strategies:
                last_df = pd.DataFrame([
                    {
                        "Strategy": s.get("strategy_name", "Unknown"),
                        "Preference Score": f"{s.get('preference_score', 0):.3f}"
                    }
                    for s in last_strategies
                ])
                st.dataframe(last_df, use_container_width=True, hide_index=True)
            else:
                st.info("No ranking data available")
        
        # Infer learning
        strategy_counts = {}
        for h in history:
            name = h.get("selected_strategy_name", "Unknown")
            strategy_counts[name] = strategy_counts.get(name, 0) + 1
        
        if strategy_counts:
            most_selected = max(strategy_counts, key=strategy_counts.get)
            st.success(f"✅ System learned: User prefers **{most_selected}** strategies")
    
    # Recommendation Quality Over Time
    st.subheader("📈 Recommendation Quality Over Time")
    
    quality_data = []
    for h in history:
        ranked = h.get("ranked_strategies", [])
        if ranked:
            top_score = ranked[0].get("preference_score", 0) if ranked else 0
            quality_data.append({
                "Date": h.get("date", "Unknown"),
                "Top Recommendation Score": top_score
            })
    
    if quality_data:
        quality_df = pd.DataFrame(quality_data)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=quality_df["Date"],
            y=quality_df["Top Recommendation Score"],
            mode='lines+markers',
            name='Top Recommendation Score',
            line=dict(color='#2ca02c', width=2),
            marker=dict(size=10)
        ))
        fig.update_layout(
            title="Preference Score of Top Recommendation Over Time",
            xaxis_title="Date",
            yaxis_title="Preference Score",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Higher scores indicate better alignment with user preferences")
    
    # Decision Patterns
    st.subheader("📊 Decision Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        strategy_dist = {}
        for h in history:
            name = h.get("selected_strategy_name", "Unknown")
            strategy_dist[name] = strategy_dist.get(name, 0) + 1
        
        if strategy_dist:
            fig = go.Figure(data=[go.Pie(
                labels=list(strategy_dist.keys()),
                values=list(strategy_dist.values()),
                hole=0.3
            )])
            fig.update_layout(title="Strategy Selection Distribution", height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Recent Selections:**")
        recent = history[-5:] if len(history) >= 5 else history
        if recent:
            recent_df = pd.DataFrame([
                {
                    "Date": h.get("date", "Unknown"),
                    "Strategy": h.get("selected_strategy_name", "Unknown"),
                    "Feedback": h.get("feedback_text", "")[:50] + "..." if h.get("feedback_text") else "None"
                }
                for h in recent
            ])
            st.dataframe(recent_df, use_container_width=True, hide_index=True)
        else:
            st.info("No recent selections")

# === SIDEBAR ===

def render_sidebar():
    """Render sidebar configuration"""
    with st.sidebar:
        st.title("MedFlow")
        st.markdown("**Video Demo Dashboard**")
        if CREWAI_AVAILABLE:
            st.markdown("🔄 Multi-Framework: LangGraph & CrewAI")
        else:
            st.markdown("Powered by LangGraph")
        
        st.divider()
        
        st.subheader("Configuration")
        
        # Outbreak selector
        outbreak_options = list(OUTBREAKS.keys())
        selected_outbreak = st.selectbox(
            "Outbreak Scenario",
            outbreak_options,
            index=0 if st.session_state.selected_outbreak in outbreak_options 
                  else outbreak_options.index(st.session_state.selected_outbreak) if outbreak_options else 0
        )
        st.session_state.selected_outbreak = selected_outbreak
        
        # Set simulation date from outbreak
        if selected_outbreak in OUTBREAKS:
            outbreak = OUTBREAKS[selected_outbreak]
            outbreak_start = datetime.strptime(outbreak["start_date"], "%Y-%m-%d").date()
            st.session_state.simulation_date = outbreak_start.strftime("%Y-%m-%d")
            # Update current_date if not set or if switching outbreaks
            if st.session_state.current_date is None:
                st.session_state.current_date = outbreak_start
        
        # Resource type
        resource_type = st.selectbox(
            "Resource Type",
            ["ventilators", "ppe", "o2_cylinders", "beds", "medications"],
            index=["ventilators", "ppe", "o2_cylinders", "beds", "medications"].index(
                st.session_state.resource_type
            ) if st.session_state.resource_type in ["ventilators", "ppe", "o2_cylinders", "beds", "medications"] else 0
        )
        st.session_state.resource_type = resource_type
        
        # User ID
        user_id = st.text_input(
            "User ID",
            value=st.session_state.user_id,
            help="User ID for preference learning"
        )
        st.session_state.user_id = user_id
        
        st.divider()
        
        # Framework Toggle - Prominent buttons
        st.subheader("🔄 AI Framework")
        
        if CREWAI_AVAILABLE:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(
                    "🔷 **LangGraph**",
                    type="primary" if st.session_state.framework == "langgraph" else "secondary",
                    use_container_width=True,
                    help="State-based workflow orchestration"
                ):
                    st.session_state.framework = "langgraph"
                    st.rerun()
            
            with col2:
                if st.button(
                    "🟣 **CrewAI**",
                    type="primary" if st.session_state.framework == "crewai" else "secondary",
                    use_container_width=True,
                    help="Agent-based collaboration"
                ):
                    st.session_state.framework = "crewai"
                    st.rerun()
            
            # Show current selection with icon
            framework_display = "LangGraph" if st.session_state.framework == "langgraph" else "CrewAI"
            icon = "🔷" if st.session_state.framework == "langgraph" else "🟣"
            st.success(f"{icon} **Active: {framework_display}**")
        else:
            st.session_state.framework = "langgraph"
            st.info("💡 CrewAI not available - using LangGraph")
        
        # Date setter - ensure current_date is initialized
        if "current_date" not in st.session_state or st.session_state.current_date is None:
            if st.session_state.simulation_date:
                try:
                    st.session_state.current_date = datetime.strptime(st.session_state.simulation_date, "%Y-%m-%d").date()
                except:
                    st.session_state.current_date = date(2024, 6, 15)
            else:
                st.session_state.current_date = date(2024, 6, 15)
        
        current_date = st.date_input(
            "Simulation Date",
            value=st.session_state.current_date,
            min_value=date(2024, 6, 15),
            max_value=date(2024, 7, 30),
            help="Select the date to simulate"
        )
        
        # Update session state only if current_date is not None
        if current_date is not None:
            st.session_state.current_date = current_date
            st.session_state.simulation_date = current_date.strftime("%Y-%m-%d")
        else:
            # Fallback to existing or default if somehow None
            if st.session_state.current_date is not None:
                st.session_state.simulation_date = st.session_state.current_date.strftime("%Y-%m-%d")
            else:
                st.session_state.current_date = date(2024, 6, 15)
                st.session_state.simulation_date = "2024-06-15"
        
        # Next Day button
        if st.button("➡️ Next Day", use_container_width=True):
            if current_date is not None:
                next_date = current_date + timedelta(days=1)
                max_date = date(2024, 7, 30)
                if next_date <= max_date:
                    st.session_state.current_date = next_date
                    st.session_state.simulation_date = next_date.strftime("%Y-%m-%d")
                    # Reset workflow state for new day
                    st.session_state.workflow_running = False
                    st.session_state.awaiting_review = False
                    st.session_state.current_workflow_state = None
                    st.rerun()
                else:
                    st.warning("Reached end of outbreak period")
            else:
                st.error("Please select a valid date")
        
        st.divider()
        
        # Simulation date display
        if st.session_state.simulation_date:
            st.info(f"📅 Simulation Date: {st.session_state.simulation_date}")
        
        st.divider()
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        
        # Footer
        st.markdown("**Tech Stack:**")
        st.markdown("FastAPI • LSTM • Supabase • Qdrant • LangGraph")
        st.markdown("---")
        st.markdown("[GitHub Repository](https://github.com/anshulLuhsna/MedFlow)")

# === MAIN APP ===

def main():
    """Main Streamlit app"""
    initialize_session_state()
    render_sidebar()
    
    # Header
    st.title("MedFlow")
    st.markdown("**Intelligent Resource Allocation System**")
    
    # Quick stats
    if st.session_state.simulation_date:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Simulation Date", st.session_state.simulation_date)
        with col2:
            st.metric("Active Outbreak", st.session_state.selected_outbreak)
    
    st.divider()
    
    # Tab navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏥 Capacity Dashboard",
        "🔮 Predictive Intelligence",
        "🤖 Agent Reasoning",
        "🧠 Adaptive Learning"
    ])
    
    with tab1:
        render_capacity_dashboard()
    
    with tab2:
        render_predictive_intelligence()
    
    with tab3:
        render_agent_reasoning()
    
    with tab4:
        render_adaptive_learning()

if __name__ == "__main__":
    main()
