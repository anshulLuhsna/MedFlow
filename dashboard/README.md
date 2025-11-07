# MedFlow Streamlit Dashboard

Web-based UI for running the MedFlow agentic workflow.

## Features

- **Interactive Workflow Execution**: Run the full 7-agent workflow through a web interface
- **Real-time Progress**: See workflow progress as it executes
- **Human-in-the-Loop**: Review and select strategies through the UI
- **Configuration**: Easy-to-use sidebar for configuring workflow parameters
- **Results Display**: Beautiful visualization of final recommendations

## Installation

1. Install dependencies:
```bash
pip install streamlit>=1.39.0
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Running the Dashboard

### Method 1: Using the run script (Recommended)

1. Make sure the backend is running:
```bash
cd backend
uvicorn app.main:app --reload
```

2. Run the Streamlit app from project root:
```bash
cd /home/anshul/Desktop/MedFlow
./dashboard/run.sh
```

Or make it executable first:
```bash
chmod +x dashboard/run.sh
./dashboard/run.sh
```

### Method 2: Run from project root

1. Make sure the backend is running
2. From the project root:
```bash
cd /home/anshul/Desktop/MedFlow
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export STREAMLIT_RUNNING=true
streamlit run dashboard/app.py --server.port 8501
```

### Method 3: Using Python directly

```bash
cd /home/anshul/Desktop/MedFlow
python3 -m streamlit run dashboard/app.py --server.port 8501
```

3. Open your browser to `http://localhost:8501`

## Usage

### Configuration

Use the sidebar to configure:
- **Resource Type**: Choose from ventilators, ppe, o2_cylinders, beds, medications
- **User ID**: Your user ID for preference learning (default: "default_user")
- **Hospital IDs**: Optional comma-separated list of hospital IDs to process
- **Outbreak ID**: Optional outbreak ID to simulate realistic scenarios

### Workflow Steps

1. Click **"Run Workflow"** to start the process
2. Wait for the workflow to complete (may take a few minutes)
3. When AI recommendations are ready, review the strategies
4. Select your preferred strategy
5. Optionally provide feedback
6. Click **"Submit Selection"** to complete the workflow

### Workflow Nodes

1. **Data Analyst** - Assesses current shortages and outbreaks
2. **Forecasting** - Predicts 14-day demand for at-risk hospitals
3. **Optimization** - Generates multiple allocation strategies
4. **Preference** - Ranks strategies by your preferences
5. **Reasoning** - Generates AI explanation
6. **Human Review** - You review and select a strategy
7. **Feedback** - System learns from your decision

## Troubleshooting

### Workflow Stuck on "Running..."

- Check that the backend is running on `http://localhost:8000`
- Check the Streamlit logs for errors
- Try reducing the number of hospitals using the `DEMO_HOSPITAL_LIMIT` environment variable

### Human Review Not Showing

- The workflow should automatically pause at the human review step
- If it doesn't, check the workflow state in the logs
- Try resetting and running again

### Errors

- Check that all environment variables are set correctly in `.env`
- Ensure the backend API is accessible
- Check that all required dependencies are installed

## Notes

- The workflow will pause at the human review step to allow you to select a strategy
- The human review node uses a different mechanism in Streamlit vs CLI to avoid blocking
- Session state is preserved across page refreshes within the same session

