"""
MedFlow CrewAI Configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration for MedFlow CrewAI"""
    
    # LLM Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL = "groq/llama-3.3-70b-versatile"
    LLM_TEMPERATURE = 0.1  # Very low for deterministic behavior
    
    # API Configuration
    MEDFLOW_API_BASE = os.getenv("MEDFLOW_API_BASE", "http://localhost:8000")
    API_TIMEOUT = 120.0
    
    # CrewAI Settings
    VERBOSE = False  # Reduced verbosity for performance
    MEMORY = False  # Disable to prevent context confusion
    PLANNING = False  # Disable to prevent overthinking
    
    # Agent Limits (prevent hanging) - OPTIMIZED FOR SPEED
    MAX_ITER = 1  # Reduced from 3: Most tasks are simple tool calls
    MAX_EXECUTION_TIME = 60  # Reduced from 180: 1 minute max per task
    
    # File Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    AGENTS_CONFIG = PROJECT_ROOT / "config" / "agents.yaml"
    TASKS_CONFIG = PROJECT_ROOT / "config" / "tasks.yaml"
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in environment")
        if not cls.MEDFLOW_API_BASE:
            raise ValueError("MEDFLOW_API_BASE not set")
        return True

