"""Agent Configuration"""

import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    load_dotenv(override=True)

class AgentConfig:
    """Configuration for MedFlow agents"""
    
    # API Configuration
    MEDFLOW_API_BASE: str = os.getenv("MEDFLOW_API_BASE", "http://localhost:8000")
    MEDFLOW_API_KEY: str = os.getenv("MEDFLOW_API_KEY", "")
    
    # LLM Configuration (Groq/Llama 3.3 70B)
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "llama-3.3-70b-versatile")
    DEFAULT_LLM_TEMPERATURE: float = float(os.getenv("DEFAULT_LLM_TEMPERATURE", "0.3"))
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # API Client Configuration
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    TIMEOUT_SECONDS: float = float(os.getenv("TIMEOUT_SECONDS", "60"))
    
    # Workflow Configuration
    MAX_FORECAST_HOSPITALS: int = 5  # Legacy limit (kept for backward compatibility)
    DEFAULT_N_STRATEGIES: int = 3  # Generate 3 strategies by default
    # DEMO_HOSPITAL_LIMIT controls both forecasting and optimization for consistency
    # Forecasting node uses this limit to ensure same hospitals are processed as optimization
    DEMO_HOSPITAL_LIMIT: int = int(os.getenv("DEMO_HOSPITAL_LIMIT", "5"))  # Limit hospitals for demo (default: 5)
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        required = ["MEDFLOW_API_BASE"]
        for var in required:
            if not getattr(cls, var):
                raise ValueError(f"Missing required config: {var}")
        return True
