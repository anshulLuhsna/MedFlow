"""
Configuration Management
Loads environment variables and app settings
"""

import os
from typing import List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings"""

    # API Configuration
    app_name: str = "MedFlow API"
    app_version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    environment: str = os.getenv("ENVIRONMENT", "development")

    # API Security
    api_keys: List[str] = os.getenv("API_KEYS", "dev-key-123").split(",")

    # Supabase Configuration
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_KEY", "")
    supabase_service_key: str = os.getenv("SUPABASE_SERVICE_KEY", "")

    # Groq Configuration (for ML Core)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")

    # CORS Configuration
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:8000",
    ]

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Singleton settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings
