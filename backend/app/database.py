"""
Database Client
Supabase client initialization and connection management
"""

from supabase import create_client, Client
from functools import lru_cache
from .config import get_settings


@lru_cache()
def get_supabase() -> Client:
    """
    Get Supabase client (singleton pattern)

    Returns:
        Supabase client instance

    Raises:
        ValueError: If Supabase credentials are not configured
    """
    settings = get_settings()

    if not settings.supabase_url or not settings.supabase_key:
        raise ValueError(
            "Supabase credentials not configured. "
            "Please set SUPABASE_URL and SUPABASE_KEY in .env file"
        )

    return create_client(settings.supabase_url, settings.supabase_key)


def get_ml_core():
    """
    Get ML Core instance with Supabase client

    Returns:
        MLCore instance
    """
    # Import here to avoid circular imports
    import sys
    from pathlib import Path

    # Add project root to path (where ml_core directory is)
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Now import ml_core
    from ml_core.core import MLCore

    supabase = get_supabase()
    return MLCore(supabase_client=supabase)
