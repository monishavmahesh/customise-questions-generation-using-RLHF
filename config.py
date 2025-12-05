"""
config.py — Central configuration for RLHF PoC (Python 3.11)

Uses python-dotenv to load Azure OpenAI credentials and app paths.
"""

from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Azure settings
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()

# Data directories
DATA_DIR: Path = BASE_DIR / "data"
PREFS_PATH: Path = DATA_DIR / "preferences.jsonl"
REWARD_DIR: Path = DATA_DIR / "reward_model"

DEFAULT_EMBEDDER: str = os.getenv(
    "EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Ensure folders exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
REWARD_DIR.mkdir(parents=True, exist_ok=True)

def ensure_azure_config() -> None:
    """Ensure all required Azure OpenAI credentials are available."""
    if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY and AZURE_OPENAI_DEPLOYMENT):
        raise EnvironmentError(
            "Azure OpenAI credentials missing. Set AZURE_OPENAI_ENDPOINT, "
            "AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT in .env"
        )
