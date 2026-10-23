"""Application configuration loaded from environment variables and .env."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from groq import Groq


class GroqConfigurationError(ValueError):
    """Raised when required Groq settings are missing or invalid."""


PROJECT_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge_base"

load_dotenv(PROJECT_ROOT / ".env")


def _project_path_from_env(name: str, default: str) -> Path:
    configured_path = Path(os.getenv(name, default)).expanduser()
    if not configured_path.is_absolute():
        configured_path = PROJECT_ROOT / configured_path
    return configured_path.resolve()


CHROMA_DB_PATH = _project_path_from_env("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL = (
    os.getenv("EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
)


def get_groq_model() -> str:
    """Return the explicitly configured chat model or explain how to set it."""
    model = os.getenv("GROQ_MODEL", "").strip()
    if not model:
        raise GroqConfigurationError(
            "GROQ_MODEL is not configured. Select a currently available chat "
            "model in your GroqCloud console and set its model ID in .env."
        )
    return model


def _configured_groq_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise GroqConfigurationError(
            "GROQ_API_KEY is not configured. Set it in the environment or .env "
            "before making a Groq API request."
        )
    return api_key


def create_groq_client() -> Groq:
    """Create a Groq client after validating the required settings."""
    get_groq_model()
    api_key = _configured_groq_api_key()
    from groq import Groq

    return Groq(api_key=api_key)
