"""
Configuration for the Virtual Clinical Trial Agent.

Loads API keys from environment variables and defines model settings.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_openai_api_key() -> str:
    """Get OpenAI API key from environment."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return key


def get_gemini_api_key() -> str:
    """Get Gemini API key from environment."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    return key


# Model configurations
OPENAI_MODEL = "gpt-5.2"
GEMINI_MODEL = "gemini-3-pro-preview"

# Agent settings
MAX_VALIDATION_ITERATIONS = 3
