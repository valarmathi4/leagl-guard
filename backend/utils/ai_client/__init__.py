"""
AI Client Package

A modular and clean implementation of AI clients for legal document analysis.
Provides contract analysis, metadata extraction, and compliance checking capabilities.
Supports both IBM WatsonX and Google Gemini AI services.
"""

from .config import ModelType, WatsonXConfig
from .client import WatsonXClient
from .gemini_config import GeminiConfig
from .gemini_client import GeminiClient
from .exceptions import WatsonXError, AuthenticationError, APIError, ConfigurationError, ResponseParsingError

__all__ = [
    'ModelType',
    'WatsonXConfig', 
    'WatsonXClient',
    'GeminiConfig',
    'GeminiClient',
    'WatsonXError',
    'AuthenticationError',
    'APIError',
    'ConfigurationError',
    'ResponseParsingError'
]

__version__ = "1.1.0"
