"""
Configuration classes for Gemini AI client components.
"""
import os
from dataclasses import dataclass
from typing import Optional
from .exceptions import ConfigurationError


@dataclass
class GeminiConfig:
    """Configuration for Gemini AI client."""
    api_key: str
    model_name: str = "gemini-pro"  # Default model, can be changed to gemini-1.5-pro for better performance
    temperature: float = 0.1
    max_tokens: int = 8192  # Maximum allowed by Gemini
    top_p: float = 0.95
    timeout: int = 120  # Increased from 60 to 120 seconds for longer documents

    @classmethod
    def from_environment(cls) -> 'GeminiConfig':
        """Create configuration from environment variables."""
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ConfigurationError("GEMINI_API_KEY must be set")
        
        # Optionally get model name from environment
        model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-pro")
        
        return cls(
            api_key=api_key,
            model_name=model_name
        )

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.api_key:
            raise ConfigurationError("API key is required")
        if not self.model_name:
            raise ConfigurationError("Model name is required")
        if self.temperature < 0 or self.temperature > 1:
            raise ConfigurationError("Temperature must be between 0 and 1")
        if self.max_tokens <= 0:
            raise ConfigurationError("Max tokens must be positive")
        if self.top_p <= 0 or self.top_p > 1:
            raise ConfigurationError("Top-p must be between 0 and 1")