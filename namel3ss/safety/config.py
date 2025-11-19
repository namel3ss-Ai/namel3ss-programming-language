"""Configuration and factory functions for safety components.

This module handles initialization and selection of safety classifier
and PII redactor implementations based on environment configuration.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Any

from .runtime import SafetyClassifier, PIIRedactor


class SafetyConfig:
    """Global safety system configuration."""
    
    def __init__(self):
        self.classifier_type = os.getenv("NAMEL3SS_SAFETY_CLASSIFIER", "none")
        self.redactor_type = os.getenv("NAMEL3SS_PII_REDACTOR", "regex")
        self.fail_mode = os.getenv("NAMEL3SS_SAFETY_FAIL_MODE", "closed")  # closed or open
        self.enable_metrics = os.getenv("NAMEL3SS_SAFETY_METRICS", "true").lower() == "true"
        
    def should_fail_closed(self) -> bool:
        """Return True if system should fail closed (block on errors)."""
        return self.fail_mode == "closed"


# Global config instance
_config = SafetyConfig()


def get_safety_config() -> SafetyConfig:
    """Get the global safety configuration."""
    return _config


# -----------------------------------------------------------------------------
# Default Implementations
# -----------------------------------------------------------------------------


class NoOpSafetyClassifier(SafetyClassifier):
    """No-op classifier for development/testing that allows everything."""
    
    async def classify(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Always returns safe classification."""
        return {
            "categories": [],
            "flagged": False,
            "confidence": 1.0,
            "category_scores": {},
            "metadata": {"classifier": "noop"},
        }


class RegexPIIRedactor(PIIRedactor):
    """Simple regex-based PII redactor for common patterns.
    
    This is a basic implementation suitable for development. Production
    deployments should use more sophisticated NER-based or service-based
    redaction.
    """
    
    # Common PII patterns
    PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b',
        "ssn": r'\b(?!000|666|9\d{2})\d{3}-?(?!00)\d{2}-?(?!0{4})\d{4}\b',
        "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        "ip_address": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
    }
    
    def __init__(self, pii_types: Optional[List[str]] = None):
        """Initialize with optional list of PII types to detect.
        
        Args:
            pii_types: List of PII types to detect. If None, detects all types.
        """
        self.pii_types = pii_types or list(self.PATTERNS.keys())
        self.patterns = {
            name: re.compile(pattern)
            for name, pattern in self.PATTERNS.items()
            if name in self.pii_types
        }
    
    async def redact(self, text: str, pii_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Detect and redact PII using regex patterns."""
        redacted_text = text
        entities = []
        
        # Use provided pii_types or fall back to instance default
        types_to_check = pii_types or self.pii_types
        
        for pii_type in types_to_check:
            if pii_type not in self.patterns:
                continue
                
            pattern = self.patterns[pii_type]
            matches = list(pattern.finditer(redacted_text))
            
            for match in matches:
                entities.append({
                    "type": pii_type,
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                })
            
            # Redact matches (replace with placeholder)
            redacted_text = pattern.sub(f"[{pii_type.upper()}_REDACTED]", redacted_text)
        
        return {
            "text": redacted_text,
            "redacted": len(entities) > 0,
            "entities": entities,
            "metadata": {"redactor": "regex", "types_checked": types_to_check},
        }


class OpenAIModerationClassifier(SafetyClassifier):
    """Safety classifier using OpenAI's Moderation API.
    
    This provides production-grade content moderation for:
    - Hate speech
    - Self-harm
    - Sexual content
    - Violence
    
    Requires OPENAI_API_KEY environment variable.
    """
    
    # Map OpenAI moderation categories to generic category names
    CATEGORY_MAP = {
        "hate": "hate",
        "hate/threatening": "hate",
        "self-harm": "self-harm",
        "self-harm/intent": "self-harm",
        "self-harm/instructions": "self-harm",
        "sexual": "sexual",
        "sexual/minors": "sexual_minors",
        "violence": "violence",
        "violence/graphic": "violence",
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with OpenAI API key.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        """
        import openai
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required for moderation classifier")
        
        self.client = openai.OpenAI(api_key=self.api_key)
    
    async def classify(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Classify text using OpenAI Moderation API."""
        try:
            response = self.client.moderations.create(input=text)
            result = response.results[0]
            
            # Extract flagged categories
            categories = []
            category_scores = {}
            
            for category, flagged in result.categories.dict().items():
                if flagged:
                    generic_cat = self.CATEGORY_MAP.get(category, category)
                    if generic_cat not in categories:
                        categories.append(generic_cat)
                
                # Store scores for all categories
                generic_cat = self.CATEGORY_MAP.get(category, category)
                score = result.category_scores.dict().get(category, 0.0)
                category_scores[generic_cat] = max(
                    category_scores.get(generic_cat, 0.0),
                    score
                )
            
            return {
                "categories": categories,
                "flagged": result.flagged,
                "confidence": 1.0,  # OpenAI provides binary decisions
                "category_scores": category_scores,
                "metadata": {"classifier": "openai_moderation", "raw_result": result.dict()},
            }
            
        except Exception as e:
            # Return error result - enforcement layer will handle fail-closed logic
            raise RuntimeError(f"OpenAI Moderation API error: {e}") from e


# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------


def get_safety_classifier() -> SafetyClassifier:
    """Get the configured safety classifier implementation.
    
    Returns:
        SafetyClassifier instance based on NAMEL3SS_SAFETY_CLASSIFIER env var:
        - "none": NoOpSafetyClassifier (default for dev)
        - "openai": OpenAIModerationClassifier
        - Custom: Load from plugin/extension system (future)
    """
    classifier_type = _config.classifier_type.lower()
    
    if classifier_type == "none":
        return NoOpSafetyClassifier()
    elif classifier_type == "openai":
        return OpenAIModerationClassifier()
    else:
        # Future: Load custom classifier from registry
        raise ValueError(f"Unknown safety classifier type: {classifier_type}")


def get_pii_redactor() -> PIIRedactor:
    """Get the configured PII redactor implementation.
    
    Returns:
        PIIRedactor instance based on NAMEL3SS_PII_REDACTOR env var:
        - "none": No redaction
        - "regex": RegexPIIRedactor (default)
        - Custom: Load from plugin/extension system (future)
    """
    redactor_type = _config.redactor_type.lower()
    
    if redactor_type == "none":
        # Return a no-op redactor
        class NoOpRedactor(PIIRedactor):
            async def redact(self, text: str, pii_types: Optional[List[str]] = None) -> Dict[str, Any]:
                return {
                    "text": text,
                    "redacted": False,
                    "entities": [],
                    "metadata": {"redactor": "noop"},
                }
        return NoOpRedactor()
    elif redactor_type == "regex":
        return RegexPIIRedactor()
    else:
        # Future: Load custom redactor from registry
        raise ValueError(f"Unknown PII redactor type: {redactor_type}")
