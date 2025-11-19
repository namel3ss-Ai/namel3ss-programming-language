"""Core safety runtime abstractions for policy enforcement.

This module defines the interfaces and implementation for:
- Input/output safety classification
- PII detection and redaction
- Policy enforcement logic
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class SafetyAction(str, Enum):
    """Action to take based on safety classification."""
    ALLOW = "allow"
    BLOCK = "block"
    REDACT = "redact"
    ALERT = "alert"


@dataclass
class SafetyViolation:
    """Details of a safety policy violation."""
    categories: List[str]
    severity: str = "high"  # high, medium, low
    confidence: float = 1.0
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnforcedInput:
    """Result of input policy enforcement."""
    action: SafetyAction
    text: str  # Original or sanitized text
    violation: Optional[SafetyViolation] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_blocked(self) -> bool:
        return self.action == SafetyAction.BLOCK
    
    @property
    def is_allowed(self) -> bool:
        return self.action == SafetyAction.ALLOW


@dataclass
class EnforcedOutput:
    """Result of output policy enforcement."""
    action: SafetyAction
    text: str  # Redacted or original text
    original_text: Optional[str] = None  # Store original if redacted
    violation: Optional[SafetyViolation] = None
    pii_redacted: bool = False
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_blocked(self) -> bool:
        return self.action == SafetyAction.BLOCK
    
    @property
    def is_allowed(self) -> bool:
        return self.action == SafetyAction.ALLOW


@dataclass
class PolicyRuntime:
    """Runtime representation of a safety policy.
    
    This is the configuration that drives safety enforcement during
    chain/agent execution.
    """
    name: str
    block_categories: List[str] = field(default_factory=list)
    allow_categories: List[str] = field(default_factory=list)
    alert_only_categories: List[str] = field(default_factory=list)
    redact_pii: bool = False
    max_tokens: Optional[int] = None
    fallback_message: Optional[str] = None
    log_level: str = "full"  # full, minimal, none
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def should_block_category(self, category: str) -> bool:
        """Check if a category should be blocked."""
        # Explicit allow takes precedence
        if self.allow_categories and category in self.allow_categories:
            return False
        # Check block list
        return category in self.block_categories
    
    def should_alert_only(self, category: str) -> bool:
        """Check if a category should only trigger an alert, not a block."""
        return category in self.alert_only_categories
    
    def get_fallback_message(self) -> str:
        """Get the fallback message to return on violations."""
        return self.fallback_message or "I'm unable to help with that request."


class SafetyClassifier(ABC):
    """Abstract interface for safety classification services.
    
    Implementations can wrap:
    - Cloud-based safety APIs (OpenAI Moderation, Azure Content Safety, etc.)
    - Local safety models
    - Custom classification logic
    """
    
    @abstractmethod
    async def classify(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Classify text for safety violations.
        
        Args:
            text: The text to classify
            context: Optional context (e.g., conversation history, user metadata)
            
        Returns:
            Dictionary with:
                - categories: List[str] - Detected harmful categories
                - flagged: bool - Whether any violations were detected
                - confidence: float - Overall confidence (0.0-1.0)
                - category_scores: Dict[str, float] - Per-category scores
                - metadata: Dict[str, Any] - Additional classifier-specific data
        """
        pass


class PIIRedactor(ABC):
    """Abstract interface for PII detection and redaction.
    
    Implementations can use:
    - Regex-based pattern matching
    - NER models
    - Cloud-based PII detection services
    """
    
    @abstractmethod
    async def redact(self, text: str, pii_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Detect and redact PII from text.
        
        Args:
            text: The text to redact
            pii_types: Optional list of PII types to focus on (email, phone, ssn, etc.)
            
        Returns:
            Dictionary with:
                - text: str - Redacted text
                - redacted: bool - Whether any PII was found and redacted
                - entities: List[Dict] - List of detected PII entities with types and positions
                - metadata: Dict[str, Any] - Additional redactor-specific data
        """
        pass


async def enforce_input_policy(
    policy: PolicyRuntime,
    text: str,
    classifier: SafetyClassifier,
    context: Optional[Dict[str, Any]] = None,
) -> EnforcedInput:
    """Enforce a safety policy on input text.
    
    This is called before processing user input in a chain or agent.
    
    Args:
        policy: The policy to enforce
        text: User input text to check
        classifier: Safety classifier implementation
        context: Optional context for classification
        
    Returns:
        EnforcedInput with action and any violations
    """
    start_time = time.time()
    
    try:
        # Run safety classification
        result = await classifier.classify(text, context)
        
        categories = result.get("categories", [])
        flagged = result.get("flagged", False)
        confidence = result.get("confidence", 1.0)
        
        # Check if any detected categories should be blocked
        blocked_categories = [
            cat for cat in categories
            if policy.should_block_category(cat) and not policy.should_alert_only(cat)
        ]
        
        alert_categories = [
            cat for cat in categories
            if policy.should_alert_only(cat)
        ]
        
        latency_ms = (time.time() - start_time) * 1000
        
        if blocked_categories:
            # Block the input
            violation = SafetyViolation(
                categories=blocked_categories,
                confidence=confidence,
                reason="Input violates safety policy",
                metadata=result.get("category_scores", {}),
            )
            return EnforcedInput(
                action=SafetyAction.BLOCK,
                text=text,
                violation=violation,
                latency_ms=latency_ms,
                metadata={"classifier_result": result},
            )
        
        if alert_categories:
            # Allow but flag for logging
            violation = SafetyViolation(
                categories=alert_categories,
                severity="medium",
                confidence=confidence,
                reason="Input flagged for monitoring",
                metadata=result.get("category_scores", {}),
            )
            return EnforcedInput(
                action=SafetyAction.ALERT,
                text=text,
                violation=violation,
                latency_ms=latency_ms,
                metadata={"classifier_result": result},
            )
        
        # Allow the input
        return EnforcedInput(
            action=SafetyAction.ALLOW,
            text=text,
            latency_ms=latency_ms,
            metadata={"classifier_result": result},
        )
        
    except Exception as e:
        # Fail closed: block on classifier errors
        latency_ms = (time.time() - start_time) * 1000
        violation = SafetyViolation(
            categories=["classification_error"],
            reason=f"Safety classifier failed: {type(e).__name__}",
            metadata={"error": str(e)},
        )
        return EnforcedInput(
            action=SafetyAction.BLOCK,
            text=text,
            violation=violation,
            latency_ms=latency_ms,
            metadata={"error": str(e)},
        )


async def enforce_output_policy(
    policy: PolicyRuntime,
    text: str,
    classifier: SafetyClassifier,
    redactor: Optional[PIIRedactor] = None,
    context: Optional[Dict[str, Any]] = None,
) -> EnforcedOutput:
    """Enforce a safety policy on output text.
    
    This is called after generating output from an LLM or tool.
    
    Args:
        policy: The policy to enforce
        text: Generated output text to check
        classifier: Safety classifier implementation
        redactor: Optional PII redactor implementation
        context: Optional context for classification
        
    Returns:
        EnforcedOutput with action, redacted text, and any violations
    """
    start_time = time.time()
    original_text = text
    pii_redacted = False
    
    try:
        # First check for safety violations
        result = await classifier.classify(text, context)
        
        categories = result.get("categories", [])
        confidence = result.get("confidence", 1.0)
        
        # Check if any detected categories should be blocked
        blocked_categories = [
            cat for cat in categories
            if policy.should_block_category(cat) and not policy.should_alert_only(cat)
        ]
        
        alert_categories = [
            cat for cat in categories
            if policy.should_alert_only(cat)
        ]
        
        if blocked_categories:
            # Block the output
            latency_ms = (time.time() - start_time) * 1000
            violation = SafetyViolation(
                categories=blocked_categories,
                confidence=confidence,
                reason="Output violates safety policy",
                metadata=result.get("category_scores", {}),
            )
            return EnforcedOutput(
                action=SafetyAction.BLOCK,
                text=policy.get_fallback_message(),
                original_text=original_text,
                violation=violation,
                latency_ms=latency_ms,
                metadata={"classifier_result": result},
            )
        
        # If PII redaction is enabled, apply it
        if policy.redact_pii and redactor:
            redact_result = await redactor.redact(text)
            if redact_result.get("redacted", False):
                text = redact_result["text"]
                pii_redacted = True
        
        latency_ms = (time.time() - start_time) * 1000
        
        if alert_categories:
            # Allow but flag for logging
            violation = SafetyViolation(
                categories=alert_categories,
                severity="medium",
                confidence=confidence,
                reason="Output flagged for monitoring",
                metadata=result.get("category_scores", {}),
            )
            return EnforcedOutput(
                action=SafetyAction.ALERT,
                text=text,
                original_text=original_text if pii_redacted else None,
                violation=violation,
                pii_redacted=pii_redacted,
                latency_ms=latency_ms,
                metadata={"classifier_result": result},
            )
        
        # Allow the output
        return EnforcedOutput(
            action=SafetyAction.ALLOW,
            text=text,
            original_text=original_text if pii_redacted else None,
            pii_redacted=pii_redacted,
            latency_ms=latency_ms,
            metadata={"classifier_result": result},
        )
        
    except Exception as e:
        # Fail closed: block on errors
        latency_ms = (time.time() - start_time) * 1000
        violation = SafetyViolation(
            categories=["classification_error"],
            reason=f"Safety check failed: {type(e).__name__}",
            metadata={"error": str(e)},
        )
        return EnforcedOutput(
            action=SafetyAction.BLOCK,
            text=policy.get_fallback_message(),
            original_text=original_text,
            violation=violation,
            latency_ms=latency_ms,
            metadata={"error": str(e)},
        )
