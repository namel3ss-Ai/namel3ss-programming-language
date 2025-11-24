"""Tests for safety and policy system."""

import pytest
import asyncio
from namel3ss.safety import (
    PolicyRuntime,
    SafetyClassifier,
    PIIRedactor,
    SafetyAction,
    enforce_input_policy,
    enforce_output_policy,
)
from namel3ss.safety.config import NoOpSafetyClassifier, RegexPIIRedactor


class MockSafetyClassifier(SafetyClassifier):
    """Mock classifier for testing."""
    
    def __init__(self, flagged_categories=None):
        self.flagged_categories = flagged_categories or []
    
    async def classify(self, text, context=None):
        # Detect harmful content by keywords
        detected = []
        if "violence" in text.lower():
            detected.append("violence")
        if "hate" in text.lower():
            detected.append("hate")
        if "self-harm" in text.lower() or "suicide" in text.lower():
            detected.append("self-harm")
        
        return {
            "categories": detected,
            "flagged": len(detected) > 0,
            "confidence": 1.0,
            "category_scores": {cat: 0.9 for cat in detected},
            "metadata": {"classifier": "mock"},
        }


@pytest.mark.asyncio
async def test_policy_allows_safe_input():
    """Test that safe input passes through."""
    policy = PolicyRuntime(
        name="test_policy",
        block_categories=["violence", "hate", "self-harm"],
    )
    classifier = MockSafetyClassifier()
    
    result = await enforce_input_policy(policy, "Hello, how are you?", classifier)
    
    assert result.action == SafetyAction.ALLOW
    assert result.violation is None
    assert not result.is_blocked


@pytest.mark.asyncio
async def test_policy_blocks_unsafe_input():
    """Test that unsafe input is blocked."""
    policy = PolicyRuntime(
        name="test_policy",
        block_categories=["violence", "hate", "self-harm"],
        fallback_message="I can't help with that request.",
    )
    classifier = MockSafetyClassifier()
    
    result = await enforce_input_policy(policy, "How to commit violence", classifier)
    
    assert result.action == SafetyAction.BLOCK
    assert result.violation is not None
    assert "violence" in result.violation.categories
    assert result.is_blocked


@pytest.mark.asyncio
async def test_policy_alert_only_mode():
    """Test alert-only categories don't block."""
    policy = PolicyRuntime(
        name="test_policy",
        block_categories=[],
        alert_only_categories=["violence"],
    )
    classifier = MockSafetyClassifier()
    
    result = await enforce_input_policy(policy, "Minor violence reference", classifier)
    
    assert result.action == SafetyAction.ALERT
    assert result.violation is not None
    assert not result.is_blocked


@pytest.mark.asyncio
async def test_policy_allow_categories_override():
    """Test that allow_categories override block_categories."""
    policy = PolicyRuntime(
        name="test_policy",
        block_categories=["violence"],
        allow_categories=["violence"],  # Explicitly allow
    )
    classifier = MockSafetyClassifier()
    
    result = await enforce_input_policy(policy, "Educational violence content", classifier)
    
    assert result.action == SafetyAction.ALLOW
    assert not result.is_blocked


@pytest.mark.asyncio
async def test_output_pii_redaction():
    """Test PII redaction in outputs."""
    policy = PolicyRuntime(
        name="test_policy",
        block_categories=[],
        redact_pii=True,
    )
    classifier = NoOpSafetyClassifier()
    redactor = RegexPIIRedactor()
    
    text_with_pii = "Contact me at john@example.com or call 555-123-4567"
    result = await enforce_output_policy(policy, text_with_pii, classifier, redactor)
    
    assert result.action == SafetyAction.ALLOW
    assert result.pii_redacted
    assert "john@example.com" not in result.text
    assert "555-123-4567" not in result.text
    assert "[EMAIL_REDACTED]" in result.text
    assert "[PHONE_REDACTED]" in result.text


@pytest.mark.asyncio
async def test_output_blocking():
    """Test blocking unsafe outputs."""
    policy = PolicyRuntime(
        name="test_policy",
        block_categories=["hate"],
        fallback_message="I apologize, I cannot provide that information.",
    )
    classifier = MockSafetyClassifier()
    
    result = await enforce_output_policy(policy, "This is hate speech content", classifier)
    
    assert result.action == SafetyAction.BLOCK
    assert result.is_blocked
    assert result.text == "I apologize, I cannot provide that information."
    assert result.original_text == "This is hate speech content"


@pytest.mark.asyncio
async def test_classifier_error_fails_closed():
    """Test that classifier errors block content (fail closed)."""
    
    class ErrorClassifier(SafetyClassifier):
        async def classify(self, text, context=None):
            raise RuntimeError("Classifier service down")
    
    policy = PolicyRuntime(name="test_policy", block_categories=["violence"])
    classifier = ErrorClassifier()
    
    result = await enforce_input_policy(policy, "Any text", classifier)
    
    assert result.action == SafetyAction.BLOCK
    assert result.violation is not None
    assert "classification_error" in result.violation.categories


@pytest.mark.asyncio
async def test_regex_pii_redactor():
    """Test regex-based PII redaction."""
    redactor = RegexPIIRedactor()
    
    text = """
    Contact: john.doe@company.com
    Phone: 555-123-4567
    SSN: 123-45-6789
    Card: 4111-1111-1111-1111
    IP: 192.168.1.1
    """
    
    result = await redactor.redact(text)
    
    assert result["redacted"]
    assert len(result["entities"]) == 5
    assert "[EMAIL_REDACTED]" in result["text"]
    assert "[PHONE_REDACTED]" in result["text"]
    assert "[SSN_REDACTED]" in result["text"]
    assert "[CREDIT_CARD_REDACTED]" in result["text"]
    assert "[IP_ADDRESS_REDACTED]" in result["text"]


def test_policy_runtime_should_block():
    """Test PolicyRuntime category checking logic."""
    policy = PolicyRuntime(
        name="test",
        block_categories=["violence", "hate"],
        allow_categories=["educational_violence"],
        alert_only_categories=["profanity"],
    )
    
    # Should block
    assert policy.should_block_category("violence")
    assert policy.should_block_category("hate")
    
    # Should not block (allowed)
    assert not policy.should_block_category("educational_violence")
    
    # Should not block (alert only)
    assert not policy.should_block_category("profanity")
    assert policy.should_alert_only("profanity")
    
    # Should not block (not in any list)
    assert not policy.should_block_category("unknown_category")


def test_policy_fallback_message():
    """Test fallback message retrieval."""
    policy1 = PolicyRuntime(name="test1", fallback_message="Custom message")
    assert policy1.get_fallback_message() == "Custom message"
    
    policy2 = PolicyRuntime(name="test2")
    assert policy2.get_fallback_message() == "I'm unable to help with that request."
