"""Tests for policy DSL parsing and validation."""

import pytest
from namel3ss.parser import Parser
from namel3ss.ast.policy import PolicyDefinition
from namel3ss.ast.ai import Chain


def _parse_source(source: str) -> any:
    """Helper to parse source code."""
    parser = Parser(source, path="test.n3")
    return parser.parse()


def test_parse_policy_basic():
    """Test parsing a basic policy definition."""
    source = '''
app "Test App".

policy safety {
    block_categories: ["violence", "hate", "self-harm"]
    redact_pii: true
    max_tokens: 512
    fallback_message: "I can't help with that."
}
'''
    
    module = _parse_source(source)
    assert module.body[0].policies
    
    policy = module.body[0].policies[0]
    assert isinstance(policy, PolicyDefinition)
    assert policy.name == "safety"
    assert "violence" in policy.block_categories
    assert "hate" in policy.block_categories
    assert "self-harm" in policy.block_categories
    assert policy.redact_pii is True
    assert policy.max_tokens == 512
    assert policy.fallback_message == "I can't help with that."


def test_parse_policy_all_fields():
    """Test parsing a policy with all optional fields."""
    source = '''
app "Test App".

policy comprehensive {
    block_categories: ["violence", "hate"]
    allow_categories: ["educational"]
    alert_only_categories: ["profanity"]
    redact_pii: true
    max_tokens: 1024
    fallback_message: "Safety violation detected."
    log_level: "full"
}
'''
    
    module = _parse_source(source)
    policy = module.body[0].policies[0]
    
    assert policy.name == "comprehensive"
    assert policy.block_categories == ["violence", "hate"]
    assert policy.allow_categories == ["educational"]
    assert policy.alert_only_categories == ["profanity"]
    assert policy.redact_pii is True
    assert policy.max_tokens == 1024
    assert policy.fallback_message == "Safety violation detected."
    assert policy.log_level == "full"


def test_parse_policy_minimal():
    """Test parsing a minimal policy with just block_categories."""
    source = '''
app "Test App".

policy minimal {
    block_categories: ["violence"]
}
'''
    
    module = _parse_source(source)
    policy = module.body[0].policies[0]
    
    assert policy.name == "minimal"
    assert policy.block_categories == ["violence"]
    assert policy.redact_pii is False
    assert policy.max_tokens is None
    assert policy.fallback_message is None


def test_chain_with_policy():
    """Test parsing a chain with policy attachment."""
    source = '''
app "Test App".

llm openai_model:
    provider: openai
    model: gpt-4

policy safety {
    block_categories: ["violence", "hate"]
    fallback_message: "Cannot process that request."
}

define chain "safe_qa":
    policy: safety
    input -> template.qa_prompt | llm.openai_model
'''
    
    module = _parse_source(source)
    app = module.body[0]
    
    # Verify policy exists
    assert len(app.policies) == 1
    policy = app.policies[0]
    assert policy.name == "safety"
    
    # Verify chain references policy
    assert len(app.chains) == 1
    chain = app.chains[0]
    assert isinstance(chain, Chain)
    assert chain.policy_name == "safety"


def test_multiple_policies():
    """Test parsing multiple policy definitions."""
    source = '''
app "Test App".

policy strict {
    block_categories: ["violence", "hate", "self-harm", "sexual"]
    redact_pii: true
}

policy lenient {
    alert_only_categories: ["profanity"]
    redact_pii: false
}
'''
    
    module = _parse_source(source)
    policies = module.body[0].policies
    
    assert len(policies) == 2
    assert policies[0].name == "strict"
    assert policies[1].name == "lenient"
    assert policies[0].redact_pii is True
    assert policies[1].redact_pii is False


def test_policy_validation_missing_reference():
    """Test that validator catches missing policy references."""
    from namel3ss.resolver import ModuleResolutionError, resolve_program
    from namel3ss.ast.program import Program
    
    source = '''
app "Test App".

llm openai_model:
    provider: openai
    model: gpt-4

define chain "safe_qa":
    policy: nonexistent_policy
    input -> template.qa_prompt | llm.openai_model
'''
    
    module = _parse_source(source)
    module.name = "test"  # Give module a name for resolution
    program = Program(modules=[module])
    
    with pytest.raises(ModuleResolutionError, match="unknown policy"):
        resolve_program(program)


def test_policy_with_special_characters_in_message():
    """Test policy with special characters in fallback message."""
    source = '''
app "Test App".

policy test {
    block_categories: ["violence"]
    fallback_message: "I can't help with that, but I can assist with other questions!"
}
'''
    
    module = _parse_source(source)
    policy = module.body[0].policies[0]
    
    assert policy.fallback_message == "I can't help with that, but I can assist with other questions!"


def test_policy_empty_categories():
    """Test policy with empty category lists."""
    source = '''
app "Test App".

policy monitor_only {
    block_categories: []
    alert_only_categories: ["profanity"]
    redact_pii: true
}
'''
    
    module = _parse_source(source)
    policy = module.body[0].policies[0]
    
    assert policy.block_categories == []
    assert policy.alert_only_categories == ["profanity"]
    assert policy.redact_pii is True
