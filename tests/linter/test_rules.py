"""Test individual lint rules."""

import pytest
from namel3ss.linter.builtin_rules import *
from namel3ss.linter.core import LintContext, LintSeverity
from namel3ss.parser import Parser
from namel3ss.effects.analyzer import EffectAnalyzer


def create_context(source: str) -> LintContext:
    """Helper to create lint context from source."""
    parser = Parser(source, path="test.ai")
    ast = parser.parse()
    effect_analyzer = EffectAnalyzer(ast)
    effect_analyzer.analyze()
    
    return LintContext(
        source_text=source,
        file_path="test.ai",
        ast=ast,
        effect_analyzer=effect_analyzer,
    )


class TestUnusedDefinitionRule:
    """Test the unused definition detection rule."""
    
    def test_unused_prompt(self):
        source = '''app "TestApp" {
    prompt "used" { model: "gpt-4", template: "Hello" }
    prompt "unused" { model: "gpt-4", template: "Bye" }
    
    chain "flow" {
        prompt "used"
    }
}'''
        
        rule = UnusedDefinitionRule()
        context = create_context(source)
        findings = rule.check(context)
        
        unused_findings = [f for f in findings if "unused" in f.message]
        assert len(unused_findings) == 1
        assert unused_findings[0].severity == LintSeverity.WARNING
    
    def test_all_used(self):
        source = '''app "TestApp" {
    prompt "greeting" { model: "gpt-4", template: "Hello" }
    
    chain "flow" {
        prompt "greeting"
    }
    
    page "/home" {
        action "greet" {
            run chain: flow
        }
    }
}'''
        
        rule = UnusedDefinitionRule()
        context = create_context(source)
        findings = rule.check(context)
        
        # Should have no unused findings
        unused_findings = [f for f in findings if "unused" in f.message.lower()]
        assert len(unused_findings) == 0


class TestNamingConventionRule:
    """Test the naming convention rule."""
    
    def test_snake_case_valid(self):
        source = '''app "TestApp" {
    prompt "valid_snake_case" { model: "gpt-4", template: "Hello" }
    chain "another_valid_name" { }
    dataset "user_data" { }
}'''
        
        rule = NamingConventionRule()
        context = create_context(source)
        findings = rule.check(context)
        
        # Should have no naming violations
        assert len(findings) == 0
    
    def test_camel_case_invalid(self):
        source = '''app "TestApp" {
    prompt "CamelCasePrompt" { model: "gpt-4", template: "Hello" }
    chain "AnotherCamelCase" { }
}'''
        
        rule = NamingConventionRule()
        context = create_context(source)
        findings = rule.check(context)
        
        assert len(findings) >= 2
        assert all(f.severity == LintSeverity.INFO for f in findings)
        assert any("CamelCasePrompt" in f.message for f in findings)
        assert any("AnotherCamelCase" in f.message for f in findings)
    
    def test_suggestions_provided(self):
        source = '''app "TestApp" {
    prompt "BadName123" { model: "gpt-4", template: "Hello" }
}'''
        
        rule = NamingConventionRule()
        context = create_context(source)
        findings = rule.check(context)
        
        assert len(findings) == 1
        assert findings[0].suggestion is not None
        assert "snake_case" in findings[0].suggestion


class TestEmptyBlockRule:
    """Test the empty block detection rule."""
    
    def test_empty_chain(self):
        source = '''app "TestApp" {
    chain "empty_chain" {
    }
}'''
        
        rule = EmptyBlockRule()
        context = create_context(source)
        findings = rule.check(context)
        
        assert len(findings) == 1
        assert "empty_chain" in findings[0].message
        assert findings[0].severity == LintSeverity.WARNING
    
    def test_empty_page(self):
        source = '''app "TestApp" {
    page "/empty" {
    }
}'''
        
        rule = EmptyBlockRule()
        context = create_context(source)
        findings = rule.check(context)
        
        assert len(findings) == 1
        assert "/empty" in findings[0].message
        assert findings[0].severity == LintSeverity.WARNING
    
    def test_non_empty_blocks(self):
        source = '''app "TestApp" {
    chain "good_chain" {
        prompt "greeting"
    }
    
    page "/good_page" {
        show text: "Hello"
    }
}'''
        
        rule = EmptyBlockRule()
        context = create_context(source)
        findings = rule.check(context)
        
        assert len(findings) == 0


class TestSecurityRule:
    """Test the security issue detection rule."""
    
    def test_hardcoded_password(self):
        source = '''app "TestApp" {
    config "db" {
        password: "secret123"
    }
}'''
        
        rule = SecurityRule()
        context = create_context(source)
        findings = rule.check(context)
        
        password_findings = [f for f in findings if "password" in f.message.lower()]
        assert len(password_findings) >= 1
        assert password_findings[0].severity == LintSeverity.WARNING
        assert password_findings[0].line is not None
    
    def test_hardcoded_api_key(self):
        source = '''app "TestApp" {
    config "api" {
        api_key: "sk-1234567890abcdef"
    }
}'''
        
        rule = SecurityRule()
        context = create_context(source)
        findings = rule.check(context)
        
        api_findings = [f for f in findings if "api" in f.message.lower()]
        assert len(api_findings) >= 1
        assert api_findings[0].severity == LintSeverity.WARNING
    
    def test_no_security_issues(self):
        source = '''app "TestApp" {
    config "safe" {
        timeout: 30
        debug: false
        name: "my_app"
    }
}'''
        
        rule = SecurityRule()
        context = create_context(source)
        findings = rule.check(context)
        
        # Should have no security findings
        assert len(findings) == 0


class TestPerformanceRule:
    """Test the performance issue detection rule."""
    
    def test_long_chain(self):
        # Create a chain with many steps
        steps = "\\n".join([f'        prompt "step{i}"' for i in range(25)])
        source = f'''app "TestApp" {{
    chain "very_long_chain" {{
{steps}
    }}
}}'''
        
        rule = PerformanceRule()
        context = create_context(source)
        findings = rule.check(context)
        
        long_chain_findings = [f for f in findings if "very_long_chain" in f.message]
        assert len(long_chain_findings) >= 1
        assert long_chain_findings[0].severity == LintSeverity.INFO
    
    def test_select_star_query(self):
        source = '''app "TestApp" {
    dataset "inefficient" {
        query: "SELECT * FROM users WHERE active = true"
    }
}'''
        
        rule = PerformanceRule()
        context = create_context(source)
        findings = rule.check(context)
        
        select_findings = [f for f in findings if "SELECT *" in f.message]
        assert len(select_findings) >= 1
        assert select_findings[0].severity == LintSeverity.INFO
    
    def test_normal_performance(self):
        source = '''app "TestApp" {
    chain "normal_chain" {
        prompt "step1"
        prompt "step2"
    }
    
    dataset "efficient" {
        query: "SELECT id, name FROM users WHERE active = true"
    }
}'''
        
        rule = PerformanceRule()
        context = create_context(source)
        findings = rule.check(context)
        
        # Should have no performance findings
        assert len(findings) == 0


class TestEffectViolationRule:
    """Test the effect system violation rule."""
    
    @pytest.mark.skip(reason="Depends on effect analysis implementation details")
    def test_pure_with_ai_operations(self):
        source = '''app "TestApp" {
    chain "pure_but_ai" effect pure {
        prompt "greeting"
    }
}'''
        
        rule = EffectViolationRule()
        context = create_context(source)
        findings = rule.check(context)
        
        # Should detect pure chain with AI operations
        pure_violations = [f for f in findings if "pure" in f.message]
        assert len(pure_violations) >= 1
        assert pure_violations[0].severity == LintSeverity.ERROR
    
    @pytest.mark.skip(reason="Depends on effect analysis implementation details") 
    def test_ai_without_ai_operations(self):
        source = '''app "TestApp" {
    chain "ai_but_pure" effect ai {
        log "Just logging"
    }
}'''
        
        rule = EffectViolationRule()
        context = create_context(source)
        findings = rule.check(context)
        
        # Should detect AI chain without AI operations
        ai_violations = [f for f in findings if "ai" in f.message.lower()]
        assert len(ai_violations) >= 1
        assert ai_violations[0].severity == LintSeverity.WARNING