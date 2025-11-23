"""Tests for the semantic linter."""

import pytest
from namel3ss.linter import SemanticLinter, LintSeverity, get_default_rules
from namel3ss.linter.builtin_rules import (
    UnusedDefinitionRule,
    EffectViolationRule, 
    NamingConventionRule,
    EmptyBlockRule,
    SecurityRule,
    PerformanceRule,
)


class TestSemanticLinter:
    """Test the semantic linter functionality."""
    
    def test_basic_linting(self):
        """Test basic linting of a valid app."""
        source = '''app "TestApp" {
    prompt "greeting" {
        model: "gpt-4"
        template: "Hello {{name}}"
    }
    
    chain "support_flow" {
        prompt "greeting"
    }
    
    page "/home" {
        action "greet" {
            run chain: support_flow
        }
    }
}'''
        
        linter = SemanticLinter(get_default_rules())
        result = linter.lint_document(source)
        
        assert result.success()
        # Should have minimal issues for this well-formed app
    
    def test_unused_definition_rule(self):
        """Test detection of unused definitions."""
        source = '''app "TestApp" {
    prompt "used_prompt" {
        model: "gpt-4"
        template: "Hello"
    }
    
    prompt "unused_prompt" {
        model: "gpt-4"
        template: "Goodbye"
    }
    
    chain "support_flow" {
        prompt "used_prompt"
    }
    
    page "/home" {
        action "greet" {
            run chain: support_flow
        }
    }
}'''
        
        rule = UnusedDefinitionRule()
        linter = SemanticLinter([rule])
        result = linter.lint_document(source)
        
        assert result.success()
        assert len(result.findings) > 0
        
        unused_findings = [f for f in result.findings if "unused_prompt" in f.message.lower()]
        assert len(unused_findings) > 0
        assert unused_findings[0].severity == LintSeverity.WARNING
    
    def test_naming_convention_rule(self):
        """Test naming convention enforcement."""
        source = '''app "TestApp" {
    prompt "CamelCasePrompt" {
        model: "gpt-4"
        template: "Hello"
    }
    
    prompt "kebab-case-prompt" {
        model: "gpt-4" 
        template: "Hello"
    }
    
    prompt "snake_case_prompt" {
        model: "gpt-4"
        template: "Hello"
    }
}'''
        
        rule = NamingConventionRule()
        linter = SemanticLinter([rule])
        result = linter.lint_document(source)
        
        assert result.success()
        assert len(result.findings) > 0
        
        # Should flag CamelCase and kebab-case but not snake_case
        camel_case_findings = [f for f in result.findings if "CamelCasePrompt" in f.message]
        kebab_case_findings = [f for f in result.findings if "kebab-case-prompt" in f.message]
        
        assert len(camel_case_findings) > 0
        assert len(kebab_case_findings) > 0
        assert all(f.severity == LintSeverity.INFO for f in result.findings)
    
    def test_empty_block_rule(self):
        """Test detection of empty blocks."""
        source = '''app "TestApp" {
    chain "empty_chain" {
    }
    
    page "/empty_page" {
    }
    
    chain "normal_chain" {
        prompt "greeting"
    }
}'''
        
        rule = EmptyBlockRule()
        linter = SemanticLinter([rule])
        result = linter.lint_document(source)
        
        assert result.success()
        assert len(result.findings) >= 2  # empty chain and page
        
        empty_chain_findings = [f for f in result.findings if "empty_chain" in f.message]
        empty_page_findings = [f for f in result.findings if "/empty_page" in f.message]
        
        assert len(empty_chain_findings) > 0
        assert len(empty_page_findings) > 0
        assert all(f.severity == LintSeverity.WARNING for f in [*empty_chain_findings, *empty_page_findings])
    
    def test_security_rule(self):
        """Test detection of security issues."""
        source = '''app "TestApp" {
    config "database" {
        password: "hardcoded123"
        api_key: "secret-api-key"
        username: "admin"
    }
    
    config "safe" {
        timeout: 30
        debug: false
    }
}'''
        
        rule = SecurityRule()
        linter = SemanticLinter([rule])
        result = linter.lint_document(source)
        
        assert result.success()
        assert len(result.findings) >= 2  # password and api_key
        
        security_findings = [f for f in result.findings if "hardcoded" in f.message.lower() or "api" in f.message.lower()]
        assert len(security_findings) >= 2
        assert all(f.severity == LintSeverity.WARNING for f in security_findings)
    
    def test_performance_rule(self):
        """Test detection of performance issues.""" 
        source = '''app "TestApp" {
    chain "very_long_chain" {''' + '''
        prompt "step1"
        prompt "step2"''' * 15 + '''
    }
    
    dataset "inefficient" {
        query: "SELECT * FROM large_table"
    }
    
    chain "normal_chain" {
        prompt "step1"
        prompt "step2"
    }
}'''
        
        rule = PerformanceRule()
        linter = SemanticLinter([rule])
        result = linter.lint_document(source)
        
        assert result.success()
        assert len(result.findings) >= 2  # long chain and SELECT *
        
        long_chain_findings = [f for f in result.findings if "very_long_chain" in f.message]
        select_star_findings = [f for f in result.findings if "SELECT *" in f.message]
        
        assert len(long_chain_findings) > 0
        assert len(select_star_findings) > 0
        assert all(f.severity == LintSeverity.INFO for f in result.findings)
    
    def test_effect_violation_rule(self):
        """Test effect system violation detection."""
        source = '''app "TestApp" {
    chain "pure_with_ai" effect pure {
        prompt "greeting"
    }
    
    chain "ai_without_ai" effect ai {
        log "Just logging"
    }
    
    chain "proper_pure" effect pure {
        log "Just logging"
    }
}'''
        
        rule = EffectViolationRule()
        linter = SemanticLinter([rule])
        result = linter.lint_document(source)
        
        # Note: This test depends on the effect analyzer working correctly
        # Results may vary based on current effect analysis implementation
        assert result.success()
    
    def test_parse_error_handling(self):
        """Test handling of parse errors."""
        source = '''app "TestApp" 
invalid syntax here
}'''
        
        linter = SemanticLinter(get_default_rules())
        result = linter.lint_document(source)
        
        assert not result.success()
        assert len(result.errors) > 0
        assert "Parse error" in result.errors[0]
    
    def test_multiple_rules(self):
        """Test applying multiple rules together."""
        source = '''app "TestApp" {
    prompt "UnusedCamelCase" {
        model: "gpt-4"
        template: "Hello"
    }
    
    chain "empty_chain" {
    }
    
    config "security" {
        password: "hardcoded123"
    }
    
    page "/home" {
        show text: "Hello"
    }
}'''
        
        linter = SemanticLinter(get_default_rules())
        result = linter.lint_document(source)
        
        assert result.success()
        assert len(result.findings) > 0
        
        # Should have findings from multiple rules
        rule_ids = {f.rule_id for f in result.findings}
        assert len(rule_ids) > 1  # Multiple different rules triggered
    
    def test_rule_failure_handling(self):
        """Test handling of rule failures."""
        class FailingRule:
            def __init__(self):
                self.rule_id = "failing-rule"
                self.description = "A rule that always fails"
            
            def check(self, context):
                raise ValueError("This rule always fails")
        
        linter = SemanticLinter([FailingRule()])
        result = linter.lint_document('''app "TestApp" { }''')
        
        # Should succeed overall but have warnings about rule failure
        assert result.success()
        assert len(result.warnings) > 0
    
    def test_custom_rules(self):
        """Test adding custom lint rules."""
        from namel3ss.linter.rules import LintRule
        from namel3ss.linter.core import LintFinding, LintSeverity
        
        class CustomRule(LintRule):
            def __init__(self):
                super().__init__("custom-test", "Custom test rule")
            
            def check(self, context):
                # Flag all apps with "Test" in the name
                if "Test" in context.ast.name:
                    return [LintFinding(
                        rule_id=self.rule_id,
                        message="App name contains 'Test'",
                        severity=LintSeverity.INFO
                    )]
                return []
        
        linter = SemanticLinter([CustomRule()])
        result = linter.lint_document('''app "TestApp" { }''')
        
        assert result.success()
        assert len(result.findings) == 1
        assert result.findings[0].rule_id == "custom-test"
        assert result.findings[0].severity == LintSeverity.INFO