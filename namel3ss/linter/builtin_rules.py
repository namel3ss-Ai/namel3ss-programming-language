"""Built-in semantic lint rules for Namel3ss."""

from __future__ import annotations

from typing import List, Set
import re

from namel3ss.ast import (
    App, Page, Chain, Prompt, Dataset, Model, Action, ShowForm, 
    Statement, ChainStep, Expression, NameRef
)
from .rules import LintRule
from .core import LintContext, LintFinding, LintSeverity


class UnusedDefinitionRule(LintRule):
    """Detect unused prompts, chains, datasets, and other definitions."""
    
    def __init__(self):
        super().__init__(
            rule_id="unused-definition",
            description="Detect unused prompts, chains, datasets, and other definitions"
        )
    
    def check(self, context: LintContext) -> List[LintFinding]:
        findings = []
        ast = context.ast
        
        # Track all defined names
        defined_prompts = {p.name for p in ast.prompts}
        defined_chains = {c.name for c in ast.chains}
        defined_datasets = {d.name for d in ast.datasets}
        defined_models = {m.name for m in ast.models}
        
        # Track all referenced names
        used_prompts = set()
        used_chains = set()
        used_datasets = set()
        used_models = set()
        
        # Scan chains for usage
        for chain in ast.chains:
            for step in chain.steps or []:
                if hasattr(step, 'target'):
                    if hasattr(step, 'kind') and step.kind == 'prompt':
                        used_prompts.add(step.target)
                    elif hasattr(step, 'kind') and step.kind == 'connector':
                        used_datasets.add(step.target)
        
        # Scan prompts for model usage
        for prompt in ast.prompts:
            if hasattr(prompt, 'model') and prompt.model:
                used_models.add(prompt.model)
        
        # Scan pages for usage
        for page in ast.pages:
            self._scan_statements_for_usage(page.statements, used_datasets, used_chains, used_prompts)
        
        # Report unused definitions
        for unused_prompt in defined_prompts - used_prompts:
            findings.append(LintFinding(
                rule_id=self.rule_id,
                message=f"Unused prompt: '{unused_prompt}'",
                severity=LintSeverity.WARNING,
                suggestion=f"Remove the unused prompt '{unused_prompt}' or add references to it"
            ))
        
        for unused_chain in defined_chains - used_chains:
            findings.append(LintFinding(
                rule_id=self.rule_id,
                message=f"Unused chain: '{unused_chain}'",
                severity=LintSeverity.WARNING,
                suggestion=f"Remove the unused chain '{unused_chain}' or add references to it"
            ))
        
        for unused_dataset in defined_datasets - used_datasets:
            findings.append(LintFinding(
                rule_id=self.rule_id,
                message=f"Unused dataset: '{unused_dataset}'",
                severity=LintSeverity.WARNING,
                suggestion=f"Remove the unused dataset '{unused_dataset}' or add references to it"
            ))
        
        for unused_model in defined_models - used_models:
            findings.append(LintFinding(
                rule_id=self.rule_id,
                message=f"Unused model: '{unused_model}'",
                severity=LintSeverity.WARNING,
                suggestion=f"Remove the unused model '{unused_model}' or add references to it"
            ))
        
        return findings
    
    def _scan_statements_for_usage(self, statements, used_datasets, used_chains, used_prompts):
        """Scan page statements for references."""
        for stmt in statements or []:
            if hasattr(stmt, 'dataset') and stmt.dataset:
                used_datasets.add(stmt.dataset)
            # Scan for references in action operations
            if isinstance(stmt, Action):
                for op in stmt.operations or []:
                    if hasattr(op, 'target'):
                        # Could be chain or prompt references
                        used_chains.add(op.target)
                        used_prompts.add(op.target)


class EffectViolationRule(LintRule):
    """Detect effect system violations using the existing EffectAnalyzer."""
    
    def __init__(self):
        super().__init__(
            rule_id="effect-violation",
            description="Detect effect system violations and inconsistencies"
        )
    
    def check(self, context: LintContext) -> List[LintFinding]:
        findings = []
        effect_analyzer = context.effect_analyzer
        ast = context.ast
        
        # Check for chains with effect violations
        for chain in ast.chains:
            if hasattr(chain, 'declared_effect') and hasattr(chain, 'effects'):
                declared = chain.declared_effect
                actual = chain.effects
                
                if declared and declared != "pure" and actual and "ai" not in actual:
                    findings.append(LintFinding(
                        rule_id=self.rule_id,
                        message=f"Chain '{chain.name}' declares effect '{declared}' but has no AI operations",
                        severity=LintSeverity.WARNING,
                        suggestion=f"Either add AI operations or declare the chain as 'pure'"
                    ))
                
                if declared == "pure" and actual and "ai" in actual:
                    findings.append(LintFinding(
                        rule_id=self.rule_id,
                        message=f"Chain '{chain.name}' declared as 'pure' but contains AI operations",
                        severity=LintSeverity.ERROR,
                        suggestion=f"Remove AI operations or change effect declaration to 'ai'"
                    ))
        
        return findings


class NamingConventionRule(LintRule):
    """Enforce naming conventions for better maintainability."""
    
    def __init__(self):
        super().__init__(
            rule_id="naming-convention",
            description="Enforce consistent naming conventions"
        )
    
    def check(self, context: LintContext) -> List[LintFinding]:
        findings = []
        ast = context.ast
        
        # Check snake_case for names
        snake_case_pattern = re.compile(r'^[a-z][a-z0-9_]*$')
        
        for prompt in ast.prompts:
            if not snake_case_pattern.match(prompt.name):
                findings.append(LintFinding(
                    rule_id=self.rule_id,
                    message=f"Prompt name '{prompt.name}' should use snake_case",
                    severity=LintSeverity.INFO,
                    suggestion=f"Consider renaming to use snake_case like '{self._to_snake_case(prompt.name)}'"
                ))
        
        for chain in ast.chains:
            if not snake_case_pattern.match(chain.name):
                findings.append(LintFinding(
                    rule_id=self.rule_id,
                    message=f"Chain name '{chain.name}' should use snake_case",
                    severity=LintSeverity.INFO,
                    suggestion=f"Consider renaming to use snake_case like '{self._to_snake_case(chain.name)}'"
                ))
        
        for dataset in ast.datasets:
            if not snake_case_pattern.match(dataset.name):
                findings.append(LintFinding(
                    rule_id=self.rule_id,
                    message=f"Dataset name '{dataset.name}' should use snake_case",
                    severity=LintSeverity.INFO,
                    suggestion=f"Consider renaming to use snake_case like '{self._to_snake_case(dataset.name)}'"
                ))
        
        return findings
    
    def _to_snake_case(self, name: str) -> str:
        """Convert a name to snake_case."""
        # Simple conversion - add underscores before capital letters and lowercase
        result = re.sub('([a-z0-9])([A-Z])', r'\\1_\\2', name)
        return result.lower()


class EmptyBlockRule(LintRule):
    """Detect empty blocks that might indicate incomplete implementation."""
    
    def __init__(self):
        super().__init__(
            rule_id="empty-block",
            description="Detect empty blocks that might be incomplete"
        )
    
    def check(self, context: LintContext) -> List[LintFinding]:
        findings = []
        ast = context.ast
        
        # Check for chains with no steps
        for chain in ast.chains:
            if not chain.steps or len(chain.steps) == 0:
                findings.append(LintFinding(
                    rule_id=self.rule_id,
                    message=f"Chain '{chain.name}' is empty",
                    severity=LintSeverity.WARNING,
                    suggestion="Add steps to the chain or remove it if not needed"
                ))
        
        # Check for pages with no statements
        for page in ast.pages:
            if not page.statements or len(page.statements) == 0:
                findings.append(LintFinding(
                    rule_id=self.rule_id,
                    message=f"Page '{page.route}' is empty",
                    severity=LintSeverity.WARNING,
                    suggestion="Add content to the page or remove it if not needed"
                ))
        
        return findings


class SecurityRule(LintRule):
    """Detect potential security issues."""
    
    def __init__(self):
        super().__init__(
            rule_id="security",
            description="Detect potential security vulnerabilities"
        )
    
    def check(self, context: LintContext) -> List[LintFinding]:
        findings = []
        source_lines = context.get_lines()
        
        # Check for hardcoded credentials or secrets
        sensitive_patterns = [
            (r'password\s*[:=]\s*["\'][^"\']+["\']', "Possible hardcoded password"),
            (r'api_?key\s*[:=]\s*["\'][^"\']+["\']', "Possible hardcoded API key"),
            (r'secret\s*[:=]\s*["\'][^"\']+["\']', "Possible hardcoded secret"),
            (r'token\s*[:=]\s*["\'][^"\']+["\']', "Possible hardcoded token"),
        ]
        
        for line_num, line in enumerate(source_lines, 1):
            for pattern, message in sensitive_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append(LintFinding(
                        rule_id=self.rule_id,
                        message=message,
                        severity=LintSeverity.WARNING,
                        line=line_num,
                        code_context=line.strip(),
                        suggestion="Use environment variables or configuration files for sensitive data"
                    ))
        
        return findings


class PerformanceRule(LintRule):
    """Detect potential performance issues."""
    
    def __init__(self):
        super().__init__(
            rule_id="performance",
            description="Detect potential performance problems"
        )
    
    def check(self, context: LintContext) -> List[LintFinding]:
        findings = []
        ast = context.ast
        
        # Check for very long chains that might be inefficient
        for chain in ast.chains:
            step_count = len(chain.steps or [])
            if step_count > 20:
                findings.append(LintFinding(
                    rule_id=self.rule_id,
                    message=f"Chain '{chain.name}' has {step_count} steps, consider breaking it down",
                    severity=LintSeverity.INFO,
                    suggestion="Long chains can be hard to debug and maintain. Consider splitting into smaller chains."
                ))
        
        # Check for potential inefficient dataset queries
        for dataset in ast.datasets:
            if hasattr(dataset, 'query') and dataset.query:
                query_lower = dataset.query.lower()
                if 'select *' in query_lower:
                    findings.append(LintFinding(
                        rule_id=self.rule_id,
                        message=f"Dataset '{dataset.name}' uses 'SELECT *', consider selecting specific columns",
                        severity=LintSeverity.INFO,
                        suggestion="Selecting specific columns improves performance and clarity"
                    ))
        
        return findings


def get_default_rules() -> List[LintRule]:
    """Get the default set of lint rules."""
    return [
        UnusedDefinitionRule(),
        EffectViolationRule(),
        NamingConventionRule(),
        EmptyBlockRule(),
        SecurityRule(),
        PerformanceRule(),
    ]