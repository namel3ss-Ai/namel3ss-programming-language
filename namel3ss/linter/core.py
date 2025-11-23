"""Core semantic linter infrastructure."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union, Any
import logging

from namel3ss.ast import App
from namel3ss.parser import Parser
from namel3ss.effects.analyzer import EffectAnalyzer
from namel3ss.errors import N3Error
from .rules import LintRule


class LintSeverity(Enum):
    """Severity levels for lint findings."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


@dataclass
class LintFinding:
    """A single lint finding."""
    rule_id: str
    message: str
    severity: LintSeverity
    line: Optional[int] = None
    column: Optional[int] = None
    suggestion: Optional[str] = None
    code_context: Optional[str] = None


@dataclass
class LintResult:
    """Result of semantic linting."""
    findings: List[LintFinding]
    errors: List[str]
    warnings: List[str]
    
    def success(self) -> bool:
        """Check if linting completed without errors."""
        return len(self.errors) == 0
    
    def has_issues(self) -> bool:
        """Check if any issues were found."""
        return len(self.findings) > 0
    
    def error_count(self) -> int:
        """Count of error-level findings."""
        return sum(1 for f in self.findings if f.severity == LintSeverity.ERROR)
    
    def warning_count(self) -> int:
        """Count of warning-level findings."""
        return sum(1 for f in self.findings if f.severity == LintSeverity.WARNING)


class SemanticLinter:
    """
    Production-grade semantic linter for Namel3ss.
    
    This linter analyzes Namel3ss AST for semantic issues including:
    - Effect system violations
    - Type inconsistencies  
    - Unused definitions
    - Best practice violations
    - Performance concerns
    - Security issues
    """
    
    def __init__(self, rules: Optional[List[LintRule]] = None):
        self.rules = rules or []
        self.logger = logging.getLogger(__name__)
    
    def lint_document(self, source_text: str, file_path: str = "untitled.ai") -> LintResult:
        """
        Perform semantic analysis on a Namel3ss document.
        
        Args:
            source_text: Source code to analyze
            file_path: File path for context
            
        Returns:
            LintResult with findings and status
        """
        findings = []
        errors = []
        warnings = []
        
        try:
            # Parse the source into AST
            parser = Parser(source_text, path=file_path)
            ast = parser.parse()
            
            # Analyze effects
            effect_analyzer = EffectAnalyzer(ast)
            effect_analyzer.analyze()
            
            # Create analysis context for rules
            context = LintContext(
                source_text=source_text,
                file_path=file_path,
                ast=ast,
                effect_analyzer=effect_analyzer,
            )
            
            # Apply all lint rules
            for rule in self.rules:
                try:
                    rule_findings = rule.check(context)
                    findings.extend(rule_findings)
                except Exception as exc:
                    self.logger.warning(f"Rule {rule.rule_id} failed: {exc}")
                    warnings.append(f"Rule {rule.rule_id} encountered an error: {str(exc)}")
            
        except N3Error as e:
            # Syntax errors are handled by the parser, not the linter
            errors.append(f"Parse error: {e.message}")
        except Exception as e:
            errors.append(f"Linting error: {str(e)}")
        
        return LintResult(
            findings=findings,
            errors=errors,
            warnings=warnings,
        )


@dataclass
class LintContext:
    """Context provided to lint rules for analysis."""
    source_text: str
    file_path: str
    ast: App
    effect_analyzer: EffectAnalyzer
    
    def get_lines(self) -> List[str]:
        """Get source lines for line-based analysis."""
        return self.source_text.splitlines()
    
    def get_line(self, line_number: int) -> Optional[str]:
        """Get a specific source line (1-indexed)."""
        lines = self.get_lines()
        if 1 <= line_number <= len(lines):
            return lines[line_number - 1]
        return None