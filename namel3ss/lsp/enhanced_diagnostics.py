"""
Enhanced diagnostics provider for better error reporting.

Provides improved error messages with:
- Context-aware suggestions
- Common mistake detection
- Legacy syntax migration hints
- Performance-optimized parsing with caching
"""

from typing import List, Optional, Dict, Any
import re

from lsprotocol.types import Diagnostic, DiagnosticSeverity, Position, Range

from namel3ss.errors import N3Error, N3SyntaxError
from namel3ss.parser import Parser


class EnhancedDiagnosticsProvider:
    """Provides enhanced error diagnostics with helpful suggestions."""
    
    # Common error patterns and their improved messages
    ERROR_IMPROVEMENTS = {
        "Expected: colon": {
            "pattern": r"Expected: colon.*Found: string",
            "message": "Missing colon in property declaration. Use: `property: value` instead of `property value`",
            "hint": "In modern N3 syntax, properties require a colon separator"
        },
        "Expected: app": {
            "pattern": r'Expected: app "Name"',
            "message": "Invalid app declaration syntax. Use: `app \"Name\" { ... }` with proper braces",
            "hint": "App declarations need curly braces and quoted names"
        },
        "Unexpected token": {
            "pattern": r"Unexpected token.*Expected: (\w+)",
            "message": lambda m: f"Syntax error: expected {m.group(1)} but found different token",
            "hint": "Check for missing punctuation or incorrect keyword spelling"
        },
        "Indentation": {
            "pattern": r"indentation|indent",
            "message": "Indentation error. N3 requires consistent spacing within blocks",
            "hint": "Use consistent spaces (recommended: 4 spaces) or tabs, but not mixed"
        }
    }
    
    # Legacy syntax patterns and modernization suggestions
    LEGACY_PATTERNS = {
        r'show\s+text\s+"([^"]*)"': {
            "message": "Legacy syntax detected. Use: `show text: \"{}\"` (with colon)",
            "suggestion": "Modern N3 uses colon syntax for properties"
        },
        r'show\s+form\s+"([^"]*)"': {
            "message": "Legacy form syntax. Use object syntax: `show form: {{ ... }}`",
            "suggestion": "Forms now use structured object definitions"
        },
        r'field\s+"([^"]*)"\s+type="([^"]*)"': {
            "message": "Legacy field syntax. Use: `field: {{ name: \"{}\", type: \"{}\" }}`",
            "suggestion": "Field definitions now use object syntax"
        }
    }
    
    def enhance_diagnostic(self, error: N3Error, document_text: str, line: int = None) -> Diagnostic:
        """Create enhanced diagnostic with better error messages and suggestions."""
        
        # Get basic diagnostic info
        error_line = max((error.line or 1) - 1, 0)
        error_column = max((error.column or 1) - 1, 0)
        
        # Extract context around error
        lines = document_text.splitlines()
        context_line = lines[error_line] if error_line < len(lines) else ""
        
        # Enhance the error message
        enhanced_message = self._enhance_error_message(error.message, context_line, lines, error_line)
        
        # Determine severity based on error type
        severity = DiagnosticSeverity.Error
        if "warning" in error.message.lower():
            severity = DiagnosticSeverity.Warning
        elif "hint" in error.message.lower() or "suggestion" in error.message.lower():
            severity = DiagnosticSeverity.Hint
        
        # Create range for the error
        start = Position(line=error_line, character=error_column)
        end = Position(line=error_line, character=error_column + 1)
        
        # Try to highlight the actual problematic token
        if context_line and error_column < len(context_line):
            # Find word boundaries around error position
            word_start = error_column
            word_end = error_column + 1
            
            # Expand to word boundaries
            while word_start > 0 and context_line[word_start - 1].isalnum():
                word_start -= 1
            while word_end < len(context_line) and context_line[word_end].isalnum():
                word_end += 1
            
            if word_end > word_start:
                end = Position(line=error_line, character=word_end)
                start = Position(line=error_line, character=word_start)
        
        return Diagnostic(
            range=Range(start=start, end=end),
            message=enhanced_message,
            severity=severity,
            source="namel3ss-enhanced",
            code=error.code or "syntax_error",
        )
    
    def _enhance_error_message(self, original_message: str, context_line: str, all_lines: List[str], line_number: int) -> str:
        """Enhance error message with context-aware suggestions."""
        
        # Check for known error patterns
        for pattern_name, pattern_info in self.ERROR_IMPROVEMENTS.items():
            if re.search(pattern_info["pattern"], original_message, re.IGNORECASE):
                message = pattern_info["message"]
                if callable(message):
                    match = re.search(pattern_info["pattern"], original_message, re.IGNORECASE)
                    message = message(match) if match else str(message)
                
                hint = pattern_info.get("hint", "")
                if hint:
                    return f"{message}. {hint}"
                return message
        
        # Check for legacy syntax patterns
        for pattern, pattern_info in self.LEGACY_PATTERNS.items():
            if re.search(pattern, context_line, re.IGNORECASE):
                match = re.search(pattern, context_line, re.IGNORECASE)
                if match:
                    message = pattern_info["message"]
                    try:
                        message = message.format(*match.groups())
                    except (IndexError, AttributeError):
                        pass
                    
                    suggestion = pattern_info.get("suggestion", "")
                    if suggestion:
                        return f"{message}. {suggestion}"
                    return message
        
        # Add context information to generic errors
        if "Unexpected token" in original_message:
            # Try to provide context about what was expected
            return f"{original_message}. Check the syntax around line {line_number + 1}: '{context_line.strip()}'"
        
        if "Expected:" in original_message and context_line.strip():
            return f"{original_message}. Found at: '{context_line.strip()}'"
        
        # Return enhanced original message with line context
        if context_line.strip():
            return f"{original_message} (line {line_number + 1}: '{context_line.strip()}')"
        
        return original_message
    
    def check_for_legacy_syntax_warnings(self, document_text: str) -> List[Diagnostic]:
        """Generate warnings for detected legacy syntax patterns."""
        diagnostics = []
        lines = document_text.splitlines()
        
        for line_idx, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith('#'):
                continue
            
            # Check for legacy patterns
            for pattern, info in self.LEGACY_PATTERNS.items():
                match = re.search(pattern, line)
                if match:
                    message = info["message"]
                    try:
                        message = message.format(*match.groups())
                    except (IndexError, AttributeError):
                        pass
                    
                    # Create warning diagnostic
                    start_col = match.start()
                    end_col = match.end()
                    
                    diagnostic = Diagnostic(
                        range=Range(
                            start=Position(line=line_idx, character=start_col),
                            end=Position(line=line_idx, character=end_col)
                        ),
                        message=f"Legacy syntax: {message}",
                        severity=DiagnosticSeverity.Warning,
                        source="namel3ss-legacy",
                        code="legacy_syntax"
                    )
                    diagnostics.append(diagnostic)
        
        return diagnostics


def enhance_document_diagnostics(document_state) -> None:
    """Enhance document state with improved diagnostics."""
    
    # Store original diagnostic creation method
    original_diagnostic_from_error = document_state._diagnostic_from_error
    enhanced_provider = EnhancedDiagnosticsProvider()
    
    def enhanced_diagnostic_from_error(error: N3Error) -> Diagnostic:
        """Create enhanced diagnostic with better error messages."""
        return enhanced_provider.enhance_diagnostic(error, document_state.text)
    
    # Replace method
    document_state._diagnostic_from_error = enhanced_diagnostic_from_error
    
    # Store original rebuild method
    original_rebuild = document_state.rebuild
    
    def enhanced_rebuild():
        """Enhanced rebuild that includes legacy syntax warnings and semantic analysis."""
        # Run original rebuild
        original_rebuild()
        
        # Add legacy syntax warnings
        legacy_warnings = enhanced_provider.check_for_legacy_syntax_warnings(document_state.text)
        document_state.diagnostics.extend(legacy_warnings)
        
        # Add semantic lint findings
        try:
            from namel3ss.linter import SemanticLinter, get_default_rules
            from lsprotocol.types import DiagnosticSeverity as LspSeverity
            
            linter = SemanticLinter(get_default_rules())
            result = linter.lint_document(document_state.text, document_state.uri)
            
            if result.success():  # Only add findings if linting succeeded
                for finding in result.findings:
                    # Map linter severity to LSP severity
                    lsp_severity = LspSeverity.Error
                    if finding.severity.value == "warning":
                        lsp_severity = LspSeverity.Warning
                    elif finding.severity.value == "info":
                        lsp_severity = LspSeverity.Information
                    elif finding.severity.value == "hint":
                        lsp_severity = LspSeverity.Hint
                    
                    # Create diagnostic from finding
                    line = max((finding.line or 1) - 1, 0)
                    start_char = finding.column or 0
                    end_char = start_char + 1  # Default range
                    
                    # Try to get better range from context
                    lines = document_state.text.splitlines()
                    if line < len(lines) and finding.code_context:
                        line_text = lines[line]
                        context_pos = line_text.find(finding.code_context.strip())
                        if context_pos >= 0:
                            start_char = context_pos
                            end_char = context_pos + len(finding.code_context.strip())
                    
                    diagnostic = Diagnostic(
                        range=Range(
                            start=Position(line=line, character=start_char),
                            end=Position(line=line, character=end_char)
                        ),
                        message=finding.message,
                        severity=lsp_severity,
                        source="namel3ss-semantic",
                        code=finding.rule_id
                    )
                    
                    document_state.diagnostics.append(diagnostic)
        
        except ImportError:
            # Semantic linter not available, skip
            pass
        except Exception as exc:
            # Log error but don't fail diagnostics
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Semantic linting failed: {exc}")
    
    # Replace rebuild method  
    document_state.rebuild = enhanced_rebuild