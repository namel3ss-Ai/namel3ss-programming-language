"""
Editor/IDE Integration API for Namel3ss

This module provides LSP-ready functions for building editor integrations:
- Syntax parsing and validation
- Symbol resolution and lookup
- Find references and definitions
- Structured diagnostics
- Hover information
- Code completion context

This API is designed to be consumed by LSP servers or IDE extensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import ast as python_ast

from namel3ss.ast import Module, App, Expression
from namel3ss.ast.expressions import VarExpr, CallExpr, LambdaExpr
from namel3ss.lang.parser import parse_module, N3SyntaxError
from namel3ss.lang.parser.errors import N3Error, N3SemanticError
from namel3ss.types import check_module_static, N3TypeError
from namel3ss.modules.system import ModuleResolver, load_multi_module_project


# ==================== Position and Location ====================


@dataclass
class Position:
    """A position in a source file (0-indexed)."""
    line: int
    character: int
    
    def __lt__(self, other: Position) -> bool:
        if self.line != other.line:
            return self.line < other.line
        return self.character < other.character
    
    def __le__(self, other: Position) -> bool:
        return self == other or self < other


@dataclass
class Range:
    """A range in a source file."""
    start: Position
    end: Position
    
    def contains(self, pos: Position) -> bool:
        """Check if this range contains a position."""
        return self.start <= pos <= self.end


@dataclass
class Location:
    """A location in a source file."""
    uri: str  # File path or URI
    range: Range


# ==================== Symbols ====================


@dataclass
class Symbol:
    """
    A symbol in the source code.
    
    Represents any named entity: variables, functions, datasets, prompts, etc.
    """
    name: str
    kind: str  # "variable", "function", "dataset", "prompt", "chain", "page", "model", "parameter"
    location: Location
    type_info: Optional[str] = None  # Human-readable type string
    container: Optional[str] = None  # Containing scope (e.g. "app.main")
    documentation: Optional[str] = None


@dataclass
class SymbolReference:
    """A reference to a symbol."""
    symbol: Symbol
    location: Location
    is_definition: bool = False
    is_write: bool = False  # True if this reference writes to the symbol


# ==================== Diagnostics ====================


@dataclass
class Diagnostic:
    """
    A diagnostic message (error, warning, or info).
    
    Compatible with LSP diagnostic format.
    """
    range: Range
    message: str
    severity: str  # "error", "warning", "info", "hint"
    code: Optional[str] = None
    source: str = "namel3ss"
    related_information: List[DiagnosticRelatedInformation] = field(default_factory=list)


@dataclass
class DiagnosticRelatedInformation:
    """Related information for a diagnostic."""
    location: Location
    message: str


# ==================== Analysis Results ====================


@dataclass
class AnalysisResult:
    """
    Complete analysis result for a module.
    """
    uri: str
    diagnostics: List[Diagnostic]
    symbols: List[Symbol]
    module_ast: Optional[Module] = None
    parse_success: bool = True


# ==================== Editor API ====================


class EditorAPI:
    """
    Main API for editor integrations.
    
    Provides:
    - Parsing and validation
    - Symbol resolution
    - Find references/definitions
    - Hover information
    - Code completion context
    """
    
    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize the editor API.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.module_resolver = ModuleResolver(project_root=project_root) if project_root else None
        self.module_cache: Dict[str, AnalysisResult] = {}
    
    def parse_source(self, source: str, uri: str = "untitled") -> AnalysisResult:
        """
        Parse source code and return analysis result.
        
        Args:
            source: Source code to parse
            uri: File URI or path
        
        Returns:
            AnalysisResult with diagnostics and AST
        """
        diagnostics = []
        module_ast = None
        parse_success = False
        
        try:
            module_ast = parse_module(source, path=uri)
            parse_success = True
        except N3SyntaxError as e:
            diagnostics.append(self._syntax_error_to_diagnostic(e))
        except Exception as e:
            diagnostics.append(Diagnostic(
                range=Range(Position(0, 0), Position(0, 0)),
                message=f"Parse error: {str(e)}",
                severity="error",
                code="PARSE_ERROR"
            ))
        
        # Extract symbols if parsing succeeded
        symbols = []
        if parse_success and module_ast:
            symbols = self._extract_symbols(module_ast, uri)
        
        return AnalysisResult(
            uri=uri,
            diagnostics=diagnostics,
            symbols=symbols,
            module_ast=module_ast,
            parse_success=parse_success
        )
    
    def analyze_module(self, source: str, uri: str = "untitled", run_type_check: bool = True) -> AnalysisResult:
        """
        Full analysis of a module including type checking.
        
        Args:
            source: Source code to analyze
            uri: File URI or path
            run_type_check: Whether to run static type checking
        
        Returns:
            AnalysisResult with all diagnostics, symbols, and AST
        """
        # First parse the module
        result = self.parse_source(source, uri)
        
        # If parsing succeeded, run type checking
        if run_type_check and result.parse_success and result.module_ast:
            try:
                type_errors = check_module_static(result.module_ast, uri)
                for type_error in type_errors:
                    result.diagnostics.append(self._type_error_to_diagnostic(type_error))
            except Exception as e:
                result.diagnostics.append(Diagnostic(
                    range=Range(Position(0, 0), Position(0, 0)),
                    message=f"Type checking error: {str(e)}",
                    severity="error",
                    code="TYPE_CHECK_ERROR"
                ))
        
        # Cache result
        self.module_cache[uri] = result
        
        return result
    
    def get_symbol_at_position(self, uri: str, position: Position) -> Optional[Symbol]:
        """
        Get the symbol at a specific position.
        
        Args:
            uri: File URI
            position: Position in file
        
        Returns:
            Symbol at position, or None
        """
        result = self.module_cache.get(uri)
        if not result:
            return None
        
        # Find symbol whose location contains this position
        for symbol in result.symbols:
            if symbol.location.range.contains(position):
                return symbol
        
        return None
    
    def find_references(self, uri: str, position: Position, include_declaration: bool = True) -> List[Location]:
        """
        Find all references to the symbol at a position.
        
        Args:
            uri: File URI
            position: Position of symbol
            include_declaration: Whether to include the declaration
        
        Returns:
            List of locations where symbol is referenced
        """
        symbol = self.get_symbol_at_position(uri, position)
        if not symbol:
            return []
        
        references = []
        
        # Search in current module
        result = self.module_cache.get(uri)
        if result and result.module_ast:
            refs = self._find_symbol_references(result.module_ast, symbol.name, uri)
            references.extend(refs)
        
        # TODO: Search in other modules that import this one
        
        if not include_declaration:
            # Filter out the definition
            references = [loc for loc in references if loc != symbol.location]
        
        return references
    
    def find_definition(self, uri: str, position: Position) -> Optional[Location]:
        """
        Find the definition of the symbol at a position.
        
        Args:
            uri: File URI
            position: Position of symbol usage
        
        Returns:
            Location of symbol definition, or None
        """
        symbol = self.get_symbol_at_position(uri, position)
        if symbol:
            return symbol.location
        
        return None
    
    def get_hover_information(self, uri: str, position: Position) -> Optional[str]:
        """
        Get hover information for a symbol at a position.
        
        Args:
            uri: File URI
            position: Position to hover over
        
        Returns:
            Markdown-formatted hover text, or None
        """
        symbol = self.get_symbol_at_position(uri, position)
        if not symbol:
            return None
        
        parts = []
        
        # Symbol signature
        if symbol.type_info:
            parts.append(f"```namel3ss\n{symbol.name}: {symbol.type_info}\n```")
        else:
            parts.append(f"```namel3ss\n{symbol.name}\n```")
        
        # Kind
        parts.append(f"*{symbol.kind}*")
        
        # Documentation
        if symbol.documentation:
            parts.append("---")
            parts.append(symbol.documentation)
        
        # Container
        if symbol.container:
            parts.append(f"\nDefined in: `{symbol.container}`")
        
        return "\n\n".join(parts)
    
    def get_completion_context(self, uri: str, position: Position) -> Dict[str, Any]:
        """
        Get context for code completion at a position.
        
        Args:
            uri: File URI
            position: Position for completion
        
        Returns:
            Dictionary with completion context:
            - visible_symbols: List of symbols in scope
            - context_type: Type of context ("expression", "statement", "declaration", etc.)
            - expected_type: Expected type at this position (if known)
        """
        result = self.module_cache.get(uri)
        if not result or not result.module_ast:
            return {"visible_symbols": [], "context_type": "unknown"}
        
        # Get all symbols visible at this position
        visible_symbols = [s for s in result.symbols if s.location.range.start <= position]
        
        # TODO: Determine context type by analyzing AST structure
        context_type = "expression"  # Default
        
        return {
            "visible_symbols": visible_symbols,
            "context_type": context_type,
            "expected_type": None,
        }
    
    # ==================== Helper Methods ====================
    
    def _syntax_error_to_diagnostic(self, error: N3SyntaxError) -> Diagnostic:
        """Convert N3SyntaxError to Diagnostic."""
        line = error.line - 1 if error.line else 0
        col = error.column - 1 if error.column else 0
        
        return Diagnostic(
            range=Range(
                Position(line, col),
                Position(line, col + 1)
            ),
            message=error.message,
            severity="error",
            code=error.code or "SYNTAX_ERROR"
        )
    
    def _type_error_to_diagnostic(self, error: N3TypeError) -> Diagnostic:
        """Convert N3TypeError to Diagnostic."""
        line = error.line - 1 if error.line else 0
        col = error.column - 1 if error.column else 0
        
        return Diagnostic(
            range=Range(
                Position(line, col),
                Position(line, col + 1)
            ),
            message=error.message,
            severity="error",
            code=error.code or "TYPE_ERROR"
        )
    
    def _extract_symbols(self, module: Module, uri: str) -> List[Symbol]:
        """
        Extract all symbols from a module.
        
        Args:
            module: Parsed module AST
            uri: File URI
        
        Returns:
            List of symbols found in the module
        """
        symbols = []
        
        # Extract from app declarations
        for decl in module.body:
            if isinstance(decl, App):
                # App itself
                if hasattr(decl, 'name'):
                    symbols.append(Symbol(
                        name=decl.name,
                        kind="app",
                        location=Location(uri, Range(Position(0, 0), Position(0, len(decl.name)))),
                        documentation="Main application"
                    ))
                
                # Datasets
                for dataset in decl.datasets:
                    symbols.append(Symbol(
                        name=dataset.name,
                        kind="dataset",
                        location=self._get_location(dataset, uri),
                        type_info=f"dataset<{getattr(dataset, 'type', 'unknown')}>"
                    ))
                
                # Prompts
                for prompt in decl.prompts:
                    symbols.append(Symbol(
                        name=prompt.name,
                        kind="prompt",
                        location=self._get_location(prompt, uri),
                        type_info="prompt"
                    ))
                
                # Chains
                for chain in decl.chains:
                    symbols.append(Symbol(
                        name=chain.name,
                        kind="chain",
                        location=self._get_location(chain, uri),
                        type_info="chain"
                    ))
                
                # Pages
                for page in decl.pages:
                    symbols.append(Symbol(
                        name=page.name,
                        kind="page",
                        location=self._get_location(page, uri),
                        type_info="page"
                    ))
                
                # AI models
                for model in decl.ai_models:
                    symbols.append(Symbol(
                        name=model.name,
                        kind="model",
                        location=self._get_location(model, uri),
                        type_info="ai_model"
                    ))
        
        return symbols
    
    def _get_location(self, node: Any, uri: str) -> Location:
        """Get location for an AST node."""
        # Default location if no position information
        line = getattr(node, 'line', 0) or 0
        col = getattr(node, 'column', 0) or 0
        
        return Location(
            uri=uri,
            range=Range(
                Position(line - 1 if line > 0 else 0, col - 1 if col > 0 else 0),
                Position(line - 1 if line > 0 else 0, col)
            )
        )
    
    def _find_symbol_references(self, module: Module, symbol_name: str, uri: str) -> List[Location]:
        """
        Find all references to a symbol in a module.
        
        Args:
            module: Module to search
            symbol_name: Name of symbol to find
            uri: File URI
        
        Returns:
            List of locations where symbol is referenced
        """
        references = []
        
        # Walk the AST looking for VarExpr nodes with matching name
        def visit_expr(expr: Expression) -> None:
            if isinstance(expr, VarExpr) and expr.name == symbol_name:
                references.append(self._get_location(expr, uri))
            
            # Recursively visit child expressions
            if isinstance(expr, CallExpr):
                visit_expr(expr.func)
                for arg in expr.args:
                    visit_expr(arg)
            elif isinstance(expr, LambdaExpr):
                visit_expr(expr.body)
            # TODO: Add other expression types
        
        # Visit all expressions in the module
        for decl in module.body:
            if isinstance(decl, App):
                # Visit expressions in prompts
                for prompt in decl.prompts:
                    if hasattr(prompt, 'template'):
                        # TODO: Extract and visit expressions from template
                        pass
                
                # Visit expressions in chains
                for chain in decl.chains:
                    if hasattr(chain, 'steps'):
                        # TODO: Visit expressions in chain steps
                        pass
        
        return references


# ==================== Public API Functions ====================


def parse_source(source: str, uri: str = "untitled") -> AnalysisResult:
    """
    Parse Namel3ss source code.
    
    Args:
        source: Source code to parse
        uri: File URI or path
    
    Returns:
        AnalysisResult with diagnostics and AST
    
    Example:
        ```python
        result = parse_source(source_code, uri="file:///path/to/app.ai")
        
        if result.parse_success:
            print(f"Parsed successfully! Found {len(result.symbols)} symbols")
        else:
            for diag in result.diagnostics:
                print(f"{diag.severity}: {diag.message}")
        ```
    """
    api = EditorAPI()
    return api.parse_source(source, uri)


def analyze_module(source: str, uri: str = "untitled", run_type_check: bool = True) -> AnalysisResult:
    """
    Fully analyze a Namel3ss module including type checking.
    
    Args:
        source: Source code to analyze
        uri: File URI or path
        run_type_check: Whether to run static type checking
    
    Returns:
        AnalysisResult with all diagnostics, symbols, and AST
    
    Example:
        ```python
        result = analyze_module(source_code, uri="file:///path/to/app.ai")
        
        # Show all diagnostics
        for diag in result.diagnostics:
            print(f"[{diag.severity}] Line {diag.range.start.line}: {diag.message}")
        
        # List all symbols
        for symbol in result.symbols:
            print(f"{symbol.kind} {symbol.name}: {symbol.type_info}")
        ```
    """
    api = EditorAPI()
    return api.analyze_module(source, uri, run_type_check)


def find_symbol_at_position(source: str, line: int, character: int, uri: str = "untitled") -> Optional[Symbol]:
    """
    Find the symbol at a specific position in source code.
    
    Args:
        source: Source code
        line: Line number (0-indexed)
        character: Character offset (0-indexed)
        uri: File URI
    
    Returns:
        Symbol at position, or None
    
    Example:
        ```python
        symbol = find_symbol_at_position(source, line=10, character=15)
        if symbol:
            print(f"Found {symbol.kind}: {symbol.name}")
        ```
    """
    api = EditorAPI()
    api.analyze_module(source, uri)
    return api.get_symbol_at_position(uri, Position(line, character))


def get_hover_info(source: str, line: int, character: int, uri: str = "untitled") -> Optional[str]:
    """
    Get hover information for a position in source code.
    
    Args:
        source: Source code
        line: Line number (0-indexed)
        character: Character offset (0-indexed)
        uri: File URI
    
    Returns:
        Markdown-formatted hover text, or None
    
    Example:
        ```python
        hover_text = get_hover_info(source, line=10, character=15)
        if hover_text:
            print(hover_text)
        ```
    """
    api = EditorAPI()
    api.analyze_module(source, uri)
    return api.get_hover_information(uri, Position(line, character))


__all__ = [
    # Data structures
    "Position",
    "Range",
    "Location",
    "Symbol",
    "SymbolReference",
    "Diagnostic",
    "DiagnosticRelatedInformation",
    "AnalysisResult",
    
    # Main API
    "EditorAPI",
    
    # Public functions
    "parse_source",
    "analyze_module",
    "find_symbol_at_position",
    "get_hover_info",
]
