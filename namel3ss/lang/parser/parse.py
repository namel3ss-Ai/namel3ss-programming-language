"""Unified recursive descent parser for N3 language.

This is the single, canonical parser for Namel3ss. It replaces all legacy
parser mechanisms and provides deterministic, production-quality parsing.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import re

from namel3ss.ast.program import Module
from namel3ss.ast.modules import Import
from namel3ss.ast import (
    App, Page, Dataset, Theme, Frame,
)
# AI-related imports
from namel3ss.ast import (
    LLMDefinition as LLM,
    ToolDefinition as Tool,
    Connector,
    Prompt,
    Chain,
    Memory,
)
# RAG and Agent imports
from namel3ss.ast import (
    RagPipelineDefinition as RAGPipeline,
    IndexDefinition as Index,
    AgentDefinition as Agent,
    GraphDefinition as Graph,
)
# Logic imports
from namel3ss.ast import (
    KnowledgeModule,
    LogicQuery,
)
# Policy import
try:
    from namel3ss.ast import PolicyDefinition as Policy
except ImportError:
    Policy = None

from .grammar.lexer import Token, TokenType, tokenize
from .errors import (
    N3SyntaxError, N3SemanticError, N3IndentationError,
    N3DuplicateDeclarationError, N3ReferenceError,
    create_syntax_error,
)
from .declarations import DeclarationParsingMixin
from .expressions import ExpressionParsingMixin


class N3Parser(DeclarationParsingMixin, ExpressionParsingMixin):
    """
    Unified recursive descent parser for N3 language.
    
    This is the ONLY parser for N3. No fallback mechanisms, no dual-parser
    logic. Everything must go through this single, deterministic parser.
    
    Grammar rules follow the EBNF specification in docs/GRAMMAR.md.
    """
    
    def __init__(self, source: str, *, path: str = "", module_name: Optional[str] = None):
        """Initialize parser with source code."""
        self.source = source
        self.path = path
        self.module_name_override = module_name
        
        # Tokenize source
        self.tokens = tokenize(source, path)
        self.pos = 0
        
        # Parser state
        self.module_name: Optional[str] = None
        self.language_version: Optional[str] = None
        self.imports: List[Import] = []
        self.declarations: List[Any] = []
        self.app: Optional[App] = None
        self.explicit_app = False
        
        # Symbol table for semantic validation
        self.symbols: Dict[str, int] = {}  # name -> line number
    
    # ====================================================================
    # Token Management
    # ====================================================================
    
    def peek(self, offset: int = 0) -> Optional[Token]:
        """Peek at token without consuming."""
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return None
    
    def current(self) -> Optional[Token]:
        """Get current token."""
        return self.peek(0)
    
    def advance(self) -> Token:
        """Consume and return current token."""
        token = self.current()
        if token is None:
            raise self.error("Unexpected end of file")
        self.pos += 1
        return token
    
    def expect(self, *types: TokenType) -> Token:
        """Expect one of the given token types and consume it."""
        token = self.current()
        if token is None:
            raise self.error(
                f"Expected {' or '.join(t.name for t in types)}, got end of file"
            )
        
        if token.type not in types:
            expected = [t.name.lower().replace('_', ' ') for t in types]
            raise create_syntax_error(
                f"Unexpected token",
                path=self.path,
                line=token.line,
                column=token.column,
                expected=expected,
                found=token.type.name.lower().replace('_', ' '),
                suggestion=self._suggest_token_fix(token, types),
            )
        
        return self.advance()
    
    def match(self, *types: TokenType) -> bool:
        """Check if current token matches any of the given types."""
        token = self.current()
        return token is not None and token.type in types
    
    def check(self, token_type: TokenType) -> bool:
        """Check if current token matches the given type (alias for match)."""
        return self.match(token_type)
    
    def consume_if(self, *types: TokenType) -> Optional[Token]:
        """Consume token if it matches any of the given types."""
        if self.match(*types):
            return self.advance()
        return None
    
    def skip_newlines(self) -> None:
        """Skip any newline tokens."""
        while self.match(TokenType.NEWLINE):
            self.advance()
    
    def error(self, message: str, suggestion: Optional[str] = None) -> N3SyntaxError:
        """Create a syntax error at current position."""
        token = self.current()
        if token:
            return create_syntax_error(
                message,
                path=self.path,
                line=token.line,
                column=token.column,
                suggestion=suggestion,
            )
        else:
            return create_syntax_error(
                message,
                path=self.path,
                suggestion=suggestion,
            )
    
    def _suggest_token_fix(self, token: Token, expected: tuple[TokenType, ...]) -> Optional[str]:
        """Suggest a fix for unexpected token."""
        # Common fixes
        if TokenType.LBRACE in expected and token.type == TokenType.COLON:
            return "Use '{' instead of ':' for block syntax"
        
        if TokenType.STRING in expected and token.type == TokenType.IDENTIFIER:
            return f"Did you mean \"{token.value}\"?"
        
        return None
    
    # ====================================================================
    # Symbol Table Management
    # ====================================================================
    
    def declare_symbol(self, name: str, line: int) -> None:
        """Declare a new symbol, checking for duplicates."""
        if name in self.symbols:
            raise N3DuplicateDeclarationError(
                message=f"Symbol '{name}' already declared",
                path=self.path,
                line=line,
                name=name,
                first_line=self.symbols[name],
            )
        self.symbols[name] = line
    
    def check_reference(self, name: str, line: int) -> None:
        """Check if a symbol is declared."""
        if name not in self.symbols:
            available = list(self.symbols.keys())
            raise N3ReferenceError(
                message=f"Undefined reference: '{name}'",
                path=self.path,
                line=line,
                name=name,
                available=available,
            )
    
    # ====================================================================
    # High-Level Parsing
    # ====================================================================
    
    def parse(self) -> Module:
        """
        Parse entire module.
        
        Grammar:
            Module = [ ModuleDirectives ] , { TopLevelDecl } ;
        """
        self.skip_newlines()
        
        # Parse module directives (module, import, language_version)
        self.parse_module_directives()
        
        # Parse top-level declarations
        while not self.match(TokenType.EOF):
            self.skip_newlines()
            
            if self.match(TokenType.EOF):
                break
            
            decl = self.parse_top_level_declaration()
            if decl:
                self.declarations.append(decl)
            
            self.skip_newlines()
        
        # Build module
        return self.build_module()
    
    def parse_module_directives(self) -> None:
        """
        Parse module-level directives.
        
        Grammar:
            ModuleDirectives = { ModuleDirective } ;
            ModuleDirective = ModuleDecl | ImportDecl | LanguageVersion ;
        """
        while True:
            self.skip_newlines()
            
            if self.match(TokenType.MODULE):
                self.parse_module_declaration()
            elif self.match(TokenType.IMPORT):
                self.parse_import_declaration()
            elif self.match(TokenType.LANGUAGE_VERSION):
                self.parse_language_version()
            else:
                break
    
    def parse_module_declaration(self) -> None:
        """
        Parse module declaration.
        
        Grammar:
            ModuleDecl = "module" , ( STRING_LITERAL | DottedIdentifier ) ;
        """
        module_token = self.expect(TokenType.MODULE)
        
        if self.match(TokenType.STRING):
            name_value = self.advance().value
        elif self._can_start_identifier_segment():
            name_value = self._parse_dotted_identifier(allow_keywords=True)
        else:
            token = self.current()
            expected = ["string literal", "identifier"]
            raise create_syntax_error(
                "Expected module name",
                path=self.path,
                line=token.line if token else None,
                column=token.column if token else None,
                expected=expected,
                found=token.type.name.lower().replace('_', ' ') if token else "end of file",
            )
        
        if self.module_name is not None:
            raise create_syntax_error(
                "Module can only be declared once",
                path=self.path,
                line=module_token.line,
            )
        
        self.module_name = name_value
        self.skip_newlines()

    _IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")

    def _can_start_identifier_segment(self) -> bool:
        token = self.current()
        if token is None:
            return False
        if token.type == TokenType.IDENTIFIER:
            return True
        return bool(token.value and self._IDENTIFIER_PATTERN.match(token.value))

    def _consume_identifier_segment(self, *, allow_keywords: bool = False) -> str:
        token = self.current()
        if token is None:
            raise self.error("Expected identifier")
        if token.type == TokenType.IDENTIFIER or (allow_keywords and self._IDENTIFIER_PATTERN.match(token.value)):
            self.advance()
            return token.value
        raise create_syntax_error(
            "Expected identifier",
            path=self.path,
            line=token.line,
            column=token.column,
            expected=["identifier"],
            found=token.type.name.lower().replace('_', ' '),
        )

    def _parse_dotted_identifier(self, *, allow_keywords: bool = False) -> str:
        """Parse identifier segments separated by dots into a dotted path."""
        parts = [self._consume_identifier_segment(allow_keywords=allow_keywords)]
        while self.consume_if(TokenType.DOT):
            parts.append(self._consume_identifier_segment(allow_keywords=allow_keywords))
        return ".".join(parts)
    
    def parse_import_declaration(self) -> None:
        """
        Parse import declaration.
        
        Grammar:
            ImportDecl = "import" , ImportPath , [ "as" , IDENTIFIER ] ;
            ImportPath = IDENTIFIER , { "." , IDENTIFIER } ;
        """
        import_token = self.expect(TokenType.IMPORT)
        
        # Parse import path (e.g., ai.models)
        path_parts = [self.expect(TokenType.IDENTIFIER).value]
        
        while self.consume_if(TokenType.DOT):
            path_parts.append(self.expect(TokenType.IDENTIFIER).value)
        
        import_path = ".".join(path_parts)
        
        # Optional alias
        alias = None
        if self.consume_if(TokenType.AS):
            alias = self.expect(TokenType.IDENTIFIER).value
        
        self.imports.append(Import(
            module=import_path,
            alias=alias,
        ))
        
        self.skip_newlines()
    
    def parse_language_version(self) -> None:
        """
        Parse language version directive.
        
        Grammar:
            LanguageVersion = "language_version" , ":" , STRING_LITERAL ;
        """
        version_token = self.expect(TokenType.LANGUAGE_VERSION)
        self.expect(TokenType.COLON)
        version_value = self.expect(TokenType.STRING)
        
        if self.language_version is not None:
            raise create_syntax_error(
                "language_version can only be declared once",
                path=self.path,
                line=version_token.line,
            )
        
        self.language_version = version_value.value
        self.skip_newlines()
    
    def parse_top_level_declaration(self) -> Optional[Any]:
        """
        Parse a top-level declaration.
        
        Grammar:
            TopLevelDecl = AppDecl | PageDecl | LLMDecl | AgentDecl | ... ;
            
        All declarations (except App itself) are automatically attached to the App object.
        If no explicit app declaration exists, an implicit App is created on first use.
        """
        token = self.current()
        if not token:
            return None
        
        # Dispatch based on keyword
        dispatch = {
            TokenType.APP: self.parse_app_declaration,
            TokenType.PAGE: self.parse_page_declaration,
            TokenType.LLM: self.parse_llm_declaration,
            TokenType.AGENT: self.parse_agent_declaration,
            TokenType.PROMPT: self.parse_prompt_declaration,
            TokenType.CHAIN: self.parse_chain_declaration,
            TokenType.RAG_PIPELINE: self.parse_rag_pipeline_declaration,
            TokenType.INDEX: self.parse_index_declaration,
            TokenType.DATASET: self.parse_dataset_declaration,
            TokenType.MEMORY: self.parse_memory_declaration,
            TokenType.FN: self.parse_function_declaration,
            TokenType.TOOL: self.parse_tool_declaration,
            TokenType.CONNECTOR: self.parse_connector_declaration,
            TokenType.TEMPLATE: self.parse_template_declaration,
            TokenType.MODEL: self.parse_model_declaration,
            TokenType.TRAINING: self.parse_training_declaration,
            TokenType.POLICY: self.parse_policy_declaration,
            TokenType.GRAPH: self.parse_graph_declaration,
            TokenType.KNOWLEDGE: self.parse_knowledge_declaration,
            TokenType.QUERY: self.parse_query_declaration,
            TokenType.FRAME: self.parse_frame_declaration,
            TokenType.THEME: self.parse_theme_declaration,
        }
        
        parser_func = dispatch.get(token.type)
        if parser_func:
            decl = parser_func()
            
            # Attach declaration to App (except for App itself)
            if token.type != TokenType.APP and decl is not None:
                self._attach_to_app(decl, token.type)
            
            return decl
        
        # Handle legacy "connectors" block (identifier)
        if token.type == TokenType.IDENTIFIER and token.value == "connectors":
            # Consume the identifier and parse a generic block, attach to app.metadata
            self.advance()
            config = self.parse_block()
            app = self._ensure_app()
            metadata = getattr(app, "metadata", {}) or {}
            metadata["connectors"] = config
            app.metadata = metadata
            return None
        
        # Unknown declaration - provide helpful error message
        suggestion = "Expected a declaration keyword like 'app', 'page', 'llm', etc."
        
        # Provide more specific guidance based on what was found
        found_token = token.value.lower() if token.value else ""
        
        # Try to detect common legacy syntax patterns
        if found_token.endswith('.') and '"' in found_token:
            suggestion = "Found legacy dot syntax. Modern syntax uses braces: 'app \"Name\" { ... }' instead of 'app \"Name\".'."
        elif '"' in found_token and found_token.endswith(':'):
            suggestion = "Found legacy colon syntax. Modern syntax uses braces: 'page \"Name\" { ... }' instead of 'page \"Name\":'"
        elif any(keyword in found_token for keyword in ['show', 'filter', 'columns', 'description']):
            suggestion = "This looks like content that should be inside a declaration block. Modern syntax requires braces: '{ ... }'"
        elif found_token in ['text', 'table', 'chart', 'form']:
            suggestion = "This looks like a page component. Components must be inside a page declaration: 'page \"Name\" { show text \"...\" }'"
        elif any(keyword in found_token for keyword in ['frame', 'dataset', 'insight']):
            suggestion = f"'{found_token}' should be followed by a name in quotes and braces. Example: '{found_token} \"MyName\" {{ ... }}'"
        elif len(found_token) > 0 and found_token[0] != '"' and not found_token.isalpha():
            suggestion = "Invalid token at start of declaration. Declarations should start with keywords like 'app', 'page', etc."
        
        raise create_syntax_error(
            f"Unexpected top-level declaration",
            path=self.path,
            line=token.line,
            found=token.value,
            suggestion=suggestion,
        )
    
    def build_module(self) -> Module:
        """Build the final Module AST node."""
        # Determine module name
        final_module_name = (
            self.module_name_override
            or self.module_name
            or self.path.split('/')[-1].replace('.n3', '')
            or "main"
        )
        
        # Ensure we have an App (create implicit one if needed)
        if self.app is None and self.declarations:
            # If we have declarations but no explicit app, create an implicit one
            self.app = App(name=final_module_name or "app")
            self.explicit_app = False
        
        # Build body with app first if it exists
        body = []
        if self.app:
            body.append(self.app)
        body.extend(self.declarations)
        
        return Module(
            name=final_module_name,
            body=body,
            imports=self.imports,
            language_version=self.language_version,
            has_explicit_app=self.explicit_app,
        )
    
    def _ensure_app(self) -> App:
        """Ensure an App exists, creating an implicit one if needed."""
        if self.app is None:
            # Create implicit app with module name or default
            app_name = self.module_name or self.path.split('/')[-1].replace('.n3', '') or "app"
            self.app = App(name=app_name)
            self.explicit_app = False
        return self.app
    
    def _attach_to_app(self, decl: Any, token_type: TokenType) -> None:
        """
        Attach a parsed declaration to the appropriate App collection.
        
        This is the central wiring mechanism that ensures all declarations
        end up in the correct App field for downstream processing.
        """
        if decl is None:
            return
        
        # Ensure we have an App to attach to
        app = self._ensure_app()
        
        # Map token types to App collection names
        # This mapping must stay in sync with App dataclass fields
        collection_map = {
            TokenType.PAGE: 'pages',
            TokenType.DATASET: 'datasets',
            TokenType.FRAME: 'frames',
            TokenType.CONNECTOR: 'connectors',
            TokenType.LLM: 'llms',
            TokenType.TOOL: 'tools',
            TokenType.PROMPT: 'prompts',
            TokenType.MEMORY: 'memories',
            TokenType.TEMPLATE: 'templates',
            TokenType.CHAIN: 'chains',
            TokenType.MODEL: 'models',
            TokenType.AGENT: 'agents',
            TokenType.GRAPH: 'graphs',
            TokenType.RAG_PIPELINE: 'rag_pipelines',
            TokenType.INDEX: 'indices',
            TokenType.POLICY: 'policies',
            TokenType.KNOWLEDGE: 'knowledge_modules',
            TokenType.QUERY: 'queries',
            TokenType.FN: 'functions',
            TokenType.TRAINING: 'training_jobs',
            TokenType.THEME: None,  # Theme is special - not a collection
        }
        
        collection_name = collection_map.get(token_type)
        
        if collection_name is None:
            # For declarations that don't have a direct mapping (like theme),
            # or for future extensibility, keep them in declarations list
            return
        
        # Get the target collection from the App
        try:
            collection = getattr(app, collection_name)
        except AttributeError:
            # Collection doesn't exist on App - this shouldn't happen if our
            # mapping is correct, but handle gracefully
            raise create_syntax_error(
                f"Internal error: App has no collection '{collection_name}' for {token_type.name}",
                path=self.path,
                suggestion=f"Check that App dataclass has a '{collection_name}' field",
            )
        
        # Attach the declaration
        collection.append(decl)


__all__ = ["N3Parser"]
