"""Lexical analyzer (tokenizer) for N3 language.

Converts source text into a stream of tokens for parsing.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Iterator
import re


class TokenType(Enum):
    """Token types for N3 language."""
    
    # Literals
    STRING = auto()
    NUMBER = auto()
    BOOLEAN = auto()
    NULL = auto()
    
    # Identifiers and Keywords
    IDENTIFIER = auto()
    
    # Keywords - Module directives
    MODULE = auto()
    IMPORT = auto()
    AS = auto()
    LANGUAGE_VERSION = auto()
    
    # Keywords - Declarations
    APP = auto()
    PAGE = auto()
    LLM = auto()
    AGENT = auto()
    PROMPT = auto()
    CHAIN = auto()
    STEP = auto()
    RAG_PIPELINE = auto()
    INDEX = auto()
    DATASET = auto()
    MEMORY = auto()
    FUNCTION = auto()
    FN = auto()
    TOOL = auto()
    CONNECTOR = auto()
    TEMPLATE = auto()
    MODEL = auto()
    TRAINING = auto()
    POLICY = auto()
    GRAPH = auto()
    KNOWLEDGE = auto()
    QUERY = auto()
    FRAME = auto()
    THEME = auto()
    
    # Keywords - Control flow
    IF = auto()
    ELSE = auto()
    FOR = auto()
    WHILE = auto()
    MATCH = auto()
    CASE = auto()
    
    # Keywords - Expressions
    LET = auto()
    IN = auto()
    TRUE = auto()
    FALSE = auto()
    ENV = auto()
    
    # Keywords - Page components
    SHOW = auto()
    FILTER = auto()
    MAP = auto()
    TRANSFORM = auto()
    
    # Keywords - Connections
    CONNECTS = auto()
    TO = auto()
    AT = auto()
    FROM = auto()
    BY = auto()
    
    # Keywords - Database types
    POSTGRES = auto()
    MYSQL = auto()
    MONGODB = auto()
    TABLE = auto()
    
    # Keywords - Inline blocks
    PYTHON = auto()
    REACT = auto()
    
    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    POWER = auto()
    
    EQ = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    
    AND = auto()
    OR = auto()
    NOT = auto()
    
    ARROW = auto()
    FAT_ARROW = auto()
    PIPE = auto()
    AMPERSAND = auto()
    
    ASSIGN = auto()
    
    # Punctuation
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LPAREN = auto()
    RPAREN = auto()
    
    COLON = auto()
    SEMICOLON = auto()
    COMMA = auto()
    DOT = auto()
    ELLIPSIS = auto()
    
    # Special
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    EOF = auto()
    
    # Comments (usually skipped)
    COMMENT = auto()


@dataclass
class Token:
    """A single token with position information."""
    
    type: TokenType
    value: str
    line: int
    column: int
    
    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.column})"


# Keyword mapping
KEYWORDS = {
    "module": TokenType.MODULE,
    "import": TokenType.IMPORT,
    "as": TokenType.AS,
    "language_version": TokenType.LANGUAGE_VERSION,
    
    "app": TokenType.APP,
    "page": TokenType.PAGE,
    "llm": TokenType.LLM,
    "agent": TokenType.AGENT,
    "prompt": TokenType.PROMPT,
    "chain": TokenType.CHAIN,
    "step": TokenType.STEP,
    "rag_pipeline": TokenType.RAG_PIPELINE,
    "index": TokenType.INDEX,
    "dataset": TokenType.DATASET,
    "memory": TokenType.MEMORY,
    "function": TokenType.FUNCTION,
    "fn": TokenType.FN,
    "tool": TokenType.TOOL,
    "connector": TokenType.CONNECTOR,
    "template": TokenType.TEMPLATE,
    "model": TokenType.MODEL,
    "training": TokenType.TRAINING,
    "policy": TokenType.POLICY,
    "graph": TokenType.GRAPH,
    "knowledge": TokenType.KNOWLEDGE,
    "query": TokenType.QUERY,
    "frame": TokenType.FRAME,
    "theme": TokenType.THEME,
    
    "if": TokenType.IF,
    "else": TokenType.ELSE,
    "for": TokenType.FOR,
    "while": TokenType.WHILE,
    "match": TokenType.MATCH,
    "case": TokenType.CASE,
    
    "let": TokenType.LET,
    "in": TokenType.IN,
    "true": TokenType.TRUE,
    "false": TokenType.FALSE,
    "null": TokenType.NULL,
    "env": TokenType.ENV,
    
    "show": TokenType.SHOW,
    "filter": TokenType.FILTER,
    "map": TokenType.MAP,
    "transform": TokenType.TRANSFORM,
    
    "connects": TokenType.CONNECTS,
    "to": TokenType.TO,
    "at": TokenType.AT,
    "from": TokenType.FROM,
    "by": TokenType.BY,
    
    "postgres": TokenType.POSTGRES,
    "mysql": TokenType.MYSQL,
    "mongodb": TokenType.MONGODB,
    "table": TokenType.TABLE,
    
    # Inline blocks
    "python": TokenType.PYTHON,
    "react": TokenType.REACT,
}


class Lexer:
    """Tokenizer for N3 source code."""
    
    def __init__(self, source: str, path: str = ""):
        """Initialize lexer with source code."""
        self.source = source
        self.path = path
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
        
        # Track indentation levels for INDENT/DEDENT
        self.indent_stack = [0]
        self.at_line_start = True
    
    def error(self, message: str) -> Exception:
        """Create a lexer error."""
        from ..errors import N3SyntaxError
        return N3SyntaxError(
            message=message,
            path=self.path,
            line=self.line,
            column=self.column,
        )
    
    def peek(self, offset: int = 0) -> Optional[str]:
        """Peek at character without consuming."""
        pos = self.pos + offset
        if pos < len(self.source):
            return self.source[pos]
        return None
    
    def advance(self) -> Optional[str]:
        """Consume and return current character."""
        if self.pos >= len(self.source):
            return None
        
        char = self.source[self.pos]
        self.pos += 1
        
        if char == '\n':
            self.line += 1
            self.column = 1
            self.at_line_start = True
        else:
            self.column += 1
        
        return char
    
    def skip_whitespace(self) -> None:
        """Skip spaces and tabs (but not newlines)."""
        while self.peek() in (' ', '\t'):
            self.advance()
    
    def skip_comment(self) -> None:
        """Skip line comments starting with # or //."""
        if self.peek() == '#':
            while self.peek() and self.peek() != '\n':
                self.advance()
        elif self.peek() == '/' and self.peek(1) == '/':
            self.advance()  # /
            self.advance()  # /
            while self.peek() and self.peek() != '\n':
                self.advance()
    
    def read_string(self) -> str:
        """Read a string literal."""
        quote = self.advance()  # " or '
        chars = []
        
        # Check for triple-quoted string
        if self.peek() == quote and self.peek(1) == quote:
            self.advance()  # second quote
            self.advance()  # third quote
            # Read until closing triple quote
            while True:
                if self.peek() is None:
                    raise self.error("Unterminated triple-quoted string")
                if self.peek() == quote and self.peek(1) == quote and self.peek(2) == quote:
                    self.advance()
                    self.advance()
                    self.advance()
                    break
                chars.append(self.advance())
            return ''.join(chars)
        
        # Regular string
        while True:
            char = self.peek()
            if char is None:
                raise self.error(f"Unterminated string literal")
            if char == quote:
                self.advance()
                break
            if char == '\\':
                self.advance()
                escape = self.advance()
                if escape == 'n':
                    chars.append('\n')
                elif escape == 't':
                    chars.append('\t')
                elif escape == 'r':
                    chars.append('\r')
                elif escape in ('"', "'", '\\', '{', '}'):
                    chars.append(escape)
                else:
                    chars.append(escape)
            else:
                chars.append(self.advance())
        
        return ''.join(chars)
    
    def read_number(self) -> str:
        """Read a numeric literal."""
        chars = []
        
        # Optional minus sign
        if self.peek() == '-':
            chars.append(self.advance())
        
        # Integer part
        while self.peek() and self.peek().isdigit():
            chars.append(self.advance())
        
        # Fractional part
        if self.peek() == '.' and self.peek(1) and self.peek(1).isdigit():
            chars.append(self.advance())  # .
            while self.peek() and self.peek().isdigit():
                chars.append(self.advance())
        
        # Exponent
        if self.peek() and self.peek().lower() == 'e':
            chars.append(self.advance())  # e
            if self.peek() in ('+', '-'):
                chars.append(self.advance())
            while self.peek() and self.peek().isdigit():
                chars.append(self.advance())
        
        return ''.join(chars)
    
    def read_identifier(self) -> str:
        """Read an identifier or keyword."""
        chars = []
        while self.peek() and (self.peek().isalnum() or self.peek() == '_'):
            chars.append(self.advance())
        return ''.join(chars)
    
    def add_token(self, token_type: TokenType, value: str = "") -> None:
        """Add a token to the list."""
        self.tokens.append(Token(
            type=token_type,
            value=value,
            line=self.line,
            column=self.column - len(value),
        ))
    
    def tokenize(self) -> List[Token]:
        """Tokenize the entire source."""
        while self.pos < len(self.source):
            # Handle indentation at line start
            if self.at_line_start:
                self.handle_indentation()
                self.at_line_start = False
            
            # Skip whitespace (not newlines)
            self.skip_whitespace()
            
            # Check for EOF
            if self.pos >= len(self.source):
                break
            
            # Skip comments
            if self.peek() in ('#',) or (self.peek() == '/' and self.peek(1) == '/'):
                self.skip_comment()
                continue
            
            char = self.peek()
            
            # Newline
            if char == '\n':
                self.add_token(TokenType.NEWLINE, '\n')
                self.advance()
                continue
            
            # String literals
            if char in ('"', "'"):
                start_col = self.column
                value = self.read_string()
                self.add_token(TokenType.STRING, value)
                continue
            
            # Numbers
            if char.isdigit() or (char == '-' and self.peek(1) and self.peek(1).isdigit()):
                value = self.read_number()
                self.add_token(TokenType.NUMBER, value)
                continue
            
            # Identifiers and keywords
            if char.isalpha() or char == '_':
                value = self.read_identifier()
                token_type = KEYWORDS.get(value, TokenType.IDENTIFIER)
                self.add_token(token_type, value)
                continue
            
            # Two-character operators
            two_char = char + (self.peek(1) or '')
            if two_char == '**':
                self.add_token(TokenType.POWER, '**')
                self.advance()
                self.advance()
                continue
            elif two_char == '==':
                self.add_token(TokenType.EQ, '==')
                self.advance()
                self.advance()
                continue
            elif two_char == '!=':
                self.add_token(TokenType.NE, '!=')
                self.advance()
                self.advance()
                continue
            elif two_char == '<=':
                self.add_token(TokenType.LE, '<=')
                self.advance()
                self.advance()
                continue
            elif two_char == '>=':
                self.add_token(TokenType.GE, '>=')
                self.advance()
                self.advance()
                continue
            elif two_char == '&&':
                self.add_token(TokenType.AND, '&&')
                self.advance()
                self.advance()
                continue
            elif two_char == '||':
                self.add_token(TokenType.OR, '||')
                self.advance()
                self.advance()
                continue
            elif two_char == '=>':
                self.add_token(TokenType.FAT_ARROW, '=>')
                self.advance()
                self.advance()
                continue
            elif two_char == '->':
                self.add_token(TokenType.ARROW, '->')
                self.advance()
                self.advance()
                continue
            
            # Three-character operators
            if char == '.' and self.peek(1) == '.' and self.peek(2) == '.':
                self.add_token(TokenType.ELLIPSIS, '...')
                self.advance()
                self.advance()
                self.advance()
                continue
            
            # Single-character operators and punctuation
            char_tokens = {
                '+': TokenType.PLUS,
                '-': TokenType.MINUS,
                '*': TokenType.STAR,
                '/': TokenType.SLASH,
                '%': TokenType.PERCENT,
                '<': TokenType.LT,
                '>': TokenType.GT,
                '!': TokenType.NOT,
                '=': TokenType.ASSIGN,
                '|': TokenType.PIPE,
                '&': TokenType.AMPERSAND,
                '{': TokenType.LBRACE,
                '}': TokenType.RBRACE,
                '[': TokenType.LBRACKET,
                ']': TokenType.RBRACKET,
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                ':': TokenType.COLON,
                ';': TokenType.SEMICOLON,
                ',': TokenType.COMMA,
                '.': TokenType.DOT,
            }
            
            if char in char_tokens:
                self.add_token(char_tokens[char], char)
                self.advance()
                continue
            
            # Skip special characters that might appear in templates/strings
            if char in ('$', '@'):
                self.advance()
                continue
            
            # Unknown character
            raise self.error(f"Unexpected character: {char!r}")
        
        # Close any remaining indentation levels
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            self.add_token(TokenType.DEDENT, '')
        
        # Add EOF token
        self.add_token(TokenType.EOF, '')
        
        return self.tokens
    
    def handle_indentation(self) -> None:
        """Handle indentation at the start of a line."""
        # Count leading spaces
        indent_level = 0
        while self.peek() in (' ', '\t'):
            if self.peek() == ' ':
                indent_level += 1
            else:  # tab
                indent_level += 4  # treat tab as 4 spaces
            self.advance()
        
        # Skip blank lines and comments
        if self.peek() in ('\n', '#', None) or (self.peek() == '/' and self.peek(1) == '/'):
            return
        
        current_indent = self.indent_stack[-1]
        
        if indent_level > current_indent:
            self.indent_stack.append(indent_level)
            self.add_token(TokenType.INDENT, '')
        elif indent_level < current_indent:
            while len(self.indent_stack) > 1 and self.indent_stack[-1] > indent_level:
                self.indent_stack.pop()
                self.add_token(TokenType.DEDENT, '')
            
            if self.indent_stack[-1] != indent_level:
                raise self.error(f"Inconsistent indentation: {indent_level} spaces")


def tokenize(source: str, path: str = "") -> List[Token]:
    """Tokenize N3 source code."""
    lexer = Lexer(source, path)
    return lexer.tokenize()


__all__ = ["Token", "TokenType", "Lexer", "tokenize"]
