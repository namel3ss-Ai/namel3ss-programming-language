"""
Logic parser for Namel3ss knowledge and inference constructs.

Provides parsing for:
- Logic terms (variables, atoms, numbers, strings, compound structures)
- Facts and rules
- Knowledge modules
- Queries
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from namel3ss.ast.logic import (
    LogicAtom,
    LogicFact,
    LogicList,
    LogicNumber,
    LogicQuery,
    LogicRule,
    LogicString,
    LogicStruct,
    LogicTerm,
    LogicVar,
    KnowledgeModule,
)
from namel3ss.parser.base import ParserBase
# KeywordRegistry import removed - class does not exist


class LogicParserMixin(ParserBase):
    """
    Mixin for parsing logic programming constructs.
    
    This parser handles Prolog-style syntax for declarative knowledge representation,
    including facts, rules, queries, and knowledge modules. The logic system supports
    symbolic reasoning and inference over structured data.
    
    Syntax Example:
        knowledge "family":
            fact parent(alice, bob).
            fact parent(bob, carol).
            rule ancestor(X, Y) :- parent(X, Y).
            rule ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
        
        query "find_ancestors":
            from knowledge "family":
                goal ancestor(X, carol).
                limit: 10
    
    Features:
        - Prolog-style terms: variables (X, _Var), atoms (alice), numbers (42), strings ("text")
        - Compound structures: functor(arg1, arg2, ...)
        - Lists with pattern matching: [H|T], [1, 2, 3]
        - Facts: ground truths (parent(alice, bob))
        - Rules: inference patterns (ancestor(X, Y) :- parent(X, Y))
        - Queries: goal-directed search with knowledge sources
        - Knowledge modules: organized collections of facts and rules
    
    Supported Constructs:
        - knowledge module declarations
        - fact statements
        - rule definitions with body goals
        - query specifications with goals and constraints
    """
    
    # ========================================================================
    # Term Parsing
    # ========================================================================
    
    def _parse_logic_term(self, text: str, line_no: int = 0) -> LogicTerm:
        """
        Parse a logic term from text.
        
        Terms can be:
        - Variables: start with uppercase or underscore (X, _Result, Var1)
        - Atoms: lowercase identifiers (alice, nil, true)
        - Numbers: integers or floats (42, 3.14)
        - Strings: quoted strings ("hello", 'world')
        - Structures: functor(arg1, arg2, ...) 
        - Lists: [element1, element2, ...]
        """
        text = text.strip()
        
        if not text:
            raise self._error("Empty term", line_no, text)
        
        # Variable: starts with uppercase or underscore
        if text[0].isupper() or text[0] == '_':
            if self._is_valid_variable_name(text):
                return LogicVar(name=text, line=line_no)
        
        # Number: integer or float
        if text[0].isdigit() or (text[0] == '-' and len(text) > 1 and text[1].isdigit()):
            try:
                if '.' in text:
                    return LogicNumber(value=float(text), line=line_no)
                else:
                    return LogicNumber(value=int(text), line=line_no)
            except ValueError:
                pass  # Fall through to other parsers
        
        # String: quoted
        if text[0] in ('"', "'"):
            if len(text) >= 2 and text[-1] == text[0]:
                return LogicString(value=text[1:-1], line=line_no)
            raise self._error(f"Unterminated string: {text}", line_no, text)
        
        # List: [...]
        if text.startswith('['):
            return self._parse_logic_list(text, line_no)
        
        # Structure or atom
        if '(' in text:
            # Compound structure: functor(args)
            return self._parse_logic_struct(text, line_no)
        else:
            # Simple atom
            if self._is_valid_atom_name(text):
                return LogicAtom(value=text, line=line_no)
            raise self._error(f"Invalid atom: {text}", line_no, text)
    
    def _parse_logic_struct(self, text: str, line_no: int) -> LogicStruct:
        """Parse a compound structure like parent(alice, bob)."""
        text = text.strip()
        
        # Find the functor (everything before first '(')
        paren_pos = text.index('(')
        functor = text[:paren_pos].strip()
        
        if not self._is_valid_functor_name(functor):
            raise self._error(f"Invalid functor name: {functor}", line_no, text)
        
        # Extract arguments between parentheses
        if not text.endswith(')'):
            raise self._error(f"Structure not properly closed: {text}", line_no, text)
        
        args_text = text[paren_pos + 1:-1].strip()
        
        if not args_text:
            # Zero-arity structure (treated as atom-like)
            return LogicStruct(functor=functor, args=[], line=line_no)
        
        # Parse comma-separated arguments
        args = self._split_logic_args(args_text)
        parsed_args = [self._parse_logic_term(arg, line_no) for arg in args]
        
        return LogicStruct(functor=functor, args=parsed_args, line=line_no)
    
    def _parse_logic_list(self, text: str, line_no: int) -> LogicList:
        """Parse a list like [1, 2, 3] or [H|T]."""
        text = text.strip()
        
        if not text.startswith('[') or not text.endswith(']'):
            raise self._error(f"Invalid list syntax: {text}", line_no, text)
        
        inner = text[1:-1].strip()
        
        if not inner:
            # Empty list
            return LogicList(elements=[], line=line_no)
        
        # Check for list pattern with tail: [H|T]
        if '|' in inner:
            parts = inner.split('|', 1)
            if len(parts) != 2:
                raise self._error(f"Invalid list pattern: {text}", line_no, text)
            
            elements_text = parts[0].strip()
            tail_text = parts[1].strip()
            
            # Parse head elements
            if elements_text:
                element_strs = self._split_logic_args(elements_text)
                elements = [self._parse_logic_term(e, line_no) for e in element_strs]
            else:
                elements = []
            
            # Parse tail (must be a variable)
            tail_term = self._parse_logic_term(tail_text, line_no)
            if not isinstance(tail_term, LogicVar):
                raise self._error(f"List tail must be a variable: {tail_text}", line_no, text)
            
            return LogicList(elements=elements, tail=tail_term, line=line_no)
        
        # Regular list: parse comma-separated elements
        element_strs = self._split_logic_args(inner)
        elements = [self._parse_logic_term(e, line_no) for e in element_strs]
        
        return LogicList(elements=elements, line=line_no)
    
    def _split_logic_args(self, text: str) -> List[str]:
        """
        Split comma-separated arguments, respecting nested structures.
        
        Example: "a, f(b, c), d" -> ["a", "f(b, c)", "d"]
        """
        args = []
        current = []
        depth = 0  # Track nesting depth for (), [], {}
        in_string = False
        string_char = None
        
        for char in text:
            if in_string:
                current.append(char)
                if char == string_char and (not current or current[-2] != '\\'):
                    in_string = False
                    string_char = None
            elif char in ('"', "'"):
                in_string = True
                string_char = char
                current.append(char)
            elif char in ('(', '[', '{'):
                depth += 1
                current.append(char)
            elif char in (')', ']', '}'):
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0:
                # Top-level comma: end of argument
                arg = ''.join(current).strip()
                if arg:
                    args.append(arg)
                current = []
            else:
                current.append(char)
        
        # Add final argument
        arg = ''.join(current).strip()
        if arg:
            args.append(arg)
        
        return args
    
    def _is_valid_variable_name(self, name: str) -> bool:
        """Check if name is a valid variable name."""
        if not name:
            return False
        if name[0] not in ('_',) and not name[0].isupper():
            return False
        return all(c.isalnum() or c == '_' for c in name)
    
    def _is_valid_atom_name(self, name: str) -> bool:
        """Check if name is a valid atom name."""
        if not name:
            return False
        if name[0].isupper() or name[0] == '_':
            return False
        return all(c.isalnum() or c == '_' for c in name)
    
    def _is_valid_functor_name(self, name: str) -> bool:
        """Check if name is a valid functor name."""
        return self._is_valid_atom_name(name)
    
    # ========================================================================
    # Knowledge Module Parsing
    # ========================================================================
    
    def _parse_knowledge_module(self, line: str, line_no: int, base_indent: int) -> KnowledgeModule:
        """
        Parse a knowledge module block.
        
        Knowledge modules are containers for facts and rules that define
        a logical knowledge base for reasoning and inference.
        
        Syntax:
            knowledge "module_name":
                fact parent(alice, bob).
                rule ancestor(X, Y) :- parent(X, Y).
        
        Args:
            line: The knowledge declaration line
            line_no: Current line number
            base_indent: Indentation level of the knowledge declaration
        
        Returns:
            KnowledgeModule AST node
        """
        # Extract module name
        match = re.match(r'knowledge\s+"([^"]+)":', line.strip())
        if not match:
            match = re.match(r'knowledge\s+([a-z_][a-z0-9_]*):', line.strip())
        
        if not match:
            raise self._error(
                'Expected: knowledge "name":',
                line_no,
                line,
                hint='Knowledge modules require a name in quotes, e.g., knowledge "family":'
            )
        
        name = match.group(1)
        module = KnowledgeModule(name=name, line=line_no)
        
        # Parse knowledge elements (facts and rules)
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            
            indent = self._indent(nxt)
            stripped = nxt.strip()
            
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            
            if indent <= base_indent:
                break

            # Centralized indentation validation
            self._expect_indent_greater_than(
                base_indent,
                nxt,
                line_no,
                context="knowledge module body",
            )
            
            # Parse fact or rule
            if stripped.startswith('fact '):
                fact = self._parse_fact(nxt, line_no, indent)
                module.facts.append(fact)
                self._advance()
            elif stripped.startswith('rule '):
                rule, extra_lines = self._parse_rule(nxt, line_no, indent)
                module.rules.append(rule)
                # Consume the rule line plus any continuation lines
                self._advance()
                for _ in range(extra_lines):
                    self._advance()
            elif stripped.startswith('import '):
                # Import another knowledge module
                import_match = re.match(r'import\s+"([^"]+)"', stripped)
                if not import_match:
                    import_match = re.match(r'import\s+([a-z_][a-z0-9_]*)', stripped)
                if import_match:
                    module.imports.append(import_match.group(1))
                    self._advance()
                else:
                    raise self._error(
                        'Expected: import "module_name"',
                        line_no,
                        nxt,
                        hint='Import other knowledge modules with: import "module_name"'
                    )
            elif stripped.startswith('description:'):
                desc_text = stripped[len('description:'):].strip()
                module.description = desc_text
                self._advance()
            else:
                raise self._error(
                    f"Expected 'fact' or 'rule', got: {stripped}",
                    line_no,
                    nxt,
                    hint='Valid knowledge directives: fact, rule, import, description'
                )
        
        return module
    
    def _parse_fact(self, line: str, line_no: int, indent: int) -> LogicFact:
        """
        Parse a fact line.
        
        Example: fact parent(alice, bob).
        """
        stripped = line.strip()
        
        if not stripped.startswith('fact '):
            raise self._error("Expected 'fact'", line_no, line)
        
        fact_text = stripped[5:].strip()  # Remove 'fact '
        
        if not fact_text.endswith('.'):
            raise self._error("Fact must end with '.'", line_no, line)
        
        fact_text = fact_text[:-1].strip()  # Remove trailing '.'
        
        # Parse the head structure
        head = self._parse_logic_term(fact_text, line_no)
        
        if not isinstance(head, LogicStruct):
            raise self._error("Fact head must be a structure", line_no, line)
        
        return LogicFact(head=head, line=line_no)
    
    def _parse_rule(self, line: str, line_no: int, indent: int) -> tuple[LogicRule, int]:
        """
        Parse a rule line.
        
        Example: rule ancestor(X, Y) :- parent(X, Y).
        Example: rule ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
        """
        stripped = line.strip()
        
        if not stripped.startswith('rule '):
            raise self._error("Expected 'rule'", line_no, line)
        
        rule_text = stripped[5:].strip()  # Remove 'rule '

        # Collect any indented continuation lines for multi-line rule bodies
        consumed_continuations = 0
        continuation_parts: List[str] = []
        cursor = self.pos + 1
        while cursor < len(self.lines):
            nxt_line = self.lines[cursor]
            next_stripped = nxt_line.strip()

            # Skip empty/comment lines that are more indented
            if not next_stripped or next_stripped.startswith('#'):
                if self._indent(nxt_line) > indent:
                    consumed_continuations += 1
                    cursor += 1
                    continue
                break

            next_indent = self._indent(nxt_line)
            if next_indent <= indent:
                break

            continuation_parts.append(next_stripped)
            consumed_continuations += 1
            cursor += 1

        if continuation_parts:
            rule_text = " ".join([rule_text] + continuation_parts)

        if not rule_text.endswith('.'):
            raise self._error("Rule must end with '.'", line_no, line)

        rule_text = rule_text[:-1].strip()  # Remove trailing '.'
        
        # Split into head and body using ':-'
        if ':-' in rule_text:
            parts = rule_text.split(':-', 1)
            head_text = parts[0].strip()
            body_text = parts[1].strip()
        else:
            # Rule without body (equivalent to a fact)
            head_text = rule_text
            body_text = ""
        
        # Parse head
        head = self._parse_logic_term(head_text, line_no)
        if not isinstance(head, LogicStruct):
            raise self._error("Rule head must be a structure", line_no, line)
        
        # Parse body (comma-separated goals)
        body = []
        if body_text:
            goal_strs = self._split_logic_args(body_text)
            for goal_str in goal_strs:
                goal = self._parse_logic_term(goal_str, line_no)
                if not isinstance(goal, LogicStruct):
                    raise self._error(f"Rule body goal must be a structure: {goal_str}", line_no, line)
                body.append(goal)
        
        return LogicRule(head=head, body=body, line=line_no), consumed_continuations
    
    # ========================================================================
    # Query Parsing
    # ========================================================================
    
    def _parse_query(self, line: str, line_no: int, base_indent: int) -> LogicQuery:
        """
        Parse a query block.
        
        Queries define goal-directed searches over knowledge bases,
        with optional constraints and result limits.
        
        Syntax:
            query "query_name":
                from knowledge "module":
                    goal functor(Var1, Var2).
                    limit: 10
                    variables: X, Y, Z
        
        Args:
            line: The query declaration line
            line_no: Current line number
            base_indent: Indentation level of the query declaration
        
        Returns:
            LogicQuery AST node
        """
        # Extract query name
        match = re.match(r'query\s+"([^"]+)":', line.strip())
        if not match:
            match = re.match(r'query\s+([a-z_][a-z0-9_]*):', line.strip())
        
        if not match:
            raise self._error(
                'Expected: query "name": (see docs/QUERIES_AND_DATASETS.md; try dataset.filter(...) instead of legacy query blocks)',
                line_no,
                line,
                hint='Queries require a name in quotes, e.g., query "find_ancestors":',
            )
        
        name = match.group(1)
        query = LogicQuery(name=name, line=line_no)
        
        # Parse query configuration
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            
            indent = self._indent(nxt)
            stripped = nxt.strip()
            
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            
            if indent <= base_indent:
                break

            # Centralized indentation validation
            self._expect_indent_greater_than(
                base_indent,
                nxt,
                line_no,
                context="query body",
            )
            
            # Parse query directives
            if stripped.startswith('from knowledge '):
                # Knowledge source
                source_match = re.match(r'from knowledge\s+"([^"]+)":', stripped)
                if not source_match:
                    source_match = re.match(r'from knowledge\s+([a-z_][a-z0-9_]*):', stripped)
                if source_match:
                    query.knowledge_sources.append(source_match.group(1))
                    self._advance()
                else:
                    raise self._error(
                        'Expected: from knowledge "name":',
                        line_no,
                        nxt,
                        hint='Specify knowledge sources with: from knowledge "module_name":'
                    )
            elif stripped.startswith('from dataset '):
                dataset_match = re.match(r'from dataset\s+"([^"]+)":', stripped)
                if not dataset_match:
                    dataset_match = re.match(r'from dataset\s+([a-z_][a-z0-9_]*):', stripped)
                if dataset_match:
                    datasets = query.metadata.setdefault("datasets", [])
                    datasets.append(dataset_match.group(1))
                    self._advance()
                else:
                    raise self._error(
                        'Expected: from dataset "name":',
                        line_no,
                        nxt,
                        hint='Specify dataset adapters with: from dataset "alias":'
                    )
            elif stripped.startswith('goal '):
                # Parse goal
                goal_text = stripped[5:].strip()
                if goal_text.endswith('.'):
                    goal_text = goal_text[:-1].strip()
                
                goal = self._parse_logic_term(goal_text, line_no)
                if not isinstance(goal, LogicStruct):
                    raise self._error(
                        f"Goal must be a structure: {goal_text}",
                        line_no,
                        nxt,
                        hint='Goals must be structured terms like: goal ancestor(X, Y).'
                    )
                query.goals.append(goal)
                self._advance()
            elif stripped.startswith('limit:'):
                # Parse limit
                limit_text = stripped[6:].strip()
                try:
                    query.limit = int(limit_text)
                except ValueError:
                    raise self._error(
                        f"Invalid limit value: {limit_text}",
                        line_no,
                        nxt,
                        hint='Limit must be an integer, e.g., limit: 10'
                    )
                self._advance()
            elif stripped.startswith('variables:'):
                # Parse variable list
                vars_text = stripped[10:].strip()
                query.variables = [v.strip() for v in vars_text.split(',') if v.strip()]
                self._advance()
            else:
                raise self._error(
                    f"Unknown query directive: {stripped}",
                    line_no,
                    nxt,
                    hint='Valid query directives: from knowledge, goal, limit, variables'
                )
        
        return query


__all__ = [
    "LogicParserMixin",
]
