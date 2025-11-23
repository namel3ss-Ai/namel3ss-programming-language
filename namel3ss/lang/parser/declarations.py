"""Declaration parsing methods for N3Parser.

Contains all methods for parsing top-level declarations like app, page, llm, etc.
"""

from typing import Any, Dict, List, Optional
from .grammar.lexer import TokenType
from .errors import create_syntax_error
from .config_filter import (
    build_dataclass_with_config,
    AGENT_ALIASES,
    LLM_ALIASES,
    CHAIN_ALIASES,
    RAG_ALIASES,
    DATASET_ALIASES,
    GRAPH_ALIASES,
)


class DeclarationParsingMixin:
    """Mixin with all declaration parsing methods."""
    
    # This will be mixed into N3Parser, so we have access to all parser methods
    
    def parse_app_declaration(self):
        """
        Parse app declaration.
        
        Grammar:
            AppDecl = "app" , QuotedName , [ AppConnections ] , [ Block ] ;
        """
        from namel3ss.ast import App
        
        app_token = self.expect(TokenType.APP)
        name_token = self.expect(TokenType.STRING)
        
        if self.app is not None:
            raise create_syntax_error(
                "Only one app declaration allowed per module",
                path=self.path,
                line=app_token.line,
            )
        
        name = name_token.value
        self.declare_symbol(name, app_token.line)
        
        # Optional connections - parse but store as database for now
        database = None
        if self.consume_if(TokenType.CONNECTS):
            self.expect(TokenType.TO)
            connections = self.parse_connection_list()
            # Use first postgres connection as database
            for conn in connections:
                if conn.get("type") == "postgres":
                    database = conn.get("name")
                    break
        
        # Legacy dot terminator support
        if self.consume_if(TokenType.DOT):
            config = {}
        # Parse optional block (will contain config like description, version, etc.)
        elif self.match(TokenType.LBRACE):
            config = self.parse_block()
        else:
            config = {}
        
        # Extract database from config if present
        if "database" in config:
            database = config.pop("database")
        
        # Create app with proper fields
        self.app = App(
            name=name,
            database=database,
        )
        self.explicit_app = True
        
        return self.app
    
    def parse_connection_list(self) -> List[Dict[str, str]]:
        """Parse connection list."""
        connections = []
        
        # First connection
        connections.append(self.parse_connection())
        
        # Additional connections
        while self.consume_if(TokenType.COMMA):
            connections.append(self.parse_connection())
        
        return connections
    
    def parse_connection(self) -> Dict[str, str]:
        """Parse a single connection."""
        # Database connections: postgres "db_name"
        db_types = {
            TokenType.POSTGRES: "postgres",
            TokenType.MYSQL: "mysql",
            TokenType.MONGODB: "mongodb",
        }
        
        for token_type, db_type in db_types.items():
            if self.consume_if(token_type):
                name = self.expect(TokenType.STRING).value
                return {"type": db_type, "name": name}
        
        # Generic service connection
        service_type = self.expect(TokenType.IDENTIFIER).value
        name = self.expect(TokenType.STRING).value
        return {"type": service_type, "name": name}
    
    def parse_page_declaration(self):
        """
        Parse page declaration.
        
        Grammar:
            PageDecl = "page" , QuotedName , "at" , STRING_LITERAL , Block ;
        """
        from namel3ss.ast import Page
        
        page_token = self.expect(TokenType.PAGE)
        name_token = self.expect(TokenType.STRING)
        name = name_token.value
        
        self.declare_symbol(f"page:{name}", page_token.line)
        
        route = "/"
        if self.consume_if(TokenType.AT):
            route_token = self.expect(TokenType.STRING)
            route = route_token.value
        
        # Support both colon and brace syntax for page blocks
        statements = []
        if self.consume_if(TokenType.COLON):
            # Legacy colon syntax with indented statements
            self.skip_newlines()
            statements = self._parse_indented_page_statements()
        else:
            # Modern brace syntax
            self.expect(TokenType.LBRACE)
            self.skip_newlines()
            self.consume_if(TokenType.INDENT)  # Skip indent after opening brace
        
            while not self.match(TokenType.RBRACE):
                # Skip dedent tokens
                while self.consume_if(TokenType.DEDENT):
                    pass
                
                # Check again for closing brace after consuming dedents
                if self.match(TokenType.RBRACE):
                    break
                
                # Skip indent tokens
                while self.consume_if(TokenType.INDENT):
                    pass
                
                stmt = self.parse_page_statement()
                if stmt:
                    statements.append(stmt)
                self.skip_newlines()
            
            self.consume_if(TokenType.DEDENT)  # Skip dedent before closing brace
            self.expect(TokenType.RBRACE)
            self.skip_newlines()
        
        return Page(
            name=name,
            route=route,
            body=statements,  # Use 'body' not 'statements' (statements is just a property alias)
        )
    
    def _parse_indented_page_statements(self) -> List[Any]:
        """Parse indented page statements for legacy colon syntax."""
        statements = []
        base_indent = None
        
        while True:
            token = self.current()
            if not token or token.type == TokenType.EOF:
                break
                
            # Calculate current line indentation
            if token.type == TokenType.NEWLINE:
                self.advance()
                continue
                
            # For legacy compatibility, we need to check indentation manually
            # This is a simplified approach - in production we'd want proper indent tracking
            if base_indent is None:
                if token.type == TokenType.INDENT:
                    base_indent = 1
                    self.advance()
                else:
                    # No indentation found, end of page block
                    break
            elif not self.match(TokenType.INDENT, TokenType.IDENTIFIER, TokenType.SHOW):
                # End of indented block
                break
                
            stmt = self.parse_page_statement()
            if stmt:
                statements.append(stmt)
            self.skip_newlines()
            
        return statements
    
    def parse_llm_declaration(self):
        """
        Parse LLM declaration.
        
        Grammar:
            LLMDecl = "llm" , QuotedName , Block ;
            
        Uses config filtering with introspection to safely construct LLMDefinition.
        Unknown fields are automatically routed to the metadata field.
        """
        from namel3ss.ast import LLMDefinition
        
        llm_token = self.expect(TokenType.LLM)
        name_token = self.expect(TokenType.STRING)
        name = name_token.value
        
        self.declare_symbol(f"llm:{name}", llm_token.line)
        
        config = self.parse_block()
        
        # Use config filtering system with LLM aliases
        return build_dataclass_with_config(
            LLMDefinition,
            config=config,
            aliases=LLM_ALIASES,
            name=name,
        )
    
    def parse_agent_declaration(self):
        """
        Parse agent declaration.
        
        Grammar:
            AgentDecl = "agent" , QuotedName , Block ;
            
        Uses config filtering with DSL→AST aliasing:
        - "llm" -> "llm_name"
        - "tools" -> "tool_names"
        - "memory" -> "memory_config"
        - "system" -> "system_prompt"
        
        Unknown fields are routed to the config field.
        """
        from namel3ss.ast import AgentDefinition
        
        agent_token = self.expect(TokenType.AGENT)
        name_token = self.expect(TokenType.STRING)
        name = name_token.value
        
        self.declare_symbol(f"agent:{name}", agent_token.line)
        
        config = self.parse_block()
        
        # Use config filtering system with Agent aliases
        return build_dataclass_with_config(
            AgentDefinition,
            config=config,
            aliases=AGENT_ALIASES,
            name=name,
        )
    
    def parse_prompt_declaration(self):
        """
        Parse prompt declaration with unified config pattern.
        
        Grammar:
            PromptDecl = "prompt" , QuotedName , Block ;
            
        Modern fields supported:
        - args: List[PromptArgument] - Typed arguments (preferred over legacy 'input')
        - output_schema: OutputSchema - Structured output (preferred over legacy 'output')
        - parameters: Dict[str, Any] - Model parameters (temperature, max_tokens, etc.)
        - metadata: Dict[str, Any] - Extra metadata (version, tags, etc.)
        - effects: Set[str] - Effect tracking
        
        Legacy fields (backwards compatible):
        - input: Schema definition -> input_fields
        - output: Schema definition -> output_fields
        
        Special handling:
        - model/llm: Both accepted with validation (model preferred)
        - template: Required prompt template string
        - description: Optional documentation
        - name: Ignored in block (canonical name from declaration)
        """
        from namel3ss.ast import Prompt
        
        prompt_token = self.expect(TokenType.PROMPT)
        name_token = self.expect(TokenType.STRING)
        name = name_token.value
        
        self.declare_symbol(f"prompt:{name}", prompt_token.line)
        
        # Parse block with special handlers for legacy input/output
        self.expect(TokenType.LBRACE)
        
        # Special handlers for legacy schema fields
        def parse_input_schema():
            """Parse legacy input schema -> input_fields"""
            return self.parse_schema_definition()
        
        def parse_output_schema():
            """Parse legacy output schema -> output_fields"""
            return self.parse_schema_definition()
        
        def parse_args_list():
            """Parse modern args list -> args"""
            # Parse array of argument definitions
            return self.parse_value()
        
        def parse_output_schema_def():
            """Parse modern output_schema -> output_schema"""
            # Parse structured output schema definition
            return self.parse_value()
        
        special_handlers = {
            "input": parse_input_schema,
            "output": parse_output_schema,
            "args": parse_args_list,
            "output_schema": parse_output_schema_def,
        }
        
        # Use shared block parsing
        config = self._parse_config_block(
            allow_any_keyword=True,
            special_handlers=special_handlers
        )
        
        self.expect(TokenType.RBRACE)
        self.skip_newlines()
        
        # Extract legacy fields if present
        input_fields = config.pop('input', None)
        output_fields = config.pop('output', None)
        
        # Extract modern fields if present
        args = config.pop('args', None)
        output_schema = config.pop('output_schema', None)
        
        # If legacy fields present, map to their dataclass fields
        if input_fields is not None:
            # Convert dict/str schemas to PromptField-friendly list
            converted_inputs = []
            if isinstance(input_fields, dict):
                for fname, fdef in input_fields.items():
                    converted_inputs.append({"name": fname, **(fdef if isinstance(fdef, dict) else {})})
            elif isinstance(input_fields, list):
                converted_inputs = input_fields
            else:
                converted_inputs = [input_fields]
            config['input_fields'] = converted_inputs
        
        if output_fields is not None:
            converted_outputs = []
            if isinstance(output_fields, dict):
                for fname, fdef in output_fields.items():
                    converted_outputs.append({"name": fname, **(fdef if isinstance(fdef, dict) else {})})
            elif isinstance(output_fields, list):
                converted_outputs = output_fields
            else:
                converted_outputs = [output_fields]
            config['output_fields'] = converted_outputs
        
        # Provide default for template if not present (backwards compatibility)
        if 'template' not in config:
            config['template'] = ""
        # Provide default model if missing to satisfy constructor
        if 'model' not in config:
            config['model'] = ""
        
        # Build with unified config pattern
        return build_dataclass_with_config(
            Prompt,
            config,
            declared_name=name,
            path=self.path,
            line=prompt_token.line,
            column=prompt_token.column,
            name=name,  # Explicit: canonical name from declaration
        )
    
    def parse_chain_declaration(self):
        """
        Parse chain declaration with full workflow support.
        
        Supports both:
        1. Block syntax with 'step' definitions:
           chain "name" {
               step "step_name" {
                   kind: prompt
                   target: "prompt_name"
                   options: { ... }
               }
           }
        
        2. Config with steps list (legacy):
           chain "name" {
               steps: ["input", "rag:x", "prompt:y"]
               input_key: "data"
           }
        
        Grammar:
            ChainDecl = "chain" , QuotedName , "{" , (StepDef | ConfigItem)* , "}" ;
            StepDef = "step" , QuotedName , "{" , StepField* , "}" ;
            StepField = "kind" ":" Value
                      | "target" ":" Value
                      | "options" ":" Value
                      | "name" ":" Value
                      | "stop_on_error" ":" Boolean
                      | "evaluation" ":" Value ;
        """
        from namel3ss.ast import Chain, ChainStep, StepEvaluationConfig, WorkflowIfBlock, WorkflowForBlock, WorkflowWhileBlock
        from namel3ss.lang.parser.errors import create_syntax_error
        
        chain_token = self.expect(TokenType.CHAIN)
        name_token = self.expect(TokenType.STRING)
        name = name_token.value
        
        self.declare_symbol(f"chain:{name}", chain_token.line)
        
        # Parse block to get steps and config
        self.expect(TokenType.LBRACE)
        self.skip_newlines()
        
        # Skip indent if present (lexer may insert it after opening brace)
        self.consume_if(TokenType.INDENT)
        
        steps = []
        config = {}
        
        # Parse chain body: steps, control flow, or config
        while not self.match(TokenType.RBRACE):
            # Skip any dedent/indent tokens between lines
            while self.consume_if(TokenType.DEDENT):
                pass
            
            # Check for closing brace after consuming dedents
            if self.match(TokenType.RBRACE):
                break
                
            # Skip indent tokens
            while self.consume_if(TokenType.INDENT):
                pass
            
            # Check again for closing brace
            if self.match(TokenType.RBRACE):
                break
            
            # Check for 'step' keyword
            if self.match(TokenType.STEP):
                step_node = self._parse_step_block()
                steps.append(step_node)
            # Check for control flow (if/for/while)
            elif self.match(TokenType.IF):
                if_block = self._parse_workflow_if()
                steps.append(if_block)
            elif self.match(TokenType.FOR):
                for_block = self._parse_workflow_for()
                steps.append(for_block)
            elif self.match(TokenType.WHILE):
                while_block = self._parse_workflow_while()
                steps.append(while_block)
            # Check if it's a key:value config
            elif self.peek(1) and self.peek(1).type == TokenType.COLON:
                key = self.expect(TokenType.IDENTIFIER).value
                self.expect(TokenType.COLON)
                value = self.parse_value()
                config[key] = value
                
                # Special handling for 'steps' list (legacy format)
                if key == "steps" and isinstance(value, list):
                    # Convert legacy step references to ChainStep objects
                    for step_ref in value:
                        if isinstance(step_ref, str):
                            # Parse step reference like "prompt:name" or "rag:name"
                            if ":" in step_ref:
                                kind, target = step_ref.split(":", 1)
                            else:
                                kind = "unknown"
                                target = step_ref
                            steps.append(ChainStep(
                                kind=kind,
                                target=target,
                                options={},
                                stop_on_error=True,
                            ))
            else:
                # Unknown syntax
                current = self.current()
                raise create_syntax_error(
                    "Expected 'step', control flow (if/for/while), or config field in chain definition",
                    path=self.path,
                    line=current.line if current else None,
                    column=current.column if current else None,
                    suggestion="Use 'step \"name\" { kind: ... }' or config like 'input_key: \"data\"'",
                )
            
            self.skip_newlines()
        
        # Skip dedent if present before closing brace
        self.consume_if(TokenType.DEDENT)
        
        self.expect(TokenType.RBRACE)
        self.skip_newlines()
        
        # Extract known chain config fields
        input_key = config.pop("input_key", "input")
        metadata = config.pop("metadata", {})
        declared_effect = config.pop("declared_effect", None)
        policy_name = config.pop("policy_name", None)
        
        # Remaining config items are metadata
        if config:
            if isinstance(metadata, dict):
                metadata.update(config)
            else:
                metadata = config
        
        return Chain(
            name=name,
            input_key=input_key,
            steps=steps,
            metadata=metadata,
            declared_effect=declared_effect,
            effects=set(),
            policy_name=policy_name,
        )
    
    def _parse_step_block(self):
        """
        Parse a step block within a chain.
        
        Grammar:
            StepBlock = "step" , QuotedName , "{" , StepField* , "}" ;
            StepField = "kind" ":" Value
                      | "target" ":" Value  
                      | "options" ":" Value
                      | "arguments" ":" Value  (alias for options)
                      | "name" ":" Value
                      | "stop_on_error" ":" Boolean
                      | "evaluation" ":" Value ;
        
        Returns ChainStep AST node.
        """
        from namel3ss.ast import ChainStep, StepEvaluationConfig
        from namel3ss.lang.parser.errors import create_syntax_error
        
        step_token = self.expect(TokenType.STEP)
        
        # Step can have optional name in quotes
        step_name = None
        if self.match(TokenType.STRING):
            step_name = self.advance().value
        
        self.expect(TokenType.LBRACE)
        self.skip_newlines()
        
        # Skip indent if present
        self.consume_if(TokenType.INDENT)
        
        # Parse step fields
        kind = None
        target = None
        options = {}
        stop_on_error = True
        evaluation_config = None
        
        while not self.match(TokenType.RBRACE):
            # Skip any dedent/indent tokens between lines
            while self.consume_if(TokenType.DEDENT):
                pass
            
            # Check for closing brace
            if self.match(TokenType.RBRACE):
                break
                
            # Skip indent tokens
            while self.consume_if(TokenType.INDENT):
                pass
            
            # Check again for closing brace
            if self.match(TokenType.RBRACE):
                break
            
            if not self.match(TokenType.IDENTIFIER):
                current = self.current()
                raise create_syntax_error(
                    "Expected field name in step block",
                    path=self.path,
                    line=current.line if current else None,
                    column=current.column if current else None,
                    suggestion="Valid fields: kind, target, options, arguments, stop_on_error, evaluation",
                )
            
            field_name = self.advance().value
            self.expect(TokenType.COLON)
            
            if field_name == "kind":
                kind_value = self.parse_value()
                if isinstance(kind_value, str):
                    kind = kind_value
                else:
                    raise create_syntax_error(
                        "Step 'kind' must be a string",
                        path=self.path,
                        line=step_token.line,
                        column=step_token.column,
                        suggestion="Valid kinds: prompt, llm, tool, python, react, rag, chain, memory_read, memory_write, knowledge_query",
                    )
            elif field_name == "target":
                target_value = self.parse_value()
                if isinstance(target_value, str):
                    target = target_value
                else:
                    raise create_syntax_error(
                        "Step 'target' must be a string",
                        path=self.path,
                        line=step_token.line,
                        column=step_token.column,
                    )
            elif field_name in ("options", "arguments"):  # 'arguments' is alias for 'options'
                opts_value = self.parse_value()
                if isinstance(opts_value, dict):
                    options.update(opts_value)
                else:
                    raise create_syntax_error(
                        f"Step '{field_name}' must be an object/dict",
                        path=self.path,
                        line=step_token.line,
                        column=step_token.column,
                    )
            elif field_name == "stop_on_error":
                stop_value = self.parse_value()
                if isinstance(stop_value, bool):
                    stop_on_error = stop_value
                else:
                    raise create_syntax_error(
                        "Step 'stop_on_error' must be a boolean",
                        path=self.path,
                        line=step_token.line,
                        column=step_token.column,
                    )
            elif field_name == "evaluation":
                eval_value = self.parse_value()
                if isinstance(eval_value, dict):
                    evaluators = eval_value.get("evaluators", [])
                    guardrail = eval_value.get("guardrail", None)
                    evaluation_config = StepEvaluationConfig(
                        evaluators=evaluators if isinstance(evaluators, list) else [],
                        guardrail=guardrail if isinstance(guardrail, str) else None,
                    )
                else:
                    raise create_syntax_error(
                        "Step 'evaluation' must be an object with optional 'evaluators' (list) and 'guardrail' (string)",
                        path=self.path,
                        line=step_token.line,
                        column=step_token.column,
                    )
            else:
                # Unknown field - add to options
                options[field_name] = self.parse_value()
            
            self.skip_newlines()
        
        # Skip dedent if present before closing brace
        self.consume_if(TokenType.DEDENT)
        
        self.expect(TokenType.RBRACE)
        self.skip_newlines()
        
        # Validate required fields
        if kind is None:
            raise create_syntax_error(
                "Step block missing required field 'kind'",
                path=self.path,
                line=step_token.line,
                column=step_token.column,
                suggestion="Add 'kind: prompt' (or llm, tool, python, rag, etc.)",
            )
        if target is None:
            raise create_syntax_error(
                "Step block missing required field 'target'",
                path=self.path,
                line=step_token.line,
                column=step_token.column,
                suggestion="Add 'target: \"resource_name\"' to specify what to invoke",
            )
        
        return ChainStep(
            kind=kind,
            target=target,
            options=options,
            name=step_name,
            stop_on_error=stop_on_error,
            evaluation=evaluation_config,
        )
    
    def _parse_workflow_if(self):
        """
        Parse if/elif/else block in a workflow.
        
        Grammar:
            IfBlock = "if" , Expression , ":" , WorkflowNode+ ,
                      ("elif" , Expression , ":" , WorkflowNode+)* ,
                      ("else" , ":" , WorkflowNode+)? ;
        
        Returns WorkflowIfBlock AST node.
        """
        from namel3ss.ast import WorkflowIfBlock
        
        self.expect(TokenType.IF)
        condition = self.parse_expression()
        self.expect(TokenType.COLON)
        self.skip_newlines()
        
        # Parse then steps (indented block)
        then_steps = self._parse_workflow_block()
        
        # Parse elif branches
        elif_branches = []
        while self.match(TokenType.IDENTIFIER) and self.current() and self.current().value == "elif":
            self.advance()
            elif_condition = self.parse_expression()
            self.expect(TokenType.COLON)
            self.skip_newlines()
            elif_steps = self._parse_workflow_block()
            elif_branches.append((elif_condition, elif_steps))
        
        # Parse else branch
        else_steps = []
        if self.match(TokenType.ELSE):
            self.advance()
            self.expect(TokenType.COLON)
            self.skip_newlines()
            else_steps = self._parse_workflow_block()
        
        return WorkflowIfBlock(
            condition=condition,
            then_steps=then_steps,
            elif_steps=elif_branches,
            else_steps=else_steps,
        )
    
    def _parse_workflow_for(self):
        """
        Parse for loop in a workflow.
        
        Grammar:
            ForBlock = "for" , Identifier , "in" , Expression , ":" , WorkflowNode+ ;
        
        Returns WorkflowForBlock AST node.
        """
        from namel3ss.ast import WorkflowForBlock
        
        self.expect(TokenType.FOR)
        loop_var = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.IN)
        
        # Determine source kind
        source_kind = "expression"
        source_name = None
        source_expression = None
        
        # Check for dataset/table/frame reference
        if self.match(TokenType.DATASET):
            self.advance()
            source_kind = "dataset"
            source_name = self.expect(TokenType.STRING).value
        elif self.match(TokenType.TABLE):
            self.advance()
            source_kind = "table"
            source_name = self.expect(TokenType.IDENTIFIER).value
        elif self.match(TokenType.FRAME):
            self.advance()
            source_kind = "frame"
            source_name = self.expect(TokenType.STRING).value
        else:
            # Generic expression
            source_expression = self.parse_expression()
        
        self.expect(TokenType.COLON)
        self.skip_newlines()
        
        body = self._parse_workflow_block()
        
        return WorkflowForBlock(
            loop_var=loop_var,
            source_kind=source_kind,
            source_name=source_name,
            source_expression=source_expression,
            body=body,
            max_iterations=None,
        )
    
    def _parse_workflow_while(self):
        """
        Parse while loop in a workflow.
        
        Grammar:
            WhileBlock = "while" , Expression , ":" , WorkflowNode+ ;
        
        Returns WorkflowWhileBlock AST node.
        """
        from namel3ss.ast import WorkflowWhileBlock
        
        self.expect(TokenType.WHILE)
        condition = self.parse_expression()
        self.expect(TokenType.COLON)
        self.skip_newlines()
        
        body = self._parse_workflow_block()
        
        return WorkflowWhileBlock(
            condition=condition,
            body=body,
            max_iterations=None,
        )
    
    def _parse_workflow_block(self):
        """
        Parse a block of workflow nodes (steps or control flow).
        
        Returns list of WorkflowNode objects.
        """
        nodes = []
        
        # Handle indentation if present
        indent_level = 0
        if self.match(TokenType.INDENT):
            indent_level += 1
            self.advance()
            self.skip_newlines()
        
        while not self.match(TokenType.DEDENT) and not self.match(TokenType.RBRACE) and not self.match(TokenType.EOF):
            if self.match(TokenType.STEP):
                nodes.append(self._parse_step_block())
            elif self.match(TokenType.IF):
                nodes.append(self._parse_workflow_if())
            elif self.match(TokenType.FOR):
                nodes.append(self._parse_workflow_for())
            elif self.match(TokenType.WHILE):
                nodes.append(self._parse_workflow_while())
            else:
                # End of block
                break
            
            self.skip_newlines()
        
        # Consume dedent if we had indent
        if indent_level > 0 and self.match(TokenType.DEDENT):
            self.advance()
        
        return nodes
    
    def parse_rag_pipeline_declaration(self):
        """
        Parse RAG pipeline declaration.
        
        Uses config filtering with RAG-specific aliases.
        Unknown fields are routed to config field.
        """
        from namel3ss.ast import RagPipelineDefinition
        
        rag_token = self.expect(TokenType.RAG_PIPELINE)
        name_token = self.expect(TokenType.STRING)
        name = name_token.value
        
        self.declare_symbol(f"rag_pipeline:{name}", rag_token.line)
        
        config = self.parse_block()
        
        return build_dataclass_with_config(
            RagPipelineDefinition,
            config=config,
            aliases=RAG_ALIASES,
            name=name,
        )
    
    def parse_index_declaration(self):
        """
        Parse index declaration.
        
        Uses config filtering for IndexDefinition.
        Unknown fields are routed to config field.
        """
        from namel3ss.ast import IndexDefinition
        
        index_token = self.expect(TokenType.INDEX)
        name_token = self.expect(TokenType.STRING)
        name = name_token.value
        
        self.declare_symbol(f"index:{name}", index_token.line)
        
        config = self.parse_block()
        
        return build_dataclass_with_config(
            IndexDefinition,
            config=config,
            name=name,
        )
    
    def parse_dataset_declaration(self):
        """Parse dataset declaration."""
        from namel3ss.ast import Dataset
        
        dataset_token = self.expect(TokenType.DATASET)
        name_token = self.expect(TokenType.STRING)
        name = name_token.value
        
        self.declare_symbol(f"dataset:{name}", dataset_token.line)
        
        # Parse 'from' clause
        source = None
        source_type = "table"  # Default source type
        if self.consume_if(TokenType.FROM):
            # Check for source type keyword (table, file, api, etc.)
            if self.current().type == TokenType.IDENTIFIER:
                source_type_token = self.advance()
                source_type = source_type_token.value
            elif self.current().type.name.lower() in ("table", "query", "file", "api"):
                source_type = self.current().type.name.lower()
                self.advance()
            else:
                source_type = "table"
            
            # Parse source name (identifier or string)
            if self.check(TokenType.STRING):
                source = self.advance().value
            elif self.check(TokenType.IDENTIFIER):
                source = self.advance().value
            else:
                source = "unknown"
        
        # Parse optional block for filters/schema
        config = {}
        if self.match(TokenType.LBRACE):
            config = self.parse_dataset_block()
        elif self.consume_if(TokenType.COLON):
            # Legacy colon syntax - parse indented block
            self.skip_newlines()
            config = self._parse_indented_config_block()
        
        return Dataset(
            name=name,
            source=source or "unknown",
            source_type=source_type,
            **config,
        )
    
    def parse_dataset_block(self) -> Dict[str, Any]:
        """Parse dataset configuration block with special handling for filter."""
        self.expect(TokenType.LBRACE)
        self.skip_newlines()
        self.consume_if(TokenType.INDENT)
        
        config = {}
        
        while not self.match(TokenType.RBRACE):
            if self.consume_if(TokenType.DEDENT):
                continue
            
            # Parse key - 'filter' is keyword but should be treated as identifier here
            key_token = self.current()
            if key_token.type == TokenType.FILTER:
                key = "filter"
                self.advance()
            else:
                key_token = self.expect(TokenType.IDENTIFIER)
                key = key_token.value
            
            self.expect(TokenType.COLON)
            value = self.parse_value()
            config[key] = value
            
            self.skip_newlines()
        
        self.consume_if(TokenType.DEDENT)
        self.expect(TokenType.RBRACE)
        self.skip_newlines()
        
        return config
    
    def _parse_indented_config_block(self) -> Dict[str, Any]:
        """Parse indented configuration block for legacy colon syntax."""
        config = {}
        
        while True:
            token = self.current()
            if not token or token.type == TokenType.EOF:
                break
                
            # Skip newlines
            if token.type == TokenType.NEWLINE:
                self.advance()
                continue
                
            # Check for indentation or end of block
            if not self.match(TokenType.INDENT, TokenType.IDENTIFIER, TokenType.FILTER):
                break
                
            # Consume indent if present
            if self.match(TokenType.INDENT):
                self.advance()
                
            # Parse key-value pairs
            if self.match(TokenType.FILTER):
                key = "filter"
                self.advance()
            elif self.match(TokenType.IDENTIFIER):
                key = self.advance().value
            else:
                break
                
            # Expect 'by:' for filter or ':' for other keys
            if key == "filter":
                if self.match(TokenType.BY):
                    self.advance()
                self.expect(TokenType.COLON)
            else:
                self.expect(TokenType.COLON)
                
            # Parse the value
            value = self.parse_value()
            config[key] = value
            
            self.skip_newlines()
            
        return config
    
    def parse_memory_declaration(self):
        """Parse memory declaration."""
        from namel3ss.ast import Memory
        
        memory_token = self.expect(TokenType.MEMORY)
        name_token = self.expect(TokenType.STRING)
        name = name_token.value
        
        self.declare_symbol(f"memory:{name}", memory_token.line)
        
        config = self.parse_block()
        
        # Memory config goes into the config dict, not as top-level kwargs
        return Memory(
            name=name,
            scope=config.get('scope', 'session'),
            kind=config.get('kind', 'list'),
            max_items=config.get('max_items'),
            config=config,  # Pass all config
        )
    
    def parse_function_declaration(self):
        """Parse function declaration."""
        from namel3ss.ast import FunctionDef, Parameter
        
        fn_token = self.expect(TokenType.FN)
        name_token = self.expect(TokenType.IDENTIFIER)
        name = name_token.value
        
        self.declare_symbol(f"fn:{name}", fn_token.line)
        
        # Parse signature
        self.expect(TokenType.LPAREN)
        params = self.parse_parameter_list()
        self.expect(TokenType.RPAREN)
        
        # Optional return type
        return_type = None
        if self.consume_if(TokenType.COLON):
            return_type = self.expect(TokenType.IDENTIFIER).value
        
        # Parse body
        self.expect(TokenType.FAT_ARROW)
        body = self.parse_expression()
        
        self.skip_newlines()
        
        # Convert params to Parameter objects if they're not already
        param_objects = []
        for param in params:
            if isinstance(param, Parameter):
                param_objects.append(param)
            elif isinstance(param, dict):
                param_objects.append(Parameter(
                    name=param.get('name', ''),
                    type_hint=param.get('type'),
                    default=param.get('default'),
                ))
            elif isinstance(param, str):
                param_objects.append(Parameter(name=param))
            else:
                param_objects.append(param)
        
        return FunctionDef(
            name=name,
            params=param_objects,
            return_type=return_type,
            body=body,
        )
    
    # Stub methods for other declarations
    def parse_tool_declaration(self):
        """Parse tool declaration."""
        from namel3ss.ast import ToolDefinition
        
        tool_token = self.expect(TokenType.TOOL)
        name = self.expect(TokenType.STRING).value
        self.declare_symbol(f"tool:{name}", tool_token.line)
        config = self.parse_block()
        
        return ToolDefinition(name=name, **config)
    
    def parse_connector_declaration(self):
        """Parse connector declaration."""
        from namel3ss.ast import Connector
        
        conn_token = self.expect(TokenType.CONNECTOR)
        name = self.expect(TokenType.STRING).value
        self.declare_symbol(f"connector:{name}", conn_token.line)
        config = self.parse_block()
        
        return Connector(name=name, **config)
    
    def parse_template_declaration(self):
        """Parse template declaration."""
        from namel3ss.ast import Template
        
        template_token = self.expect(TokenType.TEMPLATE)
        name = self.expect(TokenType.STRING).value
        self.declare_symbol(f"template:{name}", template_token.line)
        config = self.parse_block()
        
        return Template(name=name, **config)
    
    def parse_model_declaration(self):
        """Parse model declaration."""
        from namel3ss.ast import Model
        
        model_token = self.expect(TokenType.MODEL)
        name = self.expect(TokenType.STRING).value
        self.declare_symbol(f"model:{name}", model_token.line)
        config = self.parse_block()
        
        return Model(name=name, **config)
    
    def parse_training_declaration(self):
        """Parse training declaration."""
        from namel3ss.ast import TrainingJob
        
        training_token = self.expect(TokenType.TRAINING)
        name = self.expect(TokenType.STRING).value
        self.declare_symbol(f"training:{name}", training_token.line)
        config = self.parse_block()
        
        return TrainingJob(name=name, **config)
    
    def parse_policy_declaration(self):
        """Parse policy declaration."""
        try:
            from namel3ss.ast import PolicyDefinition
        except ImportError:
            PolicyDefinition = dict
        policy_token = self.expect(TokenType.POLICY)
        name = self.expect(TokenType.STRING).value
        self.declare_symbol(f"policy:{name}", policy_token.line)
        config = self.parse_block()
        if PolicyDefinition is dict:
            return {"type": "policy", "name": name, **config}
        return PolicyDefinition(name=name, **config)
    
    def parse_graph_declaration(self):
        """
        Parse graph declaration.
        
        Uses config filtering with Graph-specific aliases.
        Unknown fields are routed to config field.
        """
        from namel3ss.ast import GraphDefinition
        graph_token = self.expect(TokenType.GRAPH)
        name = self.expect(TokenType.STRING).value
        self.declare_symbol(f"graph:{name}", graph_token.line)
        config = self.parse_block()
        
        return build_dataclass_with_config(
            GraphDefinition,
            config=config,
            aliases=GRAPH_ALIASES,
            name=name,
        )
    
    def parse_knowledge_declaration(self):
        """Parse knowledge declaration."""
        from namel3ss.ast import KnowledgeModule
        knowledge_token = self.expect(TokenType.KNOWLEDGE)
        name = self.expect(TokenType.STRING).value
        self.declare_symbol(f"knowledge:{name}", knowledge_token.line)
        config = self.parse_block()
        return KnowledgeModule(name=name, **config)
    
    def parse_query_declaration(self):
        """Parse query declaration."""
        from namel3ss.ast import LogicQuery
        query_token = self.expect(TokenType.QUERY)
        name = self.expect(TokenType.STRING).value
        self.declare_symbol(f"query:{name}", query_token.line)
        config = self.parse_block()
        return LogicQuery(name=name, **config)
    
    def parse_frame_declaration(self):
        """Parse frame declaration."""
        from namel3ss.ast import Frame
        frame_token = self.expect(TokenType.FRAME)
        name = self.expect(TokenType.STRING).value
        self.declare_symbol(f"frame:{name}", frame_token.line)
        config = self.parse_block()
        return Frame(name=name, **config)
    
    def parse_theme_declaration(self):
        """Parse theme declaration."""
        from namel3ss.ast import Theme
        theme_token = self.expect(TokenType.THEME)
        name = self.expect(TokenType.STRING).value
        self.declare_symbol(f"theme:{name}", theme_token.line)
        config = self.parse_block()
        return Theme(name=name, **config)


__all__ = ["DeclarationParsingMixin"]
