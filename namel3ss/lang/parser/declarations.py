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
        
        # Parse optional block (will contain config like description, version, etc.)
        config = {}
        if self.match(TokenType.LBRACE):
            config = self.parse_block()
        
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
        
        self.expect(TokenType.AT)
        route_token = self.expect(TokenType.STRING)
        route = route_token.value
        
        # Parse page block
        self.expect(TokenType.LBRACE)
        self.skip_newlines()
        self.consume_if(TokenType.INDENT)  # Skip indent after opening brace
        
        statements = []
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
            statements=statements,
        )
    
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
            config['input_fields'] = input_fields
        
        if output_fields is not None:
            config['output_fields'] = output_fields
        
        # Provide default for template if not present (backwards compatibility)
        if 'template' not in config:
            config['template'] = ""
        
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
        Parse chain declaration.
        
        Grammar:
            ChainDecl = "chain" , QuotedName , Block ;
        """
        from namel3ss.ast import Chain
        
        chain_token = self.expect(TokenType.CHAIN)
        name_token = self.expect(TokenType.STRING)
        name = name_token.value
        
        self.declare_symbol(f"chain:{name}", chain_token.line)
        
        # Parse block to get steps
        self.expect(TokenType.LBRACE)
        self.skip_newlines()
        
        steps = []
        config = {}
        
        # Look for step definitions or config
        while not self.match(TokenType.RBRACE):
            # Check if it's a key:value config or a step definition
            if self.peek(1) and self.peek(1).type == TokenType.COLON:
                # It's a config item
                key = self.expect(TokenType.IDENTIFIER).value
                self.expect(TokenType.COLON)
                config[key] = self.parse_value()
            else:
                # It's a step definition
                step = self.parse_chain_step()
                steps.append(step)
            
            self.skip_newlines()
        
        self.expect(TokenType.RBRACE)
        self.skip_newlines()
        
        return Chain(
            name=name,
            steps=steps,
            **config,
        )
    
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
            config = self.parse_indented_config()
        
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
                    type_annotation=param.get('type'),
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
