"""
Smart Code Completion Engine for N3 Language

Provides context-aware, AI-powered code completions that understand:
- N3 language semantics and best practices
- Project structure and existing components  
- Intelligent variable naming and patterns
- Framework-specific completions
"""
import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import asyncio
import logging
from ..parser.parser import N3Parser
from ..lsp.symbol_navigation import SymbolNavigationEngine
from .providers import AIProvider, create_provider, detect_available_providers

logger = logging.getLogger(__name__)

@dataclass
class CompletionContext:
    """Context information for code completion"""
    current_file: str
    cursor_position: int
    line_number: int
    column_number: int
    current_line: str
    before_cursor: str
    after_cursor: str
    surrounding_context: str
    symbols_in_scope: List[str]
    project_symbols: Dict[str, Any]
    file_type: str = "n3"

@dataclass
class SmartCompletion:
    """A smart completion suggestion"""
    text: str
    label: str
    detail: str
    documentation: str
    insert_text: str
    kind: str  # 'keyword', 'variable', 'function', 'class', 'module', 'snippet'
    priority: int
    confidence: float
    ai_generated: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class CompletionEngine:
    """AI-powered code completion engine"""
    
    def __init__(self, parser: N3Parser, symbol_navigator: SymbolNavigationEngine):
        self.parser = parser
        self.symbol_navigator = symbol_navigator
        self.ai_providers: List[AIProvider] = []
        self.completion_cache: Dict[str, List[SmartCompletion]] = {}
        self.ai_keywords = [
            'app', 'page', 'frame', 'component', 'state', 'style', 'import', 'export',
            'at', 'with', 'when', 'if', 'else', 'for', 'in', 'let', 'const', 'async',
            'await', 'try', 'catch', 'throw', 'return', 'break', 'continue',
            'string', 'int', 'float', 'bool', 'list', 'dict', 'any'
        ]
        self.ai_patterns = self._build_completion_patterns()
    
    async def initialize_ai_providers(self):
        """Initialize available AI providers"""
        available_providers = await detect_available_providers()
        logger.info(f"Available AI providers: {available_providers}")
        
        for provider_name in available_providers:
            try:
                provider = create_provider(provider_name)
                self.ai_providers.append(provider)
                logger.info(f"Initialized AI provider: {provider_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize {provider_name}: {e}")
    
    async def get_completions(self, context: CompletionContext) -> List[SmartCompletion]:
        """Get smart completions for the given context"""
        completions = []
        
        # Get basic syntax completions
        syntax_completions = self._get_syntax_completions(context)
        completions.extend(syntax_completions)
        
        # Get symbol-based completions
        symbol_completions = self._get_symbol_completions(context)
        completions.extend(symbol_completions)
        
        # Get pattern-based completions
        pattern_completions = self._get_pattern_completions(context)
        completions.extend(pattern_completions)
        
        # Get AI-powered completions (async)
        ai_completions = await self._get_ai_completions(context)
        completions.extend(ai_completions)
        
        # Sort by priority and confidence
        completions.sort(key=lambda c: (c.priority, c.confidence), reverse=True)
        
        return completions[:20]  # Limit to top 20 suggestions
    
    def _get_syntax_completions(self, context: CompletionContext) -> List[SmartCompletion]:
        """Get basic syntax-based completions"""
        completions = []
        current_word = self._get_current_word(context)
        
        # Keyword completions
        for keyword in self.ai_keywords:
            if keyword.startswith(current_word.lower()):
                completions.append(SmartCompletion(
                    text=keyword,
                    label=keyword,
                    detail=f"N3 keyword",
                    documentation=self._get_keyword_documentation(keyword),
                    insert_text=keyword,
                    kind='keyword',
                    priority=80,
                    confidence=0.9
                ))
        
        return completions
    
    def _get_symbol_completions(self, context: CompletionContext) -> List[SmartCompletion]:
        """Get symbol-based completions from project"""
        completions = []
        current_word = self._get_current_word(context)
        
        # Add symbols from current file and project
        for symbol_name, symbol_info in context.project_symbols.items():
            if symbol_name.startswith(current_word):
                completions.append(SmartCompletion(
                    text=symbol_name,
                    label=symbol_name,
                    detail=f"{symbol_info.get('kind', 'symbol')} from {symbol_info.get('file', 'unknown')}",
                    documentation=symbol_info.get('documentation', ''),
                    insert_text=symbol_name,
                    kind=symbol_info.get('kind', 'variable'),
                    priority=70,
                    confidence=0.8
                ))
        
        return completions
    
    def _get_pattern_completions(self, context: CompletionContext) -> List[SmartCompletion]:
        """Get pattern-based completions"""
        completions = []
        
        # Analyze current context for patterns
        for pattern_name, pattern_info in self.ai_patterns.items():
            if pattern_info['condition'](context):
                completion = SmartCompletion(
                    text=pattern_info['completion'],
                    label=pattern_info['label'],
                    detail=pattern_info['detail'],
                    documentation=pattern_info['documentation'],
                    insert_text=pattern_info['insert_text'],
                    kind='snippet',
                    priority=pattern_info['priority'],
                    confidence=0.85
                )
                completions.append(completion)
        
        return completions
    
    async def _get_ai_completions(self, context: CompletionContext) -> List[SmartCompletion]:
        """Get AI-powered intelligent completions"""
        completions = []
        
        if not self.ai_providers:
            return completions
        
        # Use the first available provider for completions
        provider = self.ai_providers[0]
        
        try:
            # Get AI completions
            ai_suggestions = await provider.complete_code(
                context.surrounding_context, 
                context.cursor_position
            )
            
            for i, suggestion in enumerate(ai_suggestions):
                if suggestion.strip():
                    completion = SmartCompletion(
                        text=suggestion,
                        label=f"AI: {suggestion[:30]}...",
                        detail="AI-powered suggestion",
                        documentation=f"Generated by {provider.model}",
                        insert_text=suggestion,
                        kind='text',
                        priority=60 - i * 5,  # Decrease priority for later suggestions
                        confidence=0.7,
                        ai_generated=True,
                        metadata={'provider': provider.__class__.__name__}
                    )
                    completions.append(completion)
        
        except Exception as e:
            logger.error(f"AI completion failed: {e}")
        
        return completions
    
    def _build_completion_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Build intelligent completion patterns for N3"""
        return {
            'page_definition': {
                'condition': lambda ctx: 'page' in ctx.current_line and 'at' not in ctx.current_line,
                'completion': 'page Home at "/" {\n    <h1>Welcome</h1>\n}',
                'label': 'page at route',
                'detail': 'Complete page definition with route',
                'documentation': 'Creates a page component with routing',
                'insert_text': '${1:PageName} at "${2:/route}" {\n    ${3:<h1>Content</h1>}\n}',
                'priority': 95
            },
            'frame_definition': {
                'condition': lambda ctx: 'frame' in ctx.current_line and '{' not in ctx.current_line,
                'completion': 'frame User {\n    id: int\n    name: string\n    email: string\n}',
                'label': 'frame schema',
                'detail': 'Complete frame definition with fields',
                'documentation': 'Creates a data structure frame with typed fields',
                'insert_text': '${1:FrameName} {\n    ${2:id}: ${3:int}\n    ${4:name}: ${5:string}\n}',
                'priority': 95
            },
            'app_definition': {
                'condition': lambda ctx: 'app' in ctx.current_line and '{' not in ctx.current_line,
                'completion': 'app MyApp {\n    pages: [Home, About]\n    theme: "modern"\n}',
                'label': 'app structure',
                'detail': 'Complete app definition with pages',
                'documentation': 'Creates an application with pages and configuration',
                'insert_text': '${1:AppName} {\n    pages: [${2:Home}]\n    theme: "${3:modern}"\n}',
                'priority': 95
            },
            'component_definition': {
                'condition': lambda ctx: 'component' in ctx.current_line,
                'completion': 'component Button(text: string, onClick: function) {\n    <button onclick={onClick}>{text}</button>\n}',
                'label': 'component with props',
                'detail': 'Complete component definition with properties',
                'documentation': 'Creates a reusable component with props',
                'insert_text': '${1:ComponentName}(${2:prop}: ${3:string}) {\n    ${4:<div>{prop}</div>}\n}',
                'priority': 90
            },
            'state_definition': {
                'condition': lambda ctx: 'state' in ctx.current_line,
                'completion': 'state count = 0\nstate items = []',
                'label': 'state variable',
                'detail': 'Define reactive state',
                'documentation': 'Creates reactive state that triggers re-renders',
                'insert_text': '${1:variableName} = ${2:defaultValue}',
                'priority': 85
            },
            'style_block': {
                'condition': lambda ctx: 'style' in ctx.current_line,
                'completion': 'style {\n    .container {\n        padding: 20px\n        margin: auto\n    }\n}',
                'label': 'style block',
                'detail': 'CSS-like styling block',
                'documentation': 'Creates a style block with CSS-like syntax',
                'insert_text': '{\n    .${1:className} {\n        ${2:property}: ${3:value}\n    }\n}',
                'priority': 80
            },
            'import_statement': {
                'condition': lambda ctx: ctx.current_line.strip().startswith('import'),
                'completion': 'import { Component } from "./components"',
                'label': 'import from module',
                'detail': 'Import components or functions',
                'documentation': 'Import functionality from other modules',
                'insert_text': '{ ${1:Component} } from "${2:./module}"',
                'priority': 75
            },
            'for_loop': {
                'condition': lambda ctx: 'for' in ctx.current_line and 'in' not in ctx.current_line,
                'completion': 'for item in items {\n    <div>{item.name}</div>\n}',
                'label': 'for-in loop',
                'detail': 'Iterate over collection',
                'documentation': 'Loop through items in a collection',
                'insert_text': '${1:item} in ${2:items} {\n    ${3:<div>{item}</div>}\n}',
                'priority': 85
            },
            'if_conditional': {
                'condition': lambda ctx: ctx.current_line.strip() == 'if',
                'completion': 'if condition {\n    // true case\n} else {\n    // false case\n}',
                'label': 'if-else block',
                'detail': 'Conditional logic',
                'documentation': 'Conditional execution based on boolean expression',
                'insert_text': '${1:condition} {\n    ${2:// true case}\n} else {\n    ${3:// false case}\n}',
                'priority': 85
            },
            'api_call': {
                'condition': lambda ctx: 'async' in ctx.before_cursor or 'await' in ctx.current_line,
                'completion': 'const data = await fetch("/api/users")\nconst result = await data.json()',
                'label': 'async API call',
                'detail': 'Fetch data from API',
                'documentation': 'Make asynchronous API request and parse response',
                'insert_text': 'const ${1:data} = await fetch("${2:/api/endpoint}")\nconst ${3:result} = await ${1:data}.json()',
                'priority': 80
            }
        }
    
    def _get_current_word(self, context: CompletionContext) -> str:
        """Extract the current word being typed"""
        line = context.current_line
        col = context.column_number
        
        # Find word boundaries
        start = col
        while start > 0 and line[start-1].isalnum() or line[start-1] in '_':
            start -= 1
        
        end = col
        while end < len(line) and (line[end].isalnum() or line[end] in '_'):
            end += 1
        
        return line[start:col]
    
    def _get_keyword_documentation(self, keyword: str) -> str:
        """Get documentation for N3 keywords"""
        docs = {
            'app': 'Define an application with pages and configuration',
            'page': 'Create a page component with routing',
            'frame': 'Define a data structure with typed fields',
            'component': 'Create a reusable UI component',
            'state': 'Define reactive state that triggers re-renders',
            'style': 'CSS-like styling block',
            'import': 'Import functionality from other modules',
            'export': 'Export functionality for other modules to use',
            'at': 'Specify route for pages (page Home at "/")',
            'with': 'Add additional properties or context',
            'when': 'Conditional execution based on state',
            'if': 'Conditional logic branch',
            'else': 'Alternative branch for if statements',
            'for': 'Iterate over collections',
            'in': 'Used with for loops (for item in items)',
            'let': 'Declare immutable variable',
            'const': 'Declare constant value',
            'async': 'Mark function as asynchronous',
            'await': 'Wait for promise resolution',
            'try': 'Error handling block',
            'catch': 'Handle caught errors',
            'throw': 'Throw an error',
            'return': 'Return value from function',
            'break': 'Exit from loop',
            'continue': 'Skip to next loop iteration'
        }
        return docs.get(keyword, f'N3 keyword: {keyword}')
    
    def get_completion_context(self, file_content: str, cursor_position: int, file_path: str) -> CompletionContext:
        """Build completion context from file content and cursor position"""
        lines = file_content.split('\n')
        
        # Find cursor line and column
        current_pos = 0
        line_number = 0
        column_number = 0
        
        for i, line in enumerate(lines):
            line_length = len(line) + 1  # +1 for newline
            if current_pos + line_length > cursor_position:
                line_number = i
                column_number = cursor_position - current_pos
                break
            current_pos += line_length
        
        current_line = lines[line_number] if line_number < len(lines) else ""
        
        # Get surrounding context (5 lines before and after)
        context_start = max(0, line_number - 5)
        context_end = min(len(lines), line_number + 6)
        surrounding_context = '\n'.join(lines[context_start:context_end])
        
        # Get symbols in scope
        symbols_in_scope = self._extract_local_symbols(lines[:line_number + 1])
        
        # Get project symbols
        project_symbols = {}
        if self.symbol_navigator:
            try:
                workspace_symbols = self.symbol_navigator.get_workspace_symbols("")
                for symbol in workspace_symbols:
                    project_symbols[symbol.name] = {
                        'kind': symbol.kind,
                        'file': symbol.location.uri,
                        'documentation': getattr(symbol, 'documentation', '')
                    }
            except Exception as e:
                logger.error(f"Failed to get workspace symbols: {e}")
        
        return CompletionContext(
            current_file=file_path,
            cursor_position=cursor_position,
            line_number=line_number,
            column_number=column_number,
            current_line=current_line,
            before_cursor=file_content[:cursor_position],
            after_cursor=file_content[cursor_position:],
            surrounding_context=surrounding_context,
            symbols_in_scope=symbols_in_scope,
            project_symbols=project_symbols
        )
    
    def _extract_local_symbols(self, lines: List[str]) -> List[str]:
        """Extract symbols defined in the current file scope"""
        symbols = []
        
        for line in lines:
            # Extract page names
            page_match = re.search(r'page\s+(\w+)', line)
            if page_match:
                symbols.append(page_match.group(1))
            
            # Extract frame names
            frame_match = re.search(r'frame\s+(\w+)', line)
            if frame_match:
                symbols.append(frame_match.group(1))
            
            # Extract component names
            component_match = re.search(r'component\s+(\w+)', line)
            if component_match:
                symbols.append(component_match.group(1))
            
            # Extract variable names
            var_matches = re.findall(r'(?:let|const|state)\s+(\w+)', line)
            symbols.extend(var_matches)
        
        return symbols
    
    async def get_intelligent_suggestions(self, context: CompletionContext, query: str) -> List[SmartCompletion]:
        """Get AI-powered intelligent suggestions for a specific query"""
        suggestions = []
        
        if not self.ai_providers:
            return suggestions
        
        provider = self.ai_providers[0]
        
        try:
            # Create a specialized prompt for intelligent suggestions
            prompt = f"""Given this N3 code context:

{context.surrounding_context}

Cursor position: line {context.line_number + 1}, column {context.column_number + 1}
Current line: {context.current_line}
Available symbols: {', '.join(context.symbols_in_scope)}

User query: "{query}"

Provide 3-5 intelligent N3 code suggestions that:
1. Follow N3 syntax and best practices
2. Make sense in the current context
3. Are complete and immediately usable
4. Address the user's query

Format each suggestion as:
SUGGESTION: [code]
EXPLANATION: [brief explanation]
"""
            
            from .providers import GenerationRequest
            request = GenerationRequest(
                prompt=prompt,
                context={'completion_context': context.__dict__},
                max_tokens=600,
                temperature=0.3
            )
            
            response = await provider.generate_code(request)
            
            # Parse suggestions from response
            suggestions_text = response.generated_code
            suggestion_blocks = re.split(r'SUGGESTION:', suggestions_text)[1:]
            
            for i, block in enumerate(suggestion_blocks[:5]):
                parts = block.split('EXPLANATION:')
                code = parts[0].strip()
                explanation = parts[1].strip() if len(parts) > 1 else ""
                
                if code:
                    suggestion = SmartCompletion(
                        text=code,
                        label=f"AI: {code[:30]}...",
                        detail="Intelligent AI suggestion",
                        documentation=explanation,
                        insert_text=code,
                        kind='snippet',
                        priority=90 - i * 5,
                        confidence=response.confidence,
                        ai_generated=True,
                        metadata={'provider': provider.__class__.__name__, 'reasoning': response.reasoning}
                    )
                    suggestions.append(suggestion)
        
        except Exception as e:
            logger.error(f"Intelligent suggestions failed: {e}")
        
        return suggestions