"""
AI Development Assistant for N3 Language

Main interface that combines all AI features:
- Code generation and completion
- Intelligent refactoring suggestions  
- Documentation assistance
- Code explanation and learning
- Best practices guidance
"""
import asyncio
import logging
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
from ..parser.parser import N3Parser
from ..lsp.symbol_navigation import SymbolNavigationEngine
from .providers import AIProvider, create_provider, detect_available_providers
from .completion_engine import CompletionEngine, CompletionContext, SmartCompletion
from .generation_engine import CodeGenerationEngine, GeneratedComponent

logger = logging.getLogger(__name__)

@dataclass
class AssistantRequest:
    """Request to the development assistant"""
    type: str  # 'generate', 'complete', 'explain', 'refactor', 'help', 'document'
    content: str
    context: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None

@dataclass
class AssistantResponse:
    """Response from the development assistant"""
    type: str
    content: str
    suggestions: List[str] = None
    code: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []
        if self.metadata is None:
            self.metadata = {}

class DevelopmentAssistant:
    """AI-powered development assistant for N3"""
    
    def __init__(self, parser: N3Parser, symbol_navigator: SymbolNavigationEngine):
        self.parser = parser
        self.symbol_navigator = symbol_navigator
        self.ai_providers: List[AIProvider] = []
        self.completion_engine: Optional[CompletionEngine] = None
        self.generation_engine: Optional[CodeGenerationEngine] = None
        self.conversation_history: List[Dict[str, Any]] = []
        self.user_preferences: Dict[str, Any] = {
            'code_style': 'modern',
            'verbosity': 'balanced',
            'preferred_provider': 'auto',
            'include_examples': True,
            'include_explanations': True
        }
    
    async def initialize(self):
        """Initialize the AI assistant with available providers"""
        try:
            # Detect and initialize AI providers
            available_providers = await detect_available_providers()
            logger.info(f"Available AI providers: {available_providers}")
            
            for provider_name in available_providers:
                try:
                    provider = create_provider(provider_name)
                    self.ai_providers.append(provider)
                    logger.info(f"Initialized AI provider: {provider_name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize {provider_name}: {e}")
            
            if not self.ai_providers:
                logger.warning("No AI providers available - assistant will work in limited mode")
                return
            
            # Initialize engines
            self.completion_engine = CompletionEngine(self.parser, self.symbol_navigator)
            await self.completion_engine.initialize_ai_providers()
            
            self.generation_engine = CodeGenerationEngine(self.parser, self.ai_providers)
            
            logger.info("AI Development Assistant initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI assistant: {e}")
            raise
    
    async def process_request(self, request: AssistantRequest) -> AssistantResponse:
        """Process a request and return appropriate response"""
        
        # Add to conversation history
        self.conversation_history.append({
            'type': 'request',
            'content': request.content,
            'request_type': request.type,
            'timestamp': asyncio.get_event_loop().time()
        })
        
        try:
            if request.type == 'generate':
                response = await self._handle_generation(request)
            elif request.type == 'complete':
                response = await self._handle_completion(request)
            elif request.type == 'explain':
                response = await self._handle_explanation(request)
            elif request.type == 'refactor':
                response = await self._handle_refactoring(request)
            elif request.type == 'help':
                response = await self._handle_help(request)
            elif request.type == 'document':
                response = await self._handle_documentation(request)
            elif request.type == 'chat':
                response = await self._handle_chat(request)
            else:
                response = AssistantResponse(
                    type='error',
                    content=f"Unknown request type: {request.type}",
                    confidence=0.0
                )
            
            # Add to conversation history
            self.conversation_history.append({
                'type': 'response',
                'content': response.content,
                'response_type': response.type,
                'timestamp': asyncio.get_event_loop().time()
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to process request: {e}")
            return AssistantResponse(
                type='error',
                content=f"Error processing request: {str(e)}",
                confidence=0.0
            )
    
    async def _handle_generation(self, request: AssistantRequest) -> AssistantResponse:
        """Handle code generation requests"""
        if not self.generation_engine:
            return AssistantResponse(
                type='error',
                content="Code generation not available - no AI providers configured",
                confidence=0.0
            )
        
        try:
            # Extract component type and context from request
            component_type = request.options.get('type') if request.options else None
            context = request.context or {}
            
            # Generate component
            component = await self.generation_engine.generate_component(
                request.content, component_type, context
            )
            
            # Build response with generated code and suggestions
            suggestions = [
                f"Generated {component.type}: {component.name}",
                f"Dependencies: {', '.join(component.dependencies) if component.dependencies else 'None'}",
                *component.suggestions
            ]
            
            return AssistantResponse(
                type='generate',
                content=f"Generated {component.type} component '{component.name}'",
                code=component.code,
                suggestions=suggestions,
                confidence=component.metadata.get('confidence', 0.8),
                metadata=component.metadata
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return AssistantResponse(
                type='error',
                content=f"Failed to generate code: {str(e)}",
                confidence=0.0
            )
    
    async def _handle_completion(self, request: AssistantRequest) -> AssistantResponse:
        """Handle code completion requests"""
        if not self.completion_engine:
            return AssistantResponse(
                type='error',
                content="Code completion not available",
                confidence=0.0
            )
        
        try:
            # Extract completion context
            file_content = request.options.get('file_content', '') if request.options else ''
            cursor_position = request.options.get('cursor_position', 0) if request.options else 0
            file_path = request.options.get('file_path', '') if request.options else ''
            
            # Build completion context
            context = self.completion_engine.get_completion_context(
                file_content, cursor_position, file_path
            )
            
            # Get completions
            completions = await self.completion_engine.get_completions(context)
            
            # Format response
            suggestions = []
            code_suggestions = []
            
            for completion in completions[:5]:  # Top 5 suggestions
                suggestions.append(f"{completion.label} - {completion.detail}")
                code_suggestions.append(completion.insert_text)
            
            return AssistantResponse(
                type='complete',
                content=f"Found {len(completions)} completion suggestions",
                suggestions=suggestions,
                code='\n'.join(code_suggestions) if code_suggestions else None,
                confidence=max(c.confidence for c in completions) if completions else 0.0,
                metadata={'completions': [c.__dict__ for c in completions]}
            )
            
        except Exception as e:
            logger.error(f"Completion failed: {e}")
            return AssistantResponse(
                type='error',
                content=f"Failed to get completions: {str(e)}",
                confidence=0.0
            )
    
    async def _handle_explanation(self, request: AssistantRequest) -> AssistantResponse:
        """Handle code explanation requests"""
        if not self.ai_providers:
            return AssistantResponse(
                type='error',
                content="Code explanation not available - no AI providers configured",
                confidence=0.0
            )
        
        try:
            provider = self.ai_providers[0]
            explanation = await provider.explain_code(request.content)
            
            # Parse code and provide additional insights
            suggestions = []
            try:
                ast = self.parser.parse(request.content, filename="<explanation>")
                if ast:
                    suggestions.append(f"Code parsed successfully - {len(ast.children)} top-level elements")
                    
                    # Analyze structure
                    for child in ast.children:
                        if hasattr(child, 'name'):
                            suggestions.append(f"Found {child.type}: {child.name}")
                            
            except Exception:
                suggestions.append("Code has syntax issues - explanation based on partial analysis")
            
            return AssistantResponse(
                type='explain',
                content=explanation,
                suggestions=suggestions,
                confidence=0.8,
                metadata={'analyzed_code': len(request.content.split('\n'))}
            )
            
        except Exception as e:
            logger.error(f"Explanation failed: {e}")
            return AssistantResponse(
                type='error',
                content=f"Failed to explain code: {str(e)}",
                confidence=0.0
            )
    
    async def _handle_refactoring(self, request: AssistantRequest) -> AssistantResponse:
        """Handle refactoring suggestions"""
        if not self.ai_providers:
            return AssistantResponse(
                type='error',
                content="Refactoring suggestions not available - no AI providers configured",
                confidence=0.0
            )
        
        try:
            provider = self.ai_providers[0]
            refactoring_suggestions = await provider.suggest_refactoring(request.content)
            
            # Analyze code structure for additional suggestions
            additional_suggestions = []
            try:
                # Check for common refactoring opportunities
                lines = request.content.split('\n')
                
                # Look for repeated patterns
                if len(lines) > 50:
                    additional_suggestions.append("Consider breaking down into smaller components")
                
                # Check for inline styles
                if 'style=' in request.content:
                    additional_suggestions.append("Consider extracting inline styles to style blocks")
                
                # Check for hardcoded values
                if any(c.isdigit() for c in request.content if c not in ['0', '1']):
                    additional_suggestions.append("Consider extracting magic numbers into constants")
                
            except Exception:
                pass
            
            all_suggestions = refactoring_suggestions + additional_suggestions
            
            return AssistantResponse(
                type='refactor',
                content="Refactoring analysis complete",
                suggestions=all_suggestions,
                confidence=0.7,
                metadata={'suggestion_count': len(all_suggestions)}
            )
            
        except Exception as e:
            logger.error(f"Refactoring analysis failed: {e}")
            return AssistantResponse(
                type='error',
                content=f"Failed to analyze for refactoring: {str(e)}",
                confidence=0.0
            )
    
    async def _handle_help(self, request: AssistantRequest) -> AssistantResponse:
        """Handle help and guidance requests"""
        
        help_topics = {
            'syntax': self._get_syntax_help(),
            'patterns': self._get_patterns_help(),
            'best practices': self._get_best_practices_help(),
            'examples': self._get_examples_help(),
            'debugging': self._get_debugging_help()
        }
        
        # Find relevant help topic
        query = request.content.lower()
        relevant_topics = []
        
        for topic, content in help_topics.items():
            if topic in query or any(word in query for word in topic.split()):
                relevant_topics.append((topic, content))
        
        if not relevant_topics:
            # General help
            content = """N3 Development Assistant Help

Available commands:
• generate <description> - Generate components from natural language
• explain <code> - Get detailed explanations of N3 code  
• complete - Get intelligent code completions
• refactor <code> - Get refactoring suggestions
• help <topic> - Get help on specific topics

Topics: syntax, patterns, best practices, examples, debugging

Example: "help syntax" or "generate user dashboard page"
"""
        else:
            content = f"Help for: {', '.join([t[0] for t in relevant_topics])}\n\n"
            content += '\n\n'.join([t[1] for t in relevant_topics])
        
        return AssistantResponse(
            type='help',
            content=content,
            suggestions=list(help_topics.keys()),
            confidence=1.0
        )
    
    async def _handle_documentation(self, request: AssistantRequest) -> AssistantResponse:
        """Handle documentation generation"""
        if not self.ai_providers:
            return AssistantResponse(
                type='error', 
                content="Documentation generation not available - no AI providers configured",
                confidence=0.0
            )
        
        try:
            provider = self.ai_providers[0]
            
            from .providers import GenerationRequest
            doc_request = GenerationRequest(
                prompt=f"""Generate comprehensive documentation for this N3 code:

{request.content}

Include:
1. Overview of what the code does
2. Key components and their purposes  
3. Usage examples
4. API/props documentation if applicable
5. Implementation notes
6. Best practices followed

Format as clean Markdown documentation.""",
                context={'code': request.content, 'type': 'documentation'},
                max_tokens=1200,
                temperature=0.2
            )
            
            response = await provider.generate_code(doc_request)
            
            return AssistantResponse(
                type='document',
                content=response.generated_code,
                confidence=response.confidence,
                suggestions=["Documentation generated successfully"],
                metadata={'reasoning': response.reasoning}
            )
            
        except Exception as e:
            logger.error(f"Documentation generation failed: {e}")
            return AssistantResponse(
                type='error',
                content=f"Failed to generate documentation: {str(e)}",
                confidence=0.0
            )
    
    async def _handle_chat(self, request: AssistantRequest) -> AssistantResponse:
        """Handle conversational chat requests"""
        if not self.ai_providers:
            return AssistantResponse(
                type='chat',
                content="I'm here to help with N3 development! Try asking me to generate code, explain syntax, or provide guidance.",
                confidence=1.0
            )
        
        try:
            provider = self.ai_providers[0]
            
            # Build conversational context
            recent_history = self.conversation_history[-6:] if len(self.conversation_history) > 6 else self.conversation_history
            
            context_str = ""
            for entry in recent_history:
                role = "User" if entry['type'] == 'request' else "Assistant"
                context_str += f"{role}: {entry['content']}\n"
            
            from .providers import GenerationRequest
            chat_request = GenerationRequest(
                prompt=f"""You are an expert N3 programming language assistant. Help with development questions, provide guidance, and have natural conversations about N3 coding.

Previous conversation:
{context_str}

User: {request.content}

Provide helpful, conversational assistance focused on N3 development. Be friendly and informative.""",
                context={'conversation_history': recent_history},
                max_tokens=800,
                temperature=0.4
            )
            
            response = await provider.generate_code(chat_request)
            
            return AssistantResponse(
                type='chat',
                content=response.generated_code,
                confidence=response.confidence,
                suggestions=["Feel free to ask follow-up questions!"],
                metadata={'reasoning': response.reasoning}
            )
            
        except Exception as e:
            logger.error(f"Chat handling failed: {e}")
            return AssistantResponse(
                type='chat',
                content="I'm having trouble with my AI capabilities right now. How can I help with N3 development?",
                confidence=0.5
            )
    
    def _get_syntax_help(self) -> str:
        return """N3 Language Syntax Reference

Basic Structure:
• Pages: page Home at "/" { <h1>Welcome</h1> }
• Frames: frame User { id: int, name: string }  
• Components: component Button(text: string) { <button>{text}</button> }
• Apps: app MyApp { pages: [Home, About] }

Data Types:
• string, int, float, bool, any
• list: string[], int[]
• Optional: string?, int?

State Management:
• state count = 0
• state items = []
• state user = null

Control Flow:
• if condition { ... } else { ... }
• for item in items { ... }
• when user { ... }

Event Handling:
• onclick={handleClick}
• onchange={updateValue}

Styling:
• style { .class { property: value } }"""
    
    def _get_patterns_help(self) -> str:
        return """N3 Common Patterns

API Integration:
async function loadData() {
    const response = await fetch('/api/data')
    return await response.json()
}

Form Handling:
state formData = {}
function updateField(field, value) {
    formData = { ...formData, [field]: value }
}

List Management:
state items = []
function addItem(item) {
    items = [...items, item]
}

Component Composition:
component Layout(children) {
    <div class="layout">
        <Header />
        {children}
        <Footer />
    </div>
}"""
    
    def _get_best_practices_help(self) -> str:
        return """N3 Best Practices

Code Organization:
• One component per file
• Group related functionality
• Use descriptive names

State Management:
• Keep state minimal and focused
• Use local state when possible
• Group related state together

Performance:
• Minimize re-renders with careful state updates
• Use async/await for API calls
• Cache computed values

Styling:
• Use CSS-in-N3 style blocks
• Follow consistent naming conventions
• Organize styles logically

Error Handling:
• Wrap async operations in try/catch
• Provide user feedback for errors
• Validate inputs early"""
    
    def _get_examples_help(self) -> str:
        return """N3 Code Examples

Todo App:
```n3
page TodoApp at "/todos" {
    state todos = []
    state newTodo = ""
    
    function addTodo() {
        if (newTodo.trim()) {
            todos = [...todos, { id: Date.now(), text: newTodo, done: false }]
            newTodo = ""
        }
    }
    
    <div class="todo-app">
        <h1>My Todos</h1>
        <input value={newTodo} onchange={e => newTodo = e.target.value} />
        <button onclick={addTodo}>Add</button>
        <ul>
            {for todo in todos {
                <li class={todo.done ? "done" : ""}>
                    {todo.text}
                </li>
            }}
        </ul>
    </div>
}
```"""
    
    def _get_debugging_help(self) -> str:
        return """N3 Debugging Guide

Common Issues:
• Missing brackets or semicolons
• Incorrect state updates
• Async/await usage errors
• CSS selector conflicts

Debugging Tools:
• console.log() for state inspection
• Browser DevTools for DOM inspection
• N3 LSP for syntax errors
• AI Assistant for code analysis

Performance Issues:
• Large lists without keys
• Frequent state updates
• Heavy computations in render
• Unoptimized API calls

Solutions:
• Use debugging statements strategically
• Break complex components into smaller ones
• Validate data flow and state changes
• Test with different data sets"""
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history for context"""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
    
    def set_user_preferences(self, preferences: Dict[str, Any]):
        """Update user preferences"""
        self.user_preferences.update(preferences)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get assistant capabilities"""
        return {
            'ai_providers': len(self.ai_providers),
            'available_providers': [p.__class__.__name__ for p in self.ai_providers],
            'features': {
                'code_generation': bool(self.generation_engine),
                'code_completion': bool(self.completion_engine),
                'code_explanation': len(self.ai_providers) > 0,
                'refactoring_suggestions': len(self.ai_providers) > 0,
                'documentation_generation': len(self.ai_providers) > 0,
                'conversational_chat': len(self.ai_providers) > 0
            },
            'conversation_length': len(self.conversation_history),
            'preferences': self.user_preferences
        }