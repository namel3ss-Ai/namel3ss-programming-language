"""
Enhanced completion provider for N3 with improved suggestions.

Provides context-aware completions including:
- Keywords with usage examples
- Symbol completions with type information  
- Error-aware suggestions for common mistakes
- Legacy syntax migration hints
"""

from typing import List, Optional, Set
from dataclasses import dataclass

from lsprotocol.types import (
    CompletionItem, CompletionItemKind, CompletionList,
    MarkupContent, MarkupKind, Position, Range,
    TextEdit, InsertTextFormat
)

from namel3ss.parser import Parser


@dataclass
class SmartCompletion:
    """Enhanced completion item with context awareness."""
    
    label: str
    insert_text: str
    kind: CompletionItemKind
    detail: Optional[str] = None
    documentation: Optional[str] = None
    sort_text: Optional[str] = None
    filter_text: Optional[str] = None
    additional_text_edits: Optional[List[TextEdit]] = None


class EnhancedCompletionProvider:
    """Provides intelligent completions with enhanced context awareness."""
    
    # Keywords with their usage patterns
    KEYWORD_TEMPLATES = {
        "app": {
            "template": 'app "{name}" {\n    description: "{description}"\n}',
            "detail": "Application definition",
            "doc": "Define a new N3 application with metadata and configuration"
        },
        "page": {
            "template": 'page "{name}" at "{route}" {\n    show text: "Hello, World!"\n}',
            "detail": "Page declaration",
            "doc": "Create a new page with a route and components"
        },
        "llm": {
            "template": 'llm "{name}" {\n    provider: "openai"\n    model: "gpt-4o-mini"\n    temperature: 0.7\n}',
            "detail": "LLM model definition",
            "doc": "Configure a language model for use in prompts and chains"
        },
        "prompt": {
            "template": 'prompt "{name}" {\n    model: "{llm_name}"\n    template: "You are a helpful assistant. {input}"\n}',
            "detail": "Prompt template",
            "doc": "Define a reusable prompt template with variables"
        },
        "memory": {
            "template": 'memory "{name}" {\n    scope: "user"\n    kind: "list"\n    max_items: 50\n}',
            "detail": "Memory configuration",
            "doc": "Define persistent memory storage for conversations"
        },
        "frame": {
            "template": 'frame "{name}" {\n    source_type: "csv"\n    source: "data.csv"\n    columns: id, name, status\n}',
            "detail": "Data frame definition", 
            "doc": "Define a data frame for structured data processing"
        },
        "dataset": {
            "template": 'dataset "{name}" {\n    source_type: "file"\n    source: "data.json"\n    schema: [\n        {name: "id", type: "int"},\n        {name: "value", type: "string"}\n    ]\n}',
            "detail": "Dataset definition",
            "doc": "Define a dataset with schema validation"
        }
    }
    
    # Common patterns and their corrections
    SYNTAX_FIXES = {
        "show text": {
            "pattern": r'show\s+text\s+"([^"]*)"',
            "fix": 'show text: "${1}"',
            "doc": "Use colon syntax: show text: \"message\""
        },
        "show form": {
            "pattern": r'show\s+form\s+"([^"]*)"',
            "fix": 'show form: {\n    field: {\n        name: "${1}"\n        type: "text"\n    }\n    submit: "Submit"\n}',
            "doc": "Use object syntax for forms with field definitions"
        }
    }
    
    def __init__(self):
        self.cached_suggestions: Optional[List[SmartCompletion]] = None
    
    def get_completions(self, 
                       document_text: str, 
                       position: Position,
                       context_prefix: str = "",
                       current_word: str = "") -> CompletionList:
        """Get context-aware completions for the current position."""
        
        items: List[CompletionItem] = []
        
        # Add keyword completions
        items.extend(self._get_keyword_completions(current_word, context_prefix))
        
        # Add syntax fix suggestions
        items.extend(self._get_syntax_fix_completions(context_prefix))
        
        # Add smart snippets based on context
        items.extend(self._get_context_snippets(context_prefix))
        
        return CompletionList(is_incomplete=False, items=items)
    
    def _get_keyword_completions(self, prefix: str, context: str) -> List[CompletionItem]:
        """Generate keyword completions with templates."""
        items = []
        
        for keyword, info in self.KEYWORD_TEMPLATES.items():
            if not keyword.startswith(prefix.lower()):
                continue
                
            # Create snippet completion
            completion = CompletionItem(
                label=keyword,
                kind=CompletionItemKind.Keyword,
                detail=info["detail"],
                documentation=MarkupContent(
                    kind=MarkupKind.Markdown,
                    value=f"**{keyword}** - {info['doc']}\n\n```n3\n{info['template']}\n```"
                ),
                insert_text=info["template"],
                insert_text_format=InsertTextFormat.Snippet,
                sort_text=f"00_{keyword}"  # Prioritize keywords
            )
            items.append(completion)
        
        return items
    
    def _get_syntax_fix_completions(self, context: str) -> List[CompletionItem]:
        """Suggest fixes for common syntax mistakes."""
        items = []
        
        # Check for legacy syntax patterns
        if "show text " in context and ":" not in context.split("show text")[-1]:
            items.append(CompletionItem(
                label="show text: (fix syntax)",
                kind=CompletionItemKind.Text,
                detail="Convert to modern syntax",
                documentation=MarkupContent(
                    kind=MarkupKind.Markdown,
                    value="**Syntax Fix**: Use `show text: \"message\"` instead of `show text \"message\"`"
                ),
                insert_text="text: \"${1:message}\"",
                insert_text_format=InsertTextFormat.Snippet,
                sort_text="99_fix"
            ))
        
        return items
    
    def _get_context_snippets(self, context: str) -> List[CompletionItem]:
        """Generate context-aware snippets."""
        items = []
        
        # Inside page blocks
        if "page " in context and "{" in context:
            items.append(CompletionItem(
                label="show text",
                kind=CompletionItemKind.Snippet,
                detail="Display text content",
                insert_text="show text: \"${1:Hello, World!}\"",
                insert_text_format=InsertTextFormat.Snippet,
                sort_text="10_page_content"
            ))
            
            items.append(CompletionItem(
                label="show form",
                kind=CompletionItemKind.Snippet,
                detail="Create form with fields",
                insert_text="""show form: {
    field: {
        name: "${1:field_name}"
        type: "${2:text}"
        required: ${3:true}
    }
    submit: "${4:Submit}"
}""",
                insert_text_format=InsertTextFormat.Snippet,
                sort_text="10_page_form"
            ))
        
        # Inside app blocks  
        if "app " in context and "{" in context:
            items.append(CompletionItem(
                label="description",
                kind=CompletionItemKind.Property,
                detail="App description",
                insert_text="description: \"${1:Application description}\"",
                insert_text_format=InsertTextFormat.Snippet,
                sort_text="10_app_desc"
            ))
        
        return items


def enhance_workspace_completions(workspace_index) -> None:
    """Enhance workspace with improved completion provider."""
    enhanced_provider = EnhancedCompletionProvider()
    
    # Store original completion method
    original_completion = workspace_index.completion
    
    def enhanced_completion(params):
        """Enhanced completion with better suggestions."""
        document = workspace_index.document(params.text_document.uri)
        if document is None:
            return CompletionList(is_incomplete=False, items=[])
        
        # Get current context
        position = params.position
        if position.line < len(document.lines):
            line = document.lines[position.line]
            context_prefix = line[:position.character]
            current_word = document.word_at(position)
        else:
            context_prefix = ""
            current_word = ""
        
        # Get enhanced completions
        enhanced = enhanced_provider.get_completions(
            document.text, position, context_prefix, current_word
        )
        
        # Get original completions
        original = original_completion(params)
        
        # Merge and deduplicate
        all_items = enhanced.items + original.items
        seen_labels = set()
        unique_items = []
        
        for item in all_items:
            if item.label not in seen_labels:
                seen_labels.add(item.label)
                unique_items.append(item)
        
        return CompletionList(is_incomplete=False, items=unique_items)
    
    # Replace completion method
    workspace_index.completion = enhanced_completion