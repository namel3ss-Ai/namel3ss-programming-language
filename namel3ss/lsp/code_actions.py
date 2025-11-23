"""
Advanced Code Actions Provider for N3 Language Server.

Provides intelligent code transformations including:
- Quick fixes for common syntax errors
- Legacy-to-modern syntax migration
- Code refactoring operations
- Smart imports and dependency management
- Automated code formatting and organization

Usage:
    from namel3ss.lsp.code_actions import CodeActionsProvider
    
    provider = CodeActionsProvider()
    actions = provider.get_code_actions(document, range, diagnostics)
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import re

from lsprotocol.types import (
    CodeAction, CodeActionKind, CodeActionParams,
    TextEdit, WorkspaceEdit, TextDocumentEdit, 
    OptionalVersionedTextDocumentIdentifier,
    Range, Position, Diagnostic
)

from namel3ss.parser import Parser


@dataclass
class RefactoringRule:
    """Defines a code refactoring transformation."""
    
    name: str
    description: str
    pattern: str  # Regex pattern to match
    replacement: str  # Replacement template
    kind: CodeActionKind
    is_preferred: bool = False


class CodeActionsProvider:
    """Provides intelligent code actions for N3 files."""
    
    def __init__(self):
        self.advanced_engine = None
        
    def _get_advanced_engine(self):
        """Lazy load advanced refactoring engine."""
        if self.advanced_engine is None:
            try:
                from .advanced_refactoring import AdvancedRefactoringEngine
                self.advanced_engine = AdvancedRefactoringEngine()
            except ImportError:
                pass
        return self.advanced_engine
    
    # Quick fix rules for common errors
    QUICK_FIXES = [
        RefactoringRule(
            name="Add missing colon",
            description="Add missing colon in property declaration",
            pattern=r'^(\s*)(\w+)\s+"([^"]*)"(\s*)$',
            replacement=r'\1\2: "\3"\4',
            kind=CodeActionKind.QuickFix,
            is_preferred=True
        ),
        RefactoringRule(
            name="Fix show text syntax",
            description="Convert to modern show text syntax",
            pattern=r'^(\s*)show\s+text\s+"([^"]*)"(\s*)$',
            replacement=r'\1show text: "\2"\3',
            kind=CodeActionKind.QuickFix,
            is_preferred=True
        ),
        RefactoringRule(
            name="Fix show form syntax", 
            description="Convert to modern show form syntax",
            pattern=r'^(\s*)show\s+form\s+"([^"]*)"(\s*)\{',
            replacement=r'\1show form: {\3',
            kind=CodeActionKind.QuickFix
        ),
        RefactoringRule(
            name="Fix field syntax",
            description="Convert field to modern object syntax",
            pattern=r'^(\s*)field\s+"([^"]*)"\s+type="([^"]*)"(\s+required=(\w+))?(\s*)$',
            replacement=r'\1field: {\6\1    name: "\2"\6\1    type: "\3"\6\1    required: \5\6\1}\6',
            kind=CodeActionKind.QuickFix
        )
    ]
    
    # Legacy migration rules
    LEGACY_MIGRATIONS = [
        RefactoringRule(
            name="Modernize app declaration",
            description="Update app syntax to modern format",
            pattern=r'^(\s*)app\s+"([^"]*)"\s*(\{.*?\})',
            replacement=r'\1app "\2" \3',
            kind=CodeActionKind.RefactorRewrite
        ),
        RefactoringRule(
            name="Modernize page components",
            description="Convert legacy page components to modern syntax", 
            pattern=r'^(\s*)show\s+(\w+)\s+"([^"]*)"',
            replacement=r'\1show \2: "\3"',
            kind=CodeActionKind.RefactorRewrite
        ),
        RefactoringRule(
            name="Update frame columns syntax",
            description="Convert frame columns to modern format",
            pattern=r'^(\s*)columns:\s*([^,\n]+(?:,\s*[^,\n]+)*)\s*$',
            replacement=lambda m: f'{m.group(1)}columns: {{{m.group(1)}    {", ".join(f"{col.strip()}: string" for col in m.group(2).split(","))}{m.group(1)}}}',
            kind=CodeActionKind.RefactorRewrite
        )
    ]
    
    # Code improvement suggestions
    IMPROVEMENTS = [
        RefactoringRule(
            name="Add type annotations",
            description="Add explicit type annotations to fields",
            pattern=r'^(\s*)(\w+):\s*"([^"]*)"(\s*)$',
            replacement=r'\1\2: string = "\3"\4',
            kind=CodeActionKind.RefactorRewrite
        ),
        RefactoringRule(
            name="Extract reusable component",
            description="Extract repeated pattern into reusable component",
            pattern=r'^(\s*)show\s+text:\s*"([^"]*)"(\s*)$',
            replacement=r'\1show component: "text_display" {\1    content: "\2"\1}\3',
            kind=CodeActionKind.RefactorExtract
        )
    ]
    
    def get_code_actions(self, 
                        document_text: str,
                        document_uri: str, 
                        range_selection: Range,
                        diagnostics: List[Diagnostic]) -> List[CodeAction]:
        """Get available code actions for the given context."""
        
        actions: List[CodeAction] = []
        lines = document_text.splitlines()
        
        # Get quick fixes based on diagnostics
        actions.extend(self._get_quick_fixes(document_text, document_uri, diagnostics))
        
        # Get refactoring actions for selected range
        actions.extend(self._get_refactoring_actions(document_text, document_uri, range_selection, lines))
        
        # Get improvement suggestions
        actions.extend(self._get_improvement_actions(document_text, document_uri, range_selection, lines))
        
        # Get organize actions
        actions.extend(self._get_organize_actions(document_text, document_uri))
        
        # Get advanced refactoring actions
        actions.extend(self._get_advanced_refactoring_actions(
            document_text, document_uri, range_selection, lines
        ))
        
        return actions
    
    def _get_quick_fixes(self, 
                        document_text: str, 
                        document_uri: str,
                        diagnostics: List[Diagnostic]) -> List[CodeAction]:
        """Generate quick fix actions for diagnostics."""
        
        actions = []
        lines = document_text.splitlines()
        
        for diagnostic in diagnostics:
            line_idx = diagnostic.range.start.line
            if line_idx >= len(lines):
                continue
                
            line_text = lines[line_idx]
            
            # Try each quick fix rule
            for rule in self.QUICK_FIXES:
                match = re.search(rule.pattern, line_text)
                if match:
                    # Create text edit
                    if callable(rule.replacement):
                        new_text = rule.replacement(match)
                    else:
                        new_text = re.sub(rule.pattern, rule.replacement, line_text)
                    
                    text_edit = TextEdit(
                        range=Range(
                            start=Position(line=line_idx, character=0),
                            end=Position(line=line_idx, character=len(line_text))
                        ),
                        new_text=new_text
                    )
                    
                    workspace_edit = WorkspaceEdit(
                        document_changes=[
                            TextDocumentEdit(
                                text_document=OptionalVersionedTextDocumentIdentifier(
                                    uri=document_uri,
                                    version=None
                                ),
                                edits=[text_edit]
                            )
                        ]
                    )
                    
                    action = CodeAction(
                        title=rule.name,
                        kind=rule.kind,
                        edit=workspace_edit,
                        is_preferred=rule.is_preferred,
                        diagnostics=[diagnostic]
                    )
                    actions.append(action)
        
        return actions
    
    def _get_refactoring_actions(self,
                               document_text: str,
                               document_uri: str, 
                               range_selection: Range,
                               lines: List[str]) -> List[CodeAction]:
        """Generate refactoring actions for selected code."""
        
        actions = []
        
        # Get selected text
        start_line = range_selection.start.line
        end_line = range_selection.end.line
        
        if start_line >= len(lines) or end_line >= len(lines):
            return actions
        
        # Check for legacy patterns in selection
        for line_idx in range(start_line, end_line + 1):
            if line_idx >= len(lines):
                continue
                
            line_text = lines[line_idx]
            
            for rule in self.LEGACY_MIGRATIONS:
                match = re.search(rule.pattern, line_text)
                if match:
                    # Create modernization action
                    if callable(rule.replacement):
                        new_text = rule.replacement(match)
                    else:
                        new_text = re.sub(rule.pattern, rule.replacement, line_text)
                    
                    text_edit = TextEdit(
                        range=Range(
                            start=Position(line=line_idx, character=0),
                            end=Position(line=line_idx, character=len(line_text))
                        ),
                        new_text=new_text
                    )
                    
                    workspace_edit = WorkspaceEdit(
                        document_changes=[
                            TextDocumentEdit(
                                text_document=OptionalVersionedTextDocumentIdentifier(
                                    uri=document_uri,
                                    version=None
                                ),
                                edits=[text_edit]
                            )
                        ]
                    )
                    
                    action = CodeAction(
                        title=rule.name,
                        kind=rule.kind,
                        edit=workspace_edit,
                        data={
                            "line": line_idx,
                            "pattern": rule.pattern
                        }
                    )
                    actions.append(action)
        
        return actions
    
    def _get_improvement_actions(self,
                               document_text: str,
                               document_uri: str,
                               range_selection: Range, 
                               lines: List[str]) -> List[CodeAction]:
        """Generate code improvement suggestions."""
        
        actions = []
        
        # Look for improvement opportunities
        for line_idx, line_text in enumerate(lines):
            # Skip if not in selection range (if range is specified)
            if (range_selection.start.line != range_selection.end.line and 
                (line_idx < range_selection.start.line or line_idx > range_selection.end.line)):
                continue
            
            for rule in self.IMPROVEMENTS:
                match = re.search(rule.pattern, line_text)
                if match:
                    if callable(rule.replacement):
                        new_text = rule.replacement(match)
                    else:
                        new_text = re.sub(rule.pattern, rule.replacement, line_text)
                    
                    text_edit = TextEdit(
                        range=Range(
                            start=Position(line=line_idx, character=0),
                            end=Position(line=line_idx, character=len(line_text))
                        ),
                        new_text=new_text
                    )
                    
                    workspace_edit = WorkspaceEdit(
                        document_changes=[
                            TextDocumentEdit(
                                text_document=OptionalVersionedTextDocumentIdentifier(
                                    uri=document_uri,
                                    version=None
                                ),
                                edits=[text_edit]
                            )
                        ]
                    )
                    
                    action = CodeAction(
                        title=rule.name,
                        kind=rule.kind,
                        edit=workspace_edit
                    )
                    actions.append(action)
        
        return actions
    
    def _get_organize_actions(self,
                            document_text: str,
                            document_uri: str) -> List[CodeAction]:
        """Generate document organization actions."""
        
        actions = []
        
        # Add "Sort properties" action
        actions.append(self._create_sort_properties_action(document_text, document_uri))
        
        # Add "Format document" action
        actions.append(self._create_format_document_action(document_text, document_uri))
        
        # Add "Organize imports" action (if applicable)
        actions.append(self._create_organize_imports_action(document_text, document_uri))
        
        return [action for action in actions if action is not None]
    
    def _create_sort_properties_action(self, 
                                     document_text: str,
                                     document_uri: str) -> Optional[CodeAction]:
        """Create action to sort properties alphabetically."""
        
        lines = document_text.splitlines()
        organized_lines = []
        current_block = []
        in_block = False
        block_indent = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Detect start of block
            if stripped.endswith('{') and not in_block:
                in_block = True
                block_indent = len(line) - len(line.lstrip())
                organized_lines.append(line)
                continue
            
            # Detect end of block
            if stripped == '}' and in_block:
                # Sort the collected properties
                if current_block:
                    current_block.sort(key=lambda x: x.strip().split(':')[0].strip())
                    organized_lines.extend(current_block)
                    current_block = []
                
                organized_lines.append(line)
                in_block = False
                continue
            
            # Inside block - collect properties
            if in_block and ':' in stripped and not stripped.startswith('//'):
                current_block.append(line)
            else:
                # Sort any pending block and add non-property line
                if current_block:
                    current_block.sort(key=lambda x: x.strip().split(':')[0].strip())
                    organized_lines.extend(current_block)
                    current_block = []
                organized_lines.append(line)
        
        # Join organized lines
        organized_text = '\n'.join(organized_lines)
        
        # Only create action if there are changes
        if organized_text != document_text:
            text_edit = TextEdit(
                range=Range(
                    start=Position(line=0, character=0),
                    end=Position(line=len(lines), character=0)
                ),
                new_text=organized_text + '\n' if not organized_text.endswith('\n') else organized_text
            )
            
            workspace_edit = WorkspaceEdit(
                document_changes=[
                    TextDocumentEdit(
                        text_document=OptionalVersionedTextDocumentIdentifier(
                            uri=document_uri,
                            version=None
                        ),
                        edits=[text_edit]
                    )
                ]
            )
            
            return CodeAction(
                title="Sort properties alphabetically",
                kind=CodeActionKind.SourceOrganizeImports,
                edit=workspace_edit
            )
        
        return None
    
    def _create_format_document_action(self,
                                     document_text: str, 
                                     document_uri: str) -> Optional[CodeAction]:
        """Create action to format document with consistent indentation."""
        
        lines = document_text.splitlines()
        formatted_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('//'):
                formatted_lines.append(line)
                continue
            
            # Decrease indent for closing braces
            if stripped == '}':
                indent_level = max(0, indent_level - 1)
            
            # Apply consistent indentation (4 spaces)
            formatted_line = '    ' * indent_level + stripped
            formatted_lines.append(formatted_line)
            
            # Increase indent for opening braces
            if stripped.endswith('{'):
                indent_level += 1
        
        formatted_text = '\n'.join(formatted_lines)
        
        # Only create action if there are changes
        if formatted_text != document_text:
            text_edit = TextEdit(
                range=Range(
                    start=Position(line=0, character=0),
                    end=Position(line=len(lines), character=0)
                ),
                new_text=formatted_text + '\n' if not formatted_text.endswith('\n') else formatted_text
            )
            
            workspace_edit = WorkspaceEdit(
                document_changes=[
                    TextDocumentEdit(
                        text_document=OptionalVersionedTextDocumentIdentifier(
                            uri=document_uri,
                            version=None
                        ),
                        edits=[text_edit]
                    )
                ]
            )
            
            return CodeAction(
                title="Format document",
                kind=CodeActionKind.SourceFormatDocument,
                edit=workspace_edit
            )
        
        return None
    
    def _create_organize_imports_action(self,
                                      document_text: str,
                                      document_uri: str) -> Optional[CodeAction]:
        """Create action to organize imports (if any)."""
        
        # For now, this is a placeholder - N3 doesn't have traditional imports
        # But we could organize dependencies or references
        
        lines = document_text.splitlines()
        
        # Look for potential references or dependencies
        references = []
        other_lines = []
        
        for line in lines:
            stripped = line.strip()
            # This could be expanded based on N3's dependency system
            if any(keyword in stripped for keyword in ['connects to', 'uses', 'imports']):
                references.append(line)
            else:
                other_lines.append(line)
        
        if references:
            # Sort references and place them at the top
            references.sort()
            organized_lines = references + [''] + other_lines
            organized_text = '\n'.join(organized_lines)
            
            if organized_text != document_text:
                text_edit = TextEdit(
                    range=Range(
                        start=Position(line=0, character=0),
                        end=Position(line=len(lines), character=0)
                    ),
                    new_text=organized_text
                )
                
                workspace_edit = WorkspaceEdit(
                    document_changes=[
                        TextDocumentEdit(
                            text_document=OptionalVersionedTextDocumentIdentifier(
                                uri=document_uri,
                                version=None
                            ),
                            edits=[text_edit]
                        )
                    ]
                )
                
                return CodeAction(
                    title="Organize references",
                    kind=CodeActionKind.SourceOrganizeImports,
                    edit=workspace_edit
                )
        
        return None
    
    def _get_advanced_refactoring_actions(self,
                                        document_text: str,
                                        document_uri: str,
                                        range_selection: Range,
                                        lines: List[str]) -> List[CodeAction]:
        """Get advanced refactoring actions using the advanced engine."""
        
        actions = []
        engine = self._get_advanced_engine()
        
        if engine is None:
            return actions
        
        # Add "Modernize legacy syntax" action for entire file
        actions.append(CodeAction(
            title="Modernize legacy syntax",
            kind=CodeActionKind.RefactorRewrite,
            command={
                "title": "Modernize legacy syntax",
                "command": "namel3ss.refactor.modernizeLegacy",
                "arguments": [document_uri]
            }
        ))
        
        # Add "Organize file structure" action
        actions.append(CodeAction(
            title="Organize file structure",
            kind=CodeActionKind.SourceOrganizeImports,
            command={
                "title": "Organize file structure", 
                "command": "namel3ss.refactor.organizeStructure",
                "arguments": [document_uri]
            }
        ))
        
        # Add "Extract component" if there's a selection
        if (range_selection.start.line != range_selection.end.line or 
            range_selection.start.character != range_selection.end.character):
            
            actions.append(CodeAction(
                title="Extract component",
                kind=CodeActionKind.RefactorExtract,
                command={
                    "title": "Extract component",
                    "command": "namel3ss.refactor.extractComponent", 
                    "arguments": [document_uri, range_selection]
                }
            ))
        
        return actions


def enhance_lsp_with_code_actions(language_server):
    """Enhance LSP server with code actions support."""
    
    code_actions_provider = CodeActionsProvider()
    
    @language_server.feature("textDocument/codeAction")
    async def code_actions(params: CodeActionParams):
        """Handle code action requests."""
        
        document = language_server.workspace_index.document(params.text_document.uri)
        if document is None:
            return []
        
        return code_actions_provider.get_code_actions(
            document.text,
            params.text_document.uri,
            params.range,
            params.context.diagnostics if params.context else []
        )
    
    # Code actions provider registered - logger may not be available during init