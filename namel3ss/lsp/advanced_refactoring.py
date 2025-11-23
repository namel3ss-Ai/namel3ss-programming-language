"""
Advanced Refactoring Engine for N3 Language.

Provides sophisticated code transformations including:
- AST-aware refactoring operations
- Multi-file dependency tracking
- Safe rename operations
- Extract method/component refactoring
- Legacy codebase modernization

This engine uses the N3 parser to understand code structure and
perform safe, semantically-aware transformations.
"""

from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import re

from namel3ss.parser import Parser
from namel3ss.ast import Module, App, Page, Frame, Dataset
from lsprotocol.types import (
    WorkspaceEdit, TextDocumentEdit, TextEdit, Range, Position,
    OptionalVersionedTextDocumentIdentifier
)


@dataclass
class RefactoringContext:
    """Context information for refactoring operations."""
    
    source_file: str
    target_range: Range
    module: Optional[Module] = None
    symbol_name: Optional[str] = None
    dependencies: List[str] = None


@dataclass
class RefactoringResult:
    """Result of a refactoring operation."""
    
    success: bool
    workspace_edit: Optional[WorkspaceEdit] = None
    error_message: Optional[str] = None
    affected_files: List[str] = None


class AdvancedRefactoringEngine:
    """Provides sophisticated refactoring operations using AST analysis."""
    
    def __init__(self):
        self.symbol_references: Dict[str, List[Tuple[str, Range]]] = {}
        self.module_cache: Dict[str, Module] = {}
    
    def safe_rename_symbol(self, 
                          context: RefactoringContext,
                          old_name: str, 
                          new_name: str,
                          workspace_files: Dict[str, str]) -> RefactoringResult:
        """Safely rename a symbol across all references."""
        
        try:
            # Find all references to the symbol
            references = self._find_symbol_references(old_name, workspace_files)
            
            if not references:
                return RefactoringResult(
                    success=False,
                    error_message=f"Symbol '{old_name}' not found in workspace"
                )
            
            # Validate new name
            if not self._is_valid_identifier(new_name):
                return RefactoringResult(
                    success=False,
                    error_message=f"'{new_name}' is not a valid identifier"
                )
            
            # Check for naming conflicts
            if self._has_naming_conflict(new_name, workspace_files):
                return RefactoringResult(
                    success=False,
                    error_message=f"Symbol '{new_name}' already exists"
                )
            
            # Create edits for all references
            document_edits = []
            affected_files = set()
            
            for file_uri, file_references in references.items():
                edits = []
                for ref_range in file_references:
                    edits.append(TextEdit(range=ref_range, new_text=new_name))
                
                if edits:
                    document_edits.append(TextDocumentEdit(
                        text_document=OptionalVersionedTextDocumentIdentifier(
                            uri=file_uri,
                            version=None
                        ),
                        edits=edits
                    ))
                    affected_files.add(file_uri)
            
            workspace_edit = WorkspaceEdit(document_changes=document_edits)
            
            return RefactoringResult(
                success=True,
                workspace_edit=workspace_edit,
                affected_files=list(affected_files)
            )
            
        except Exception as e:
            return RefactoringResult(
                success=False,
                error_message=f"Refactoring failed: {str(e)}"
            )
    
    def extract_component(self,
                         context: RefactoringContext,
                         component_name: str,
                         selected_content: str,
                         workspace_files: Dict[str, str]) -> RefactoringResult:
        """Extract selected content into a reusable component."""
        
        try:
            # Parse the selected content to understand its structure
            lines = selected_content.splitlines()
            
            # Detect component type based on content
            component_type = self._detect_component_type(selected_content)
            
            # Generate component definition
            component_def = self._generate_component_definition(
                component_name, component_type, selected_content
            )
            
            # Create replacement for selected content
            replacement = self._generate_component_usage(component_name, component_type)
            
            # Find appropriate location for component definition
            target_file = context.source_file
            insertion_point = self._find_component_insertion_point(
                workspace_files[target_file]
            )
            
            # Create workspace edit
            document_edits = []
            
            # Replace selected content with component usage
            document_edits.append(TextDocumentEdit(
                text_document=OptionalVersionedTextDocumentIdentifier(
                    uri=target_file,
                    version=None
                ),
                edits=[
                    TextEdit(range=context.target_range, new_text=replacement),
                    TextEdit(
                        range=Range(
                            start=Position(line=insertion_point, character=0),
                            end=Position(line=insertion_point, character=0)
                        ),
                        new_text=component_def + '\n\n'
                    )
                ]
            ))
            
            workspace_edit = WorkspaceEdit(document_changes=document_edits)
            
            return RefactoringResult(
                success=True,
                workspace_edit=workspace_edit,
                affected_files=[target_file]
            )
            
        except Exception as e:
            return RefactoringResult(
                success=False,
                error_message=f"Component extraction failed: {str(e)}"
            )
    
    def modernize_legacy_syntax(self,
                               file_content: str,
                               file_uri: str) -> RefactoringResult:
        """Modernize entire file from legacy to modern N3 syntax."""
        
        try:
            lines = file_content.splitlines()
            modernized_lines = []
            changes_made = False
            
            # Apply modernization transformations
            for line_idx, line in enumerate(lines):
                original_line = line
                modernized_line = line
                
                # Apply legacy syntax transformations
                transformations = [
                    # Fix show text syntax
                    (r'^(\s*)show\s+text\s+"([^"]*)"(\s*)$', r'\1show text: "\2"\3'),
                    
                    # Fix show form syntax
                    (r'^(\s*)show\s+form\s+"([^"]*)"(\s*)\{', r'\1show form: {\3'),
                    
                    # Fix field syntax
                    (r'^(\s*)field\s+"([^"]*)"\s+type="([^"]*)"(\s+required=(\w+))?(\s*)$', 
                     r'\1field: {\6\1    name: "\2"\6\1    type: "\3"\6\1    required: \5\6\1}\6'),
                    
                    # Fix property syntax (missing colons)
                    (r'^(\s*)(\w+)\s+"([^"]*)"(\s*)$', r'\1\2: "\3"\4'),
                    
                    # Fix frame columns syntax  
                    (r'^(\s*)columns:\s*([^{][^,\n]+(?:,\s*[^,\n]+)*)\s*$', 
                     lambda m: f'{m.group(1)}columns: {{{self._modernize_columns(m.group(2), m.group(1))}}}')
                ]
                
                for pattern, replacement in transformations:
                    if callable(replacement):
                        match = re.search(pattern, modernized_line)
                        if match:
                            modernized_line = replacement(match)
                            break
                    else:
                        new_line = re.sub(pattern, replacement, modernized_line)
                        if new_line != modernized_line:
                            modernized_line = new_line
                            break
                
                if modernized_line != original_line:
                    changes_made = True
                
                modernized_lines.append(modernized_line)
            
            if not changes_made:
                return RefactoringResult(
                    success=True,
                    workspace_edit=None,
                    affected_files=[]
                )
            
            modernized_content = '\n'.join(modernized_lines)
            
            # Create workspace edit
            text_edit = TextEdit(
                range=Range(
                    start=Position(line=0, character=0),
                    end=Position(line=len(lines), character=0)
                ),
                new_text=modernized_content + '\n' if not modernized_content.endswith('\n') else modernized_content
            )
            
            workspace_edit = WorkspaceEdit(
                document_changes=[
                    TextDocumentEdit(
                        text_document=OptionalVersionedTextDocumentIdentifier(
                            uri=file_uri,
                            version=None
                        ),
                        edits=[text_edit]
                    )
                ]
            )
            
            return RefactoringResult(
                success=True,
                workspace_edit=workspace_edit,
                affected_files=[file_uri]
            )
            
        except Exception as e:
            return RefactoringResult(
                success=False,
                error_message=f"Legacy modernization failed: {str(e)}"
            )
    
    def organize_file_structure(self,
                               file_content: str,
                               file_uri: str) -> RefactoringResult:
        """Reorganize file content for better structure and readability."""
        
        try:
            # Parse the file to understand its structure
            parser = Parser(file_content, path=file_uri)
            module = parser.parse()
            
            if not module:
                return RefactoringResult(
                    success=False,
                    error_message="Could not parse file for organization"
                )
            
            # Organize components in logical order
            organized_content = self._organize_module_content(module, file_content)
            
            if organized_content == file_content:
                return RefactoringResult(
                    success=True,
                    workspace_edit=None,
                    affected_files=[]
                )
            
            # Create workspace edit
            lines = file_content.splitlines()
            text_edit = TextEdit(
                range=Range(
                    start=Position(line=0, character=0),
                    end=Position(line=len(lines), character=0)
                ),
                new_text=organized_content
            )
            
            workspace_edit = WorkspaceEdit(
                document_changes=[
                    TextDocumentEdit(
                        text_document=OptionalVersionedTextDocumentIdentifier(
                            uri=file_uri,
                            version=None
                        ),
                        edits=[text_edit]
                    )
                ]
            )
            
            return RefactoringResult(
                success=True,
                workspace_edit=workspace_edit,
                affected_files=[file_uri]
            )
            
        except Exception as e:
            return RefactoringResult(
                success=False,
                error_message=f"File organization failed: {str(e)}"
            )
    
    # Helper methods
    
    def _find_symbol_references(self, 
                               symbol_name: str,
                               workspace_files: Dict[str, str]) -> Dict[str, List[Range]]:
        """Find all references to a symbol across workspace files."""
        
        references = {}
        
        for file_uri, content in workspace_files.items():
            file_refs = []
            lines = content.splitlines()
            
            for line_idx, line in enumerate(lines):
                # Look for symbol references (this could be more sophisticated)
                pattern = rf'\b{re.escape(symbol_name)}\b'
                for match in re.finditer(pattern, line):
                    start_char = match.start()
                    end_char = match.end()
                    
                    file_refs.append(Range(
                        start=Position(line=line_idx, character=start_char),
                        end=Position(line=line_idx, character=end_char)
                    ))
            
            if file_refs:
                references[file_uri] = file_refs
        
        return references
    
    def _is_valid_identifier(self, name: str) -> bool:
        """Check if a name is a valid N3 identifier."""
        return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name))
    
    def _has_naming_conflict(self, 
                            name: str,
                            workspace_files: Dict[str, str]) -> bool:
        """Check if a name conflicts with existing symbols."""
        
        for content in workspace_files.values():
            # Simple check - look for declarations
            patterns = [
                rf'app\s+"{name}"',
                rf'page\s+"{name}"',
                rf'frame\s+"{name}"',
                rf'dataset\s+"{name}"',
                rf'llm\s+"{name}"'
            ]
            
            for pattern in patterns:
                if re.search(pattern, content):
                    return True
        
        return False
    
    def _detect_component_type(self, content: str) -> str:
        """Detect the type of component from content."""
        
        content_lower = content.lower().strip()
        
        if content_lower.startswith('show text'):
            return 'text_component'
        elif content_lower.startswith('show form'):
            return 'form_component'
        elif content_lower.startswith('show'):
            return 'ui_component'
        else:
            return 'generic_component'
    
    def _generate_component_definition(self,
                                     name: str,
                                     component_type: str,
                                     content: str) -> str:
        """Generate a component definition from extracted content."""
        
        # This is a simplified implementation - could be much more sophisticated
        return f'''component "{name}" {{
    type: "{component_type}"
    content: {{
        {content.strip()}
    }}
}}'''
    
    def _generate_component_usage(self, name: str, component_type: str) -> str:
        """Generate usage code for a component."""
        return f'show component: "{name}"'
    
    def _find_component_insertion_point(self, file_content: str) -> int:
        """Find the best place to insert a new component definition."""
        
        lines = file_content.splitlines()
        
        # Look for end of app declaration
        for i, line in enumerate(lines):
            if line.strip() == '}' and i > 0:
                # Check if this closes an app block
                for j in range(i-1, max(-1, i-20), -1):
                    if 'app ' in lines[j]:
                        return i + 1
        
        # Default to end of file
        return len(lines)
    
    def _modernize_columns(self, columns_str: str, indent: str) -> str:
        """Convert legacy column format to modern object syntax."""
        
        columns = [col.strip() for col in columns_str.split(',')]
        modernized = []
        
        for col in columns:
            if col:
                modernized.append(f'{indent}    {col}: string')
        
        return f'\n{indent.join(modernized)}\n{indent}'
    
    def _organize_module_content(self, module: Module, original_content: str) -> str:
        """Organize module content in logical order."""
        
        # This is a simplified implementation
        # A full implementation would parse the AST and reorganize based on dependencies
        
        lines = original_content.splitlines()
        organized_lines = []
        
        # Group lines by component type
        app_lines = []
        llm_lines = []
        data_lines = []  # frames, datasets
        page_lines = []
        other_lines = []
        
        current_block = []
        current_type = None
        in_block = False
        
        for line in lines:
            stripped = line.strip()
            
            # Detect start of blocks
            block_starters = [
                ('app ', app_lines),
                ('llm ', llm_lines),
                ('frame ', data_lines),
                ('dataset ', data_lines),
                ('page ', page_lines)
            ]
            
            for starter, target_list in block_starters:
                if stripped.startswith(starter):
                    if current_block:
                        (current_type or other_lines).extend(current_block)
                        current_block = []
                    current_type = target_list
                    in_block = True
                    break
            
            current_block.append(line)
            
            # Detect end of block
            if stripped == '}' and in_block:
                if current_type is not None:
                    current_type.extend(current_block)
                    current_block = []
                    current_type = None
                in_block = False
        
        # Add any remaining lines
        if current_block:
            (current_type or other_lines).extend(current_block)
        
        # Combine in logical order: app, llms, data, pages, other
        for section in [app_lines, llm_lines, data_lines, page_lines, other_lines]:
            if section:
                organized_lines.extend(section)
                organized_lines.append('')  # Add spacing between sections
        
        # Remove trailing empty lines and join
        while organized_lines and not organized_lines[-1].strip():
            organized_lines.pop()
        
        return '\n'.join(organized_lines) + '\n' if organized_lines else original_content


def get_advanced_refactoring_actions(document_text: str,
                                   document_uri: str,
                                   selection_range: Range,
                                   workspace_files: Dict[str, str]) -> List[Dict[str, Any]]:
    """Get available advanced refactoring actions."""
    
    engine = AdvancedRefactoringEngine()
    actions = []
    
    # Add file-level refactoring actions
    actions.extend([
        {
            "title": "Modernize legacy syntax",
            "description": "Convert entire file to modern N3 syntax",
            "action": "modernize_legacy",
            "scope": "file"
        },
        {
            "title": "Organize file structure", 
            "description": "Reorganize components in logical order",
            "action": "organize_structure",
            "scope": "file"
        }
    ])
    
    # Add selection-based actions if range is not empty
    if (selection_range.start.line != selection_range.end.line or
        selection_range.start.character != selection_range.end.character):
        
        actions.extend([
            {
                "title": "Extract component",
                "description": "Extract selected content into reusable component", 
                "action": "extract_component",
                "scope": "selection"
            }
        ])
    
    return actions