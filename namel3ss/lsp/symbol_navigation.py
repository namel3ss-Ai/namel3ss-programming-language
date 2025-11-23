"""
Symbol Navigation and Semantic Analysis for N3 Language Server.

Provides intelligent symbol understanding including:
- Go-to-definition across files and projects
- Find-all-references with usage context
- Symbol hierarchy and inheritance tracking
- Semantic validation and type checking
- Dependency graph analysis
- Cross-file symbol resolution

This module transforms N3 from a syntax-aware language to a 
semantically intelligent development environment.
"""

from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import re

from lsprotocol.types import (
    Location, Position, Range, DocumentSymbol, SymbolInformation,
    SymbolKind, WorkspaceSymbol, DefinitionParams, ReferenceParams,
    DocumentSymbolParams, WorkspaceSymbolParams, Hover, HoverParams,
    MarkupContent, MarkupKind
)

from namel3ss.parser import Parser
from namel3ss.ast import Module, App, Page, Frame, Dataset


@dataclass
class SymbolReference:
    """Represents a reference to a symbol."""
    
    uri: str
    range: Range
    context: str  # The line of code containing the reference
    kind: str  # 'definition', 'reference', 'call', 'import'


@dataclass
class SymbolDefinition:
    """Comprehensive symbol definition with metadata."""
    
    name: str
    kind: SymbolKind
    uri: str
    range: Range
    selection_range: Range
    detail: str
    documentation: Optional[str] = None
    type_info: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    references: List[SymbolReference] = field(default_factory=list)
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)


@dataclass
class SemanticContext:
    """Semantic context for analysis and navigation."""
    
    workspace_root: str
    files: Dict[str, str]  # URI -> content
    modules: Dict[str, Module] = field(default_factory=dict)
    symbols: Dict[str, SymbolDefinition] = field(default_factory=dict)
    dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    reverse_dependencies: Dict[str, Set[str]] = field(default_factory=dict)


class SymbolNavigationEngine:
    """Provides intelligent symbol navigation and semantic analysis."""
    
    def __init__(self):
        self.context: Optional[SemanticContext] = None
        self.symbol_cache: Dict[str, Dict[str, SymbolDefinition]] = {}
        
    def initialize_workspace(self, workspace_root: str, files: Dict[str, str]) -> None:
        """Initialize semantic analysis for the entire workspace."""
        
        self.context = SemanticContext(
            workspace_root=workspace_root,
            files=files
        )
        
        # Parse all files and build symbol tables
        self._parse_workspace()
        self._build_symbol_index()
        self._analyze_dependencies()
        self._resolve_references()
        
    def go_to_definition(self, uri: str, position: Position) -> List[Location]:
        """Find the definition of symbol at the given position."""
        
        if not self.context:
            return []
            
        # Get symbol at position
        symbol_name = self._get_symbol_at_position(uri, position)
        if not symbol_name:
            return []
        
        # Find definition
        definition = self.context.symbols.get(symbol_name)
        if not definition:
            return []
        
        return [Location(uri=definition.uri, range=definition.range)]
    
    def find_references(self, uri: str, position: Position, include_declaration: bool = True) -> List[Location]:
        """Find all references to the symbol at the given position."""
        
        if not self.context:
            return []
        
        symbol_name = self._get_symbol_at_position(uri, position)
        if not symbol_name:
            return []
        
        definition = self.context.symbols.get(symbol_name)
        if not definition:
            return []
        
        locations = []
        
        # Include definition if requested
        if include_declaration:
            locations.append(Location(uri=definition.uri, range=definition.range))
        
        # Add all references
        for ref in definition.references:
            locations.append(Location(uri=ref.uri, range=ref.range))
        
        return locations
    
    def get_document_symbols(self, uri: str) -> List[DocumentSymbol]:
        """Get hierarchical symbols for a document."""
        
        if not self.context or uri not in self.context.files:
            return []
        
        # Get symbols defined in this file
        file_symbols = []
        
        for symbol in self.context.symbols.values():
            if symbol.uri == uri:
                doc_symbol = DocumentSymbol(
                    name=symbol.name,
                    kind=symbol.kind,
                    range=symbol.range,
                    selection_range=symbol.selection_range,
                    detail=symbol.detail
                )
                
                # Add children if any
                children = []
                for child_name in symbol.children:
                    child_symbol = self.context.symbols.get(child_name)
                    if child_symbol and child_symbol.uri == uri:
                        children.append(DocumentSymbol(
                            name=child_symbol.name,
                            kind=child_symbol.kind,
                            range=child_symbol.range,
                            selection_range=child_symbol.selection_range,
                            detail=child_symbol.detail
                        ))
                
                doc_symbol.children = children
                file_symbols.append(doc_symbol)
        
        return file_symbols
    
    def get_workspace_symbols(self, query: str) -> List[WorkspaceSymbol]:
        """Search for symbols across the workspace."""
        
        if not self.context:
            return []
        
        query_lower = query.lower()
        results = []
        
        for symbol in self.context.symbols.values():
            if query_lower in symbol.name.lower():
                results.append(WorkspaceSymbol(
                    name=symbol.name,
                    kind=symbol.kind,
                    location=Location(uri=symbol.uri, range=symbol.range),
                    container_name=symbol.parent
                ))
        
        # Sort by relevance (exact matches first, then starts-with, then contains)
        def relevance_score(ws: WorkspaceSymbol) -> int:
            name_lower = ws.name.lower()
            if name_lower == query_lower:
                return 0
            elif name_lower.startswith(query_lower):
                return 1
            else:
                return 2
        
        results.sort(key=relevance_score)
        return results[:50]  # Limit results
    
    def get_hover_info(self, uri: str, position: Position) -> Optional[Hover]:
        """Get hover information for symbol at position."""
        
        if not self.context:
            return None
        
        symbol_name = self._get_symbol_at_position(uri, position)
        if not symbol_name:
            return None
        
        definition = self.context.symbols.get(symbol_name)
        if not definition:
            return None
        
        # Build hover content
        content_parts = []
        
        # Symbol signature
        content_parts.append(f"**{definition.name}**")
        
        if definition.type_info:
            content_parts.append(f"*{definition.type_info}*")
        
        if definition.detail:
            content_parts.append(definition.detail)
        
        if definition.documentation:
            content_parts.append("---")
            content_parts.append(definition.documentation)
        
        # Usage statistics
        ref_count = len(definition.references)
        if ref_count > 0:
            content_parts.append("---")
            content_parts.append(f"Referenced {ref_count} time{'s' if ref_count != 1 else ''}")
        
        # Dependencies
        if definition.dependencies:
            deps = definition.dependencies[:3]  # Show first 3
            deps_str = ", ".join(deps)
            if len(definition.dependencies) > 3:
                deps_str += f", +{len(definition.dependencies) - 3} more"
            content_parts.append(f"Depends on: {deps_str}")
        
        hover_content = "\n\n".join(content_parts)
        
        return Hover(
            contents=MarkupContent(
                kind=MarkupKind.Markdown,
                value=hover_content
            ),
            range=self._get_symbol_range_at_position(uri, position)
        )
    
    def analyze_symbol_dependencies(self, symbol_name: str) -> Dict[str, Any]:
        """Analyze dependencies and dependents of a symbol."""
        
        if not self.context:
            return {}
        
        definition = self.context.symbols.get(symbol_name)
        if not definition:
            return {}
        
        # Get direct dependencies
        direct_deps = set(definition.dependencies)
        
        # Get transitive dependencies
        transitive_deps = set()
        to_visit = list(direct_deps)
        visited = set()
        
        while to_visit:
            current = to_visit.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            current_def = self.context.symbols.get(current)
            if current_def:
                for dep in current_def.dependencies:
                    if dep not in visited:
                        to_visit.append(dep)
                        transitive_deps.add(dep)
        
        # Get reverse dependencies (what depends on this symbol)
        reverse_deps = self.context.reverse_dependencies.get(symbol_name, set())
        
        return {
            "symbol": symbol_name,
            "direct_dependencies": list(direct_deps),
            "transitive_dependencies": list(transitive_deps),
            "dependents": list(reverse_deps),
            "reference_count": len(definition.references),
            "definition_location": {
                "uri": definition.uri,
                "range": definition.range
            }
        }
    
    # Private methods for implementation
    
    def _parse_workspace(self) -> None:
        """Parse all files in the workspace."""
        
        if not self.context:
            return
        
        for uri, content in self.context.files.items():
            try:
                parser = Parser(content, path=uri)
                module = parser.parse()
                self.context.modules[uri] = module
            except Exception as e:
                print(f"Failed to parse {uri}: {e}")
    
    def _build_symbol_index(self) -> None:
        """Build comprehensive symbol index from parsed modules."""
        
        if not self.context:
            return
        
        for uri, module in self.context.modules.items():
            self._extract_symbols_from_module(uri, module)
    
    def _extract_symbols_from_module(self, uri: str, module: Module) -> None:
        """Extract symbols from a parsed module."""
        
        if not self.context or not module or not hasattr(module, 'body'):
            return
        
        lines = self.context.files[uri].splitlines()
        
        for node in module.body:
            symbol_def = None
            
            if hasattr(node, 'name'):
                # Determine symbol type and create definition
                if node.__class__.__name__ == 'App':
                    symbol_def = self._create_app_symbol(uri, node, lines)
                elif node.__class__.__name__ == 'Page':
                    symbol_def = self._create_page_symbol(uri, node, lines)
                elif node.__class__.__name__ == 'Frame':
                    symbol_def = self._create_frame_symbol(uri, node, lines)
                elif node.__class__.__name__ == 'Dataset':
                    symbol_def = self._create_dataset_symbol(uri, node, lines)
                # Add more node types as needed
                
                if symbol_def:
                    self.context.symbols[symbol_def.name] = symbol_def
                    
                    # Extract child symbols (like page components)
                    if hasattr(node, 'statements') and node.statements:
                        self._extract_child_symbols(uri, node, symbol_def.name, lines)
    
    def _create_app_symbol(self, uri: str, app_node: App, lines: List[str]) -> SymbolDefinition:
        """Create symbol definition for app node."""
        
        name = getattr(app_node, 'name', 'Unknown')
        line_num = getattr(app_node, 'line_number', 1) - 1
        
        # Find the range of the app declaration
        app_range = self._find_declaration_range(lines, line_num, f'app "{name}"')
        
        return SymbolDefinition(
            name=name,
            kind=SymbolKind.Module,
            uri=uri,
            range=app_range,
            selection_range=Range(
                start=Position(line=line_num, character=0),
                end=Position(line=line_num, character=len(f'app "{name}"'))
            ),
            detail="Application",
            documentation=getattr(app_node, 'description', None),
            type_info="N3 Application"
        )
    
    def _create_page_symbol(self, uri: str, page_node: Page, lines: List[str]) -> SymbolDefinition:
        """Create symbol definition for page node."""
        
        name = getattr(page_node, 'name', 'Unknown')
        route = getattr(page_node, 'route', '')
        line_num = getattr(page_node, 'line_number', 1) - 1
        
        page_range = self._find_declaration_range(lines, line_num, f'page "{name}"')
        
        return SymbolDefinition(
            name=name,
            kind=SymbolKind.Class,
            uri=uri,
            range=page_range,
            selection_range=Range(
                start=Position(line=line_num, character=0),
                end=Position(line=line_num, character=len(f'page "{name}"'))
            ),
            detail=f"Page at {route}",
            type_info="N3 Page",
            documentation=f"Web page accessible at route: {route}"
        )
    
    def _create_frame_symbol(self, uri: str, frame_node: Frame, lines: List[str]) -> SymbolDefinition:
        """Create symbol definition for frame node."""
        
        name = getattr(frame_node, 'name', 'Unknown')
        source_type = getattr(frame_node, 'source_type', 'unknown')
        line_num = getattr(frame_node, 'line_number', 1) - 1
        
        frame_range = self._find_declaration_range(lines, line_num, f'frame "{name}"')
        
        columns = getattr(frame_node, 'columns', [])
        column_info = f"{len(columns)} columns" if columns else "No columns defined"
        
        return SymbolDefinition(
            name=name,
            kind=SymbolKind.Struct,
            uri=uri,
            range=frame_range,
            selection_range=Range(
                start=Position(line=line_num, character=0),
                end=Position(line=line_num, character=len(f'frame "{name}"'))
            ),
            detail=f"Frame from {source_type} ({column_info})",
            type_info="N3 Data Frame",
            documentation=f"Data frame loaded from {source_type} source"
        )
    
    def _create_dataset_symbol(self, uri: str, dataset_node: Dataset, lines: List[str]) -> SymbolDefinition:
        """Create symbol definition for dataset node."""
        
        name = getattr(dataset_node, 'name', 'Unknown')
        source_type = getattr(dataset_node, 'source_type', 'unknown')
        line_num = getattr(dataset_node, 'line_number', 1) - 1
        
        dataset_range = self._find_declaration_range(lines, line_num, f'dataset "{name}"')
        
        return SymbolDefinition(
            name=name,
            kind=SymbolKind.Object,
            uri=uri,
            range=dataset_range,
            selection_range=Range(
                start=Position(line=line_num, character=0),
                end=Position(line=line_num, character=len(f'dataset "{name}"'))
            ),
            detail=f"Dataset from {source_type}",
            type_info="N3 Dataset",
            documentation=f"Dataset loaded from {source_type} source"
        )
    
    def _extract_child_symbols(self, uri: str, parent_node: Any, parent_name: str, lines: List[str]) -> None:
        """Extract child symbols from parent node."""
        # This would extract components, variables, etc. from pages
        # Implementation depends on specific AST structure
        pass
    
    def _find_declaration_range(self, lines: List[str], start_line: int, pattern: str) -> Range:
        """Find the range of a declaration starting from a line."""
        
        if start_line >= len(lines):
            start_line = len(lines) - 1
        
        # Find opening brace
        end_line = start_line
        brace_count = 0
        found_opening = False
        
        for i in range(start_line, len(lines)):
            line = lines[i]
            
            if '{' in line:
                found_opening = True
                brace_count += line.count('{')
                brace_count -= line.count('}')
                end_line = i
                
                if brace_count == 0:
                    break
            elif found_opening:
                brace_count -= line.count('}')
                end_line = i
                
                if brace_count <= 0:
                    break
        
        return Range(
            start=Position(line=start_line, character=0),
            end=Position(line=end_line, character=len(lines[end_line]) if end_line < len(lines) else 0)
        )
    
    def _analyze_dependencies(self) -> None:
        """Analyze dependencies between symbols."""
        
        if not self.context:
            return
        
        for symbol_name, symbol_def in self.context.symbols.items():
            # This would analyze the AST to find dependencies
            # For now, implement basic pattern-based detection
            dependencies = self._find_symbol_dependencies(symbol_def)
            symbol_def.dependencies = dependencies
            
            # Build reverse dependency map
            for dep in dependencies:
                if dep not in self.context.reverse_dependencies:
                    self.context.reverse_dependencies[dep] = set()
                self.context.reverse_dependencies[dep].add(symbol_name)
    
    def _find_symbol_dependencies(self, symbol_def: SymbolDefinition) -> List[str]:
        """Find dependencies of a symbol by analyzing its content."""
        
        if not self.context:
            return []
        
        dependencies = []
        content = self.context.files.get(symbol_def.uri, '')
        lines = content.splitlines()
        
        # Extract content of the symbol definition
        start_line = symbol_def.range.start.line
        end_line = symbol_def.range.end.line
        
        symbol_content = '\n'.join(lines[start_line:end_line + 1])
        
        # Find references to other symbols
        for other_name in self.context.symbols:
            if other_name != symbol_def.name:
                # Simple pattern matching - could be more sophisticated
                pattern = rf'\b{re.escape(other_name)}\b'
                if re.search(pattern, symbol_content):
                    dependencies.append(other_name)
        
        return dependencies
    
    def _resolve_references(self) -> None:
        """Find all references to symbols across the workspace."""
        
        if not self.context:
            return
        
        for uri, content in self.context.files.items():
            lines = content.splitlines()
            
            for line_idx, line in enumerate(lines):
                for symbol_name, symbol_def in self.context.symbols.items():
                    # Find occurrences of symbol name
                    pattern = rf'\b{re.escape(symbol_name)}\b'
                    for match in re.finditer(pattern, line):
                        start_char = match.start()
                        end_char = match.end()
                        
                        # Skip if this is the definition itself
                        if (uri == symbol_def.uri and 
                            symbol_def.range.start.line <= line_idx <= symbol_def.range.end.line):
                            continue
                        
                        # Create reference
                        ref = SymbolReference(
                            uri=uri,
                            range=Range(
                                start=Position(line=line_idx, character=start_char),
                                end=Position(line=line_idx, character=end_char)
                            ),
                            context=line.strip(),
                            kind='reference'
                        )
                        
                        symbol_def.references.append(ref)
    
    def _get_symbol_at_position(self, uri: str, position: Position) -> Optional[str]:
        """Get the symbol name at the given position."""
        
        if not self.context or uri not in self.context.files:
            return None
        
        lines = self.context.files[uri].splitlines()
        if position.line >= len(lines):
            return None
        
        line = lines[position.line]
        if position.character >= len(line):
            return None
        
        # Extract word at position
        start = position.character
        end = position.character
        
        # Find word boundaries
        while start > 0 and (line[start - 1].isalnum() or line[start - 1] == '_'):
            start -= 1
        
        while end < len(line) and (line[end].isalnum() or line[end] == '_'):
            end += 1
        
        if start == end:
            return None
        
        word = line[start:end]
        
        # Check if this word is a known symbol
        return word if word in self.context.symbols else None
    
    def _get_symbol_range_at_position(self, uri: str, position: Position) -> Optional[Range]:
        """Get the range of the symbol at the given position."""
        
        if not self.context or uri not in self.context.files:
            return None
        
        lines = self.context.files[uri].splitlines()
        if position.line >= len(lines):
            return None
        
        line = lines[position.line]
        if position.character >= len(line):
            return None
        
        # Find word boundaries
        start = position.character
        end = position.character
        
        while start > 0 and (line[start - 1].isalnum() or line[start - 1] == '_'):
            start -= 1
        
        while end < len(line) and (line[end].isalnum() or line[end] == '_'):
            end += 1
        
        if start == end:
            return None
        
        return Range(
            start=Position(line=position.line, character=start),
            end=Position(line=position.line, character=end)
        )


def enhance_lsp_with_navigation(language_server):
    """Enhance LSP server with symbol navigation capabilities."""
    
    navigation_engine = SymbolNavigationEngine()
    
    @language_server.feature("textDocument/definition")
    async def go_to_definition(params: DefinitionParams):
        """Handle go-to-definition requests."""
        return navigation_engine.go_to_definition(
            params.text_document.uri,
            params.position
        )
    
    @language_server.feature("textDocument/references")  
    async def find_references(params: ReferenceParams):
        """Handle find-references requests."""
        return navigation_engine.find_references(
            params.text_document.uri,
            params.position,
            params.context.include_declaration
        )
    
    @language_server.feature("textDocument/documentSymbol")
    async def document_symbols(params: DocumentSymbolParams):
        """Handle document symbol requests."""
        return navigation_engine.get_document_symbols(params.text_document.uri)
    
    @language_server.feature("workspace/symbol")
    async def workspace_symbols(params: WorkspaceSymbolParams):
        """Handle workspace symbol search."""
        return navigation_engine.get_workspace_symbols(params.query)
    
    @language_server.feature("textDocument/hover")
    async def hover_info(params: HoverParams):
        """Handle hover requests."""
        return navigation_engine.get_hover_info(
            params.text_document.uri,
            params.position
        )
    
    # Initialize workspace when server starts
    @language_server.feature("initialized")
    async def initialize_navigation(params):
        """Initialize symbol navigation for workspace."""
        workspace = language_server.workspace_index
        if workspace.root_path:
            # Collect all N3 files
            files = {}
            for n3_file in workspace.root_path.rglob("*.n3"):
                try:
                    uri = f"file://{n3_file}"
                    content = n3_file.read_text(encoding="utf-8")
                    files[uri] = content
                except Exception as e:
                    language_server.logger.warning(f"Failed to read {n3_file}: {e}")
            
            # Initialize navigation engine
            navigation_engine.initialize_workspace(str(workspace.root_path), files)
            language_server.logger.info(f"Symbol navigation initialized with {len(files)} files")
    
    language_server.logger.info("Symbol navigation provider registered")