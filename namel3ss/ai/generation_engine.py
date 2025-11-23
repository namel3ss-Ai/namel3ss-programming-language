"""
Code Generation Engine for N3 Language

Generates complete N3 components from natural language descriptions:
- Pages with routing and styling
- Data frames with typed fields
- Reusable components with props
- Complete applications with structure
- Integration between components
"""
import re
import json
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import asyncio
import logging
from ..parser.parser import N3Parser
from .providers import AIProvider, GenerationRequest, GenerationResponse

logger = logging.getLogger(__name__)

@dataclass
class GenerationTemplate:
    """Template for code generation"""
    name: str
    description: str
    pattern: str
    variables: List[str]
    example: str
    category: str

@dataclass
class GeneratedComponent:
    """Result of component generation"""
    code: str
    name: str
    type: str  # 'page', 'frame', 'component', 'app'
    description: str
    dependencies: List[str]
    suggestions: List[str]
    metadata: Dict[str, Any]

class CodeGenerationEngine:
    """AI-powered code generation for N3 language"""
    
    def __init__(self, parser: N3Parser, ai_providers: List[AIProvider]):
        self.parser = parser
        self.ai_providers = ai_providers
        self.templates = self._build_generation_templates()
        self.generation_cache: Dict[str, GeneratedComponent] = {}
        
    def _build_generation_templates(self) -> Dict[str, GenerationTemplate]:
        """Build predefined generation templates"""
        return {
            'page': GenerationTemplate(
                name='Page Component',
                description='Generate a complete page with routing and content',
                pattern='''page {name} at "{route}" {{
    {content}
    
    style {{
        {styles}
    }}
}}''',
                variables=['name', 'route', 'content', 'styles'],
                example='page Home at "/" { <h1>Welcome</h1> }',
                category='ui'
            ),
            
            'frame': GenerationTemplate(
                name='Data Frame',
                description='Generate a data structure with typed fields',
                pattern='''frame {name} {{
    {fields}
}}''',
                variables=['name', 'fields'],
                example='frame User { id: int, name: string, email: string }',
                category='data'
            ),
            
            'component': GenerationTemplate(
                name='UI Component',
                description='Generate a reusable component with props',
                pattern='''component {name}({props}) {{
    {content}
    
    style {{
        {styles}
    }}
}}''',
                variables=['name', 'props', 'content', 'styles'],
                example='component Button(text: string, onClick: function) { <button onclick={onClick}>{text}</button> }',
                category='ui'
            ),
            
            'app': GenerationTemplate(
                name='Application',
                description='Generate a complete application structure',
                pattern='''app {name} {{
    pages: [{pages}]
    theme: "{theme}"
    {config}
}}''',
                variables=['name', 'pages', 'theme', 'config'],
                example='app MyApp { pages: [Home, About], theme: "modern" }',
                category='structure'
            ),
            
            'crud_page': GenerationTemplate(
                name='CRUD Page',
                description='Generate a complete CRUD interface for data management',
                pattern='''page {name}List at "/{entity_lower}" {{
    state items = []
    state loading = false
    state selectedItem = null
    
    async function loadItems() {{
        loading = true
        try {{
            const response = await fetch('/api/{entity_lower}')
            items = await response.json()
        }} catch (error) {{
            console.error('Failed to load {entity_lower}:', error)
        }} finally {{
            loading = false
        }}
    }}
    
    async function deleteItem(id) {{
        try {{
            await fetch(`/api/{entity_lower}/${{id}}`, {{ method: 'DELETE' }})
            items = items.filter(item => item.id !== id)
        }} catch (error) {{
            console.error('Failed to delete {entity_lower}:', error)
        }}
    }}
    
    <div class="crud-container">
        <h1>{entity} Management</h1>
        
        <div class="actions">
            <button onclick="{{loadItems()}}" disabled={{loading}}>
                {{loading ? 'Loading...' : 'Refresh'}}
            </button>
            <a href="/{entity_lower}/new" class="btn-primary">Add New {entity}</a>
        </div>
        
        <div class="data-table">
            <table>
                <thead>
                    <tr>
                        {table_headers}
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {{for item in items {{
                        <tr>
                            {table_cells}
                            <td>
                                <button onclick="{{selectedItem = item}}">Edit</button>
                                <button onclick="{{deleteItem(item.id)}}" class="btn-danger">Delete</button>
                            </td>
                        </tr>
                    }}}}
                </tbody>
            </table>
        </div>
    </div>
    
    style {{
        .crud-container {{
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .actions {{
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
        }}
        
        .data-table {{
            overflow-x: auto;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        
        .btn-primary {{
            background: #007bff;
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            text-decoration: none;
            display: inline-block;
        }}
        
        .btn-danger {{
            background: #dc3545;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
        }}
    }}
}}''',
                variables=['name', 'entity', 'entity_lower', 'table_headers', 'table_cells'],
                example='Complete CRUD interface for managing users',
                category='business'
            ),
            
            'dashboard': GenerationTemplate(
                name='Dashboard Page',
                description='Generate a data dashboard with charts and metrics',
                pattern='''page {name}Dashboard at "/{route}" {{
    state metrics = {{}}
    state chartData = []
    state loading = true
    
    async function loadDashboardData() {{
        loading = true
        try {{
            const [metricsRes, chartRes] = await Promise.all([
                fetch('/api/{entity}/metrics'),
                fetch('/api/{entity}/chart-data')
            ])
            
            metrics = await metricsRes.json()
            chartData = await chartRes.json()
        }} catch (error) {{
            console.error('Failed to load dashboard:', error)
        }} finally {{
            loading = false
        }}
    }}
    
    <div class="dashboard">
        <h1>{title}</h1>
        
        {{if loading {{
            <div class="loading">Loading dashboard...</div>
        }} else {{
            <div class="metrics-grid">
                {metric_cards}
            </div>
            
            <div class="charts-section">
                <div class="chart-container">
                    <h3>Trends</h3>
                    <div class="chart" data-chart="{{chartData}}"></div>
                </div>
            </div>
        }}}}
    </div>
    
    style {{
        .dashboard {{
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: white;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: #2563eb;
            margin-bottom: 8px;
        }}
        
        .metric-label {{
            color: #6b7280;
            font-size: 0.9rem;
        }}
        
        .charts-section {{
            background: white;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .chart-container {{
            height: 400px;
        }}
        
        .loading {{
            text-align: center;
            padding: 60px;
            color: #6b7280;
        }}
    }}
}}''',
                variables=['name', 'route', 'entity', 'title', 'metric_cards'],
                example='Analytics dashboard with metrics and charts',
                category='analytics'
            )
        }
    
    async def generate_component(self, description: str, component_type: Optional[str] = None, 
                               context: Optional[Dict[str, Any]] = None) -> GeneratedComponent:
        """Generate N3 component from natural language description"""
        
        if not self.ai_providers:
            raise RuntimeError("No AI providers available for code generation")
        
        # Use primary AI provider
        provider = self.ai_providers[0]
        
        # Detect component type if not specified
        if not component_type:
            component_type = self._detect_component_type(description)
        
        # Build generation context
        generation_context = {
            'description': description,
            'type': component_type,
            'templates': {k: v for k, v in self.templates.items() if v.category == self._get_category(component_type)},
            'n3_patterns': self._get_n3_patterns(),
            **(context or {})
        }
        
        # Create generation request
        request = GenerationRequest(
            prompt=self._build_generation_prompt(description, component_type),
            context=generation_context,
            max_tokens=1500,
            temperature=0.2,
            model_params={'top_p': 0.9}
        )
        
        try:
            # Generate code
            response = await provider.generate_code(request)
            
            # Parse and validate generated component
            component = self._parse_generated_component(response, component_type, description)
            
            # Cache successful generation
            cache_key = f"{component_type}:{description[:50]}"
            self.generation_cache[cache_key] = component
            
            return component
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            raise
    
    async def generate_application(self, description: str, features: List[str]) -> Dict[str, GeneratedComponent]:
        """Generate complete N3 application from description"""
        
        application = {}
        
        # Parse application requirements
        app_spec = self._parse_application_spec(description, features)
        
        # Generate main app structure
        app_component = await self.generate_component(
            f"Create application '{app_spec['name']}' with features: {', '.join(features)}",
            'app',
            {'spec': app_spec}
        )
        application['app'] = app_component
        
        # Generate required frames/models
        for frame_spec in app_spec.get('frames', []):
            frame_component = await self.generate_component(
                f"Create data frame '{frame_spec['name']}' with fields: {frame_spec['fields']}",
                'frame',
                {'spec': frame_spec}
            )
            application[f"frame_{frame_spec['name'].lower()}"] = frame_component
        
        # Generate pages
        for page_spec in app_spec.get('pages', []):
            page_component = await self.generate_component(
                f"Create page '{page_spec['name']}' for {page_spec['purpose']} at route '{page_spec['route']}'",
                'page',
                {'spec': page_spec}
            )
            application[f"page_{page_spec['name'].lower()}"] = page_component
        
        # Generate components
        for comp_spec in app_spec.get('components', []):
            comp_component = await self.generate_component(
                f"Create reusable component '{comp_spec['name']}' for {comp_spec['purpose']}",
                'component',
                {'spec': comp_spec}
            )
            application[f"component_{comp_spec['name'].lower()}"] = comp_component
        
        return application
    
    def _detect_component_type(self, description: str) -> str:
        """Detect component type from description"""
        description_lower = description.lower()
        
        # Type detection patterns
        if any(word in description_lower for word in ['page', 'route', 'screen', 'view']):
            return 'page'
        elif any(word in description_lower for word in ['frame', 'model', 'data', 'schema', 'structure']):
            return 'frame'
        elif any(word in description_lower for word in ['component', 'widget', 'element', 'button', 'input']):
            return 'component'
        elif any(word in description_lower for word in ['app', 'application', 'project']):
            return 'app'
        elif any(word in description_lower for word in ['crud', 'list', 'manage', 'admin']):
            return 'crud_page'
        elif any(word in description_lower for word in ['dashboard', 'analytics', 'metrics', 'stats']):
            return 'dashboard'
        else:
            return 'component'  # Default fallback
    
    def _get_category(self, component_type: str) -> str:
        """Get category for component type"""
        categories = {
            'page': 'ui',
            'frame': 'data',
            'component': 'ui',
            'app': 'structure',
            'crud_page': 'business',
            'dashboard': 'analytics'
        }
        return categories.get(component_type, 'ui')
    
    def _build_generation_prompt(self, description: str, component_type: str) -> str:
        """Build detailed prompt for AI generation"""
        
        template = self.templates.get(component_type)
        template_info = ""
        if template:
            template_info = f"""
Template for {template.name}:
{template.pattern}

Example: {template.example}
"""
        
        return f"""Generate a complete N3 {component_type} component based on this description:

"{description}"

Requirements:
1. Follow N3 language syntax and best practices
2. Include proper styling with CSS-like syntax  
3. Add interactive features where appropriate
4. Use semantic HTML elements
5. Include error handling for async operations
6. Add helpful comments explaining key functionality
7. Make it production-ready and maintainable

{template_info}

N3 Language Features to use:
- Pages with routing: page Name at "/route" {{ content }}
- Data frames: frame Model {{ field: type }}
- Components with props: component Name(prop: type) {{ content }}
- State management: state variable = value
- Async operations: async/await for API calls
- Loops: for item in items {{ content }}
- Conditionals: if condition {{ content }}
- Event handlers: onclick={{handler}}
- CSS-like styling: style {{ .class {{ property: value }} }}

Generate clean, complete, and immediately usable N3 code."""
    
    def _get_n3_patterns(self) -> Dict[str, str]:
        """Get common N3 patterns for reference"""
        return {
            'api_call': '''async function loadData() {
    try {
        const response = await fetch('/api/endpoint')
        const data = await response.json()
        return data
    } catch (error) {
        console.error('API call failed:', error)
        throw error
    }
}''',
            'form_handling': '''state formData = {}
state errors = {}

function handleSubmit(event) {
    event.preventDefault()
    // Validate form
    // Submit data
}

function updateField(field, value) {
    formData = { ...formData, [field]: value }
}''',
            'list_management': '''state items = []
state filteredItems = []
state searchTerm = ""

function addItem(newItem) {
    items = [...items, newItem]
}

function removeItem(id) {
    items = items.filter(item => item.id !== id)
}

function searchItems(term) {
    searchTerm = term
    filteredItems = items.filter(item => 
        item.name.toLowerCase().includes(term.toLowerCase())
    )
}''',
            'state_management': '''state isLoading = false
state error = null
state data = null

function setState(updates) {
    // Update multiple state values atomically
    Object.assign(this, updates)
}'''
        }
    
    def _parse_generated_component(self, response: GenerationResponse, component_type: str, description: str) -> GeneratedComponent:
        """Parse AI response into structured component"""
        
        code = response.generated_code.strip()
        
        # Extract component name from code
        name = self._extract_component_name(code, component_type)
        
        # Extract dependencies
        dependencies = self._extract_dependencies(code)
        
        # Generate suggestions for improvement
        suggestions = self._generate_suggestions(code, component_type)
        
        return GeneratedComponent(
            code=code,
            name=name,
            type=component_type,
            description=description,
            dependencies=dependencies,
            suggestions=suggestions,
            metadata={
                'confidence': response.confidence,
                'reasoning': response.reasoning,
                'generated_by': 'ai',
                'template_used': component_type in self.templates
            }
        )
    
    def _extract_component_name(self, code: str, component_type: str) -> str:
        """Extract component name from generated code"""
        
        patterns = {
            'page': r'page\s+(\w+)',
            'frame': r'frame\s+(\w+)',
            'component': r'component\s+(\w+)',
            'app': r'app\s+(\w+)',
            'crud_page': r'page\s+(\w+)',
            'dashboard': r'page\s+(\w+)'
        }
        
        pattern = patterns.get(component_type, r'(\w+)')
        match = re.search(pattern, code)
        
        if match:
            return match.group(1)
        else:
            return f"Generated{component_type.title()}"
    
    def _extract_dependencies(self, code: str) -> List[str]:
        """Extract dependencies from generated code"""
        dependencies = []
        
        # Import statements
        import_matches = re.findall(r'import\s+{([^}]+)}\s+from\s+["\']([^"\']+)["\']', code)
        for imports, module in import_matches:
            dependencies.append(module)
        
        # Frame references
        frame_matches = re.findall(r':\s*(\w+(?:\[\])?)(?:\s|,|$)', code)
        for frame in frame_matches:
            if frame not in ['string', 'int', 'float', 'bool', 'any'] and not frame.endswith('[]'):
                dependencies.append(frame)
        
        return list(set(dependencies))
    
    def _generate_suggestions(self, code: str, component_type: str) -> List[str]:
        """Generate improvement suggestions for generated code"""
        suggestions = []
        
        # Check for common patterns
        if 'fetch(' in code and 'try' not in code:
            suggestions.append("Consider adding error handling for API calls")
        
        if 'onclick=' in code and 'function' not in code:
            suggestions.append("Consider extracting event handlers into separate functions")
        
        if component_type == 'page' and 'style {' not in code:
            suggestions.append("Consider adding styling for better visual presentation")
        
        if 'state ' in code and len(re.findall(r'state\s+\w+', code)) > 5:
            suggestions.append("Consider grouping related state into objects")
        
        if component_type in ['crud_page', 'dashboard'] and 'loading' not in code.lower():
            suggestions.append("Consider adding loading states for better UX")
        
        return suggestions
    
    def _parse_application_spec(self, description: str, features: List[str]) -> Dict[str, Any]:
        """Parse application specification from description and features"""
        
        # Extract application name
        name_match = re.search(r'(?:app(?:lication)?|project)\s+(?:called\s+)?["\']?(\w+)["\']?', description.lower())
        app_name = name_match.group(1).title() if name_match else "MyApp"
        
        # Determine required frames based on features
        frames = []
        if any('user' in f.lower() for f in features):
            frames.append({
                'name': 'User',
                'fields': 'id: int, name: string, email: string, createdAt: date'
            })
        
        if any('product' in f.lower() for f in features):
            frames.append({
                'name': 'Product',
                'fields': 'id: int, name: string, price: float, description: string'
            })
        
        if any('order' in f.lower() for f in features):
            frames.append({
                'name': 'Order',
                'fields': 'id: int, userId: int, items: Product[], total: float, status: string'
            })
        
        # Determine required pages
        pages = [
            {'name': 'Home', 'route': '/', 'purpose': 'landing page'}
        ]
        
        if any('dashboard' in f.lower() for f in features):
            pages.append({'name': 'Dashboard', 'route': '/dashboard', 'purpose': 'analytics and metrics'})
        
        if any('admin' in f.lower() or 'manage' in f.lower() for f in features):
            pages.append({'name': 'Admin', 'route': '/admin', 'purpose': 'administration interface'})
        
        # Determine required components
        components = []
        if any('nav' in f.lower() or 'menu' in f.lower() for f in features):
            components.append({'name': 'Navigation', 'purpose': 'site navigation'})
        
        if any('form' in f.lower() for f in features):
            components.append({'name': 'FormInput', 'purpose': 'reusable form inputs'})
        
        return {
            'name': app_name,
            'features': features,
            'frames': frames,
            'pages': pages,
            'components': components,
            'theme': 'modern'
        }
    
    async def enhance_component(self, component: GeneratedComponent, enhancement_request: str) -> GeneratedComponent:
        """Enhance existing component with additional features"""
        
        if not self.ai_providers:
            raise RuntimeError("No AI providers available")
        
        provider = self.ai_providers[0]
        
        prompt = f"""Enhance this N3 {component.type} component:

Current code:
{component.code}

Enhancement request: {enhancement_request}

Requirements:
1. Keep existing functionality intact
2. Add the requested enhancement seamlessly
3. Maintain N3 best practices
4. Update styling if needed
5. Add comments for new functionality

Return the complete enhanced component code."""
        
        request = GenerationRequest(
            prompt=prompt,
            context={
                'original_component': component.__dict__,
                'enhancement': enhancement_request
            },
            max_tokens=2000,
            temperature=0.2
        )
        
        try:
            response = await provider.generate_code(request)
            
            # Create enhanced component
            enhanced = GeneratedComponent(
                code=response.generated_code,
                name=component.name,
                type=component.type,
                description=f"{component.description} + {enhancement_request}",
                dependencies=self._extract_dependencies(response.generated_code),
                suggestions=self._generate_suggestions(response.generated_code, component.type),
                metadata={
                    **component.metadata,
                    'enhanced': True,
                    'enhancement': enhancement_request,
                    'original_description': component.description
                }
            )
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Component enhancement failed: {e}")
            raise