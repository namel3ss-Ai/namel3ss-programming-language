# Namel3ss API Reference

This document provides a comprehensive API reference for the Namel3ss programming language Python package.

## Core Modules

### `namel3ss.parser`

#### `Parser`

Main entry point for parsing N3 source code.

```python
from namel3ss.parser import Parser

class Parser:
    def __init__(self, source: str = "", *, module_name: Optional[str] = None, path: str = ""):
        """Initialize parser with source code.
        
        Args:
            source: N3 source code to parse
            module_name: Optional module name override
            path: Source file path for error reporting
        """
    
    def parse(self) -> Module:
        """Parse source into Module AST using the unified parser.
        
        Returns:
            Module AST node
            
        Raises:
            N3SyntaxError: If source has syntax errors
            N3SemanticError: If source has semantic errors
        """
    
    def parse_app(self) -> App:
        """Parse source and extract the App node.
        
        Returns:
            App AST node
            
        Raises:
            CLIRuntimeError: If no app is found in the module
        """
```

#### Error Types

```python
from namel3ss.parser import N3SyntaxError

class N3SyntaxError(Exception):
    """Syntax error in N3 source code."""
    def __init__(self, message: str, line: int = 0, column: int = 0, path: str = ""):
        self.message = message
        self.line = line
        self.column = column
        self.path = path
```

### `namel3ss.loader`

Functions for loading complete N3 programs with imports.

```python
from namel3ss.loader import load_program, load_app_from_file

def load_program(root_path: str | PathLike[str]) -> Program:
    """Load a complete N3 program from a directory.
    
    Discovers all .ai files in the directory and resolves imports.
    
    Args:
        root_path: Directory containing N3 source files
        
    Returns:
        Program AST with all modules loaded
    """

def load_app_from_file(file_path: str | PathLike[str]) -> App:
    """Load a single app from an N3 file.
    
    Args:
        file_path: Path to .ai file
        
    Returns:
        App AST node
    """
```

### `namel3ss.resolver`

Module resolution and import handling.

```python
from namel3ss.resolver import resolve_program, resolve_app

def resolve_program(program: Program) -> ResolvedProgram:
    """Resolve all imports and references in a program.
    
    Args:
        program: Program AST from loader
        
    Returns:
        ResolvedProgram with all references resolved
        
    Raises:
        N3ResolutionError: If imports cannot be resolved
    """

def resolve_app(app: App) -> App:
    """Resolve references within a single app.
    
    Args:
        app: App AST node
        
    Returns:
        App with resolved references
    """
```

### `namel3ss.codegen`

Code generation for backend and frontend.

#### Backend Generation

```python
from namel3ss.codegen.backend import generate_backend

def generate_backend(
    app: App,
    output_dir: str,
    *,
    embed_insights: bool = False,
    enable_realtime: bool = False,
    connector_config: Optional[Dict[str, Any]] = None,
    export_schemas: bool = False,
    schema_version: str = "1.0.0",
) -> None:
    """Generate FastAPI backend from App AST.
    
    Args:
        app: App AST node
        output_dir: Directory to write generated code
        embed_insights: Embed insight results in endpoint responses
        enable_realtime: Include WebSocket streaming support
        connector_config: Database and external service configuration
        export_schemas: Generate OpenAPI schemas and SDKs
        schema_version: Version for exported schemas
    """
```

#### Frontend Generation

```python
from namel3ss.codegen.frontend import generate_site

def generate_site(
    app: App,
    output_dir: str,
    *,
    enable_realtime: bool = False,
    target: str = "static",
) -> None:
    """Generate frontend from App AST.
    
    Args:
        app: App AST node
        output_dir: Directory to write generated code
        enable_realtime: Include real-time features
        target: Frontend target ("static" or "react-vite")
    """
```

### `namel3ss.providers`

LLM provider abstraction layer.

#### Core Classes

```python
from namel3ss.providers import N3Provider, ProviderMessage, ProviderResponse

class N3Provider(ABC):
    """Abstract base class for all LLM providers."""
    
    @abstractmethod
    async def generate(
        self,
        messages: List[ProviderMessage],
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate completion from messages."""
    
    @abstractmethod
    async def stream(
        self,
        messages: List[ProviderMessage],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream completion chunks."""
    
    def supports_streaming(self) -> bool:
        """Check if provider supports streaming."""

@dataclass
class ProviderMessage:
    role: str  # "system", "user", "assistant"
    content: str

@dataclass  
class ProviderResponse:
    output_text: str
    finish_reason: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    raw: Dict[str, Any]
```

#### Factory Functions

```python
from namel3ss.providers.factory import create_provider_from_spec

def create_provider_from_spec(
    provider_type: str,
    model: str,
    *,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> N3Provider:
    """Create provider instance from specification.
    
    Args:
        provider_type: Provider type ("openai", "anthropic", "google", etc.)
        model: Model name
        name: Optional provider instance name
        config: Additional configuration parameters
        
    Returns:
        Configured provider instance
        
    Example:
        provider = create_provider_from_spec(
            "openai", 
            "gpt-4",
            config={"temperature": 0.7, "max_tokens": 1000}
        )
    """
```

#### Integration with Runtime

```python
from namel3ss.providers.integration import (
    ProviderLLMBridge,
    run_chain_with_provider,
    run_agent_with_provider,
)

class ProviderLLMBridge(BaseLLM):
    """Bridge N3Provider to existing BaseLLM interface."""
    
    def __init__(
        self,
        provider: N3Provider,
        *,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
    ):
        """Initialize bridge with provider instance."""

async def run_chain_with_provider(
    chain_steps: List[Any],
    provider: N3Provider,
    initial_input: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a chain using N3Provider for LLM steps."""

async def run_agent_with_provider(
    agent_def: Any,
    provider: N3Provider,
    user_input: str,
    tools: Optional[Dict[str, Any]] = None,
    max_turns: Optional[int] = None,
) -> Dict[str, Any]:
    """Run an agent using N3Provider."""
```

### `namel3ss.memory`

Memory and state management system.

```python
from namel3ss.memory import MemoryManager, MemoryConfig

@dataclass
class MemoryConfig:
    provider: str = "memory"  # "memory", "redis", "sqlite"
    redis_url: Optional[str] = None
    sqlite_database: Optional[str] = None
    default_ttl: Optional[int] = None
    max_memory_size: int = 1000

class MemoryManager:
    def __init__(self, config: MemoryConfig):
        """Initialize memory manager with configuration."""
    
    # Key-value operations
    async def get(self, key: str) -> Any:
        """Get value by key."""
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> None:
        """Set key-value pair with optional TTL."""
    
    async def delete(self, key: str) -> bool:
        """Delete key."""
    
    # List operations
    async def lpush(self, key: str, *values: Any) -> int:
        """Push values to list (left side)."""
    
    async def rpush(self, key: str, *values: Any) -> int:
        """Push values to list (right side)."""
    
    async def lrange(self, key: str, start: int, stop: int) -> List[Any]:
        """Get list slice."""
    
    async def llen(self, key: str) -> int:
        """Get list length."""
    
    # Conversation memory
    async def add_message(
        self,
        conversation_key: str,
        message: Dict[str, Any],
        max_messages: Optional[int] = None,
    ) -> None:
        """Add message to conversation history."""
    
    async def get_conversation(
        self,
        conversation_key: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get conversation history."""
```

### `namel3ss.eval`

Evaluation and testing framework.

#### Core Classes

```python
from namel3ss.eval import (
    EvalContext,
    EvalMetric,
    EvalMetricResult,
    EvalSuiteRunner,
    LLMJudge,
)

@dataclass
class EvalContext:
    input: Dict[str, Any]
    output: Any
    latency: float
    cost: float
    metadata: Dict[str, Any]

class EvalMetric(ABC):
    @abstractmethod
    async def compute(self, context: EvalContext) -> EvalMetricResult:
        """Compute metric from evaluation context."""

@dataclass
class EvalMetricResult:
    name: str
    value: float
    metadata: Dict[str, Any]

class LLMJudge:
    def __init__(
        self,
        model_provider: N3Provider,
        rubric: Dict[str, Any],
    ):
        """Initialize LLM judge with model and rubric."""
    
    async def score(self, context: EvalContext) -> Dict[str, Any]:
        """Score output using LLM judge."""

class EvalSuiteRunner:
    def __init__(
        self,
        suite_name: str,
        dataset_rows: List[Dict[str, Any]],
        chain_executor: Any,
        metrics: List[EvalMetric],
        judge: Optional[LLMJudge] = None,
    ):
        """Initialize evaluation suite runner."""
    
    async def run_suite(
        self,
        *,
        batch_size: int = 1,
        continue_on_error: bool = True,
        limit: Optional[int] = None,
    ) -> EvalSuiteResult:
        """Run evaluation suite on dataset."""
```

#### Built-in Metrics

```python
from namel3ss.eval.metrics import (
    BuiltinLatencyMetric,
    BuiltinCostMetric,
    create_metric,
)

def create_metric(
    metric_type: str,
    name: str,
    config: Optional[Dict[str, Any]] = None,
) -> EvalMetric:
    """Create metric instance.
    
    Args:
        metric_type: "builtin", "llm_judge", "custom"
        name: Metric name
        config: Metric-specific configuration
        
    Returns:
        EvalMetric instance
    """

# Built-in metrics
latency_metric = BuiltinLatencyMetric()
cost_metric = BuiltinCostMetric()
```

### `namel3ss.rag`

Retrieval-Augmented Generation system.

```python
from namel3ss.rag import (
    VectorStore,
    EmbeddingProvider,
    RAGPipeline,
    DocumentChunker,
)

class VectorStore(ABC):
    @abstractmethod
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        """Add documents with embeddings."""
    
    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""

class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
    
    @abstractmethod
    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for query."""

class DocumentChunker:
    def __init__(
        self,
        strategy: str = "recursive",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        """Initialize document chunker."""
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata."""

class RAGPipeline:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        llm_provider: N3Provider,
        config: Dict[str, Any],
    ):
        """Initialize RAG pipeline."""
    
    async def query(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute RAG query."""
```

### `namel3ss.cli`

Command-line interface components.

#### Context and Configuration

```python
from namel3ss.cli.context import CLIContext, get_cli_context
from namel3ss.cli.loading import load_n3_app

class CLIContext:
    config: WorkspaceConfig
    plugin_manager: PluginManager
    verbosity: int

def get_cli_context(args: argparse.Namespace) -> CLIContext:
    """Get CLI context from parsed arguments."""

def load_n3_app(
    source_path: Path,
    workspace: Optional[WorkspaceConfig] = None,
) -> App:
    """Load and validate N3 app from file."""
```

#### Command Functions

```python
from namel3ss.cli.commands import (
    cmd_build,
    cmd_run,
    cmd_eval,
    cmd_train,
    cmd_deploy,
    cmd_doctor,
)

def cmd_build(args: argparse.Namespace) -> None:
    """Handle the 'build' subcommand."""

def cmd_run(args: argparse.Namespace) -> None:
    """Handle the 'run' subcommand."""

def cmd_eval(args: argparse.Namespace) -> None:
    """Handle the 'eval' subcommand."""

def cmd_train(args: argparse.Namespace) -> None:
    """Handle the 'train' subcommand."""

def cmd_deploy(args: argparse.Namespace) -> None:
    """Handle the 'deploy' subcommand."""

def cmd_doctor(args: argparse.Namespace) -> None:
    """Handle the 'doctor' subcommand."""
```

### `namel3ss.ast`

Abstract Syntax Tree node definitions.

#### Core Application Nodes

```python
from namel3ss.ast import App, Page, Dataset, Model

@dataclass
class App:
    name: str
    database: Optional[str] = None
    theme: Theme = field(default_factory=Theme)
    variables: List[VariableAssignment] = field(default_factory=list)
    datasets: List[Dataset] = field(default_factory=list)
    pages: List[Page] = field(default_factory=list)
    models: List[Model] = field(default_factory=list)
    # AI components
    llms: List[LLMDefinition] = field(default_factory=list)
    prompts: List[Prompt] = field(default_factory=list)
    chains: List[Chain] = field(default_factory=list)
    agents: List[AgentDefinition] = field(default_factory=list)
    memories: List[Memory] = field(default_factory=list)
    # Evaluation
    evaluators: List[Evaluator] = field(default_factory=list)
    metrics: List[Metric] = field(default_factory=list)
    guardrails: List[Guardrail] = field(default_factory=list)

@dataclass
class Page:
    name: str
    route: str
    title: Optional[str] = None
    description: Optional[str] = None
    components: List[Any] = field(default_factory=list)
    layout: LayoutMeta = field(default_factory=LayoutMeta)
    refresh_policy: Optional[RefreshPolicy] = None

@dataclass
class Dataset:
    name: str
    source_type: str  # "sql", "csv", "rest", "python"
    source_config: Dict[str, Any] = field(default_factory=dict)
    schema: Dict[str, str] = field(default_factory=dict)
    transformations: List[DatasetTransformStep] = field(default_factory=list)
    cache_ttl: Optional[int] = None
    refresh_policy: Optional[str] = None
```

#### AI Component Nodes

```python
from namel3ss.ast.ai import (
    LLMDefinition,
    Prompt,
    Chain,
    Memory,
    AgentDefinition,
)

@dataclass
class LLMDefinition:
    name: str
    model: Optional[str] = None
    provider: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    tools: List[str] = field(default_factory=list)

@dataclass
class Prompt:
    name: str
    model: Optional[str] = None
    template: str = ""
    args: List[PromptArgument] = field(default_factory=list)
    output_schema: Optional[OutputSchema] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Chain:
    name: str
    description: Optional[str] = None
    steps: List[ChainStep] = field(default_factory=list)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    error_handling: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Memory:
    name: str
    scope: str  # "user", "session", "global"
    kind: str   # "conversation", "key_value", "list", "vector"
    max_items: Optional[int] = None
    ttl: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentDefinition:
    name: str
    description: Optional[str] = None
    system_prompt: str = ""
    model: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    max_turns: int = 10
    temperature: float = 0.7
```

### Configuration and Environment

#### Environment Variables

```bash
# Core configuration
export NAMEL3SS_VERBOSE=1                    # Enable verbose output
export NAMEL3SS_RERAISE=1                    # Re-raise exceptions for debugging

# Provider API keys
export NAMEL3SS_PROVIDER_OPENAI_API_KEY="sk-..."
export NAMEL3SS_PROVIDER_ANTHROPIC_API_KEY="sk-ant-..."
export NAMEL3SS_PROVIDER_GOOGLE_PROJECT_ID="my-project"
export NAMEL3SS_PROVIDER_AZURE_API_KEY="..."

# Memory configuration
export NAMEL3SS_MEMORY_PROVIDER="redis"
export NAMEL3SS_MEMORY_REDIS_URL="redis://localhost:6379"
export NAMEL3SS_MEMORY_TTL="86400"

# Vector database
export NAMEL3SS_VECTOR_STORE="chroma"
export NAMEL3SS_CHROMA_HOST="localhost"
export NAMEL3SS_CHROMA_PORT="8000"

# Observability
export NAMEL3SS_LOG_LEVEL="INFO"
export NAMEL3SS_ENABLE_TRACING="true"
export NAMEL3SS_METRICS_ENDPOINT="http://prometheus:9090"
```

## Usage Patterns

### Basic Application Development

```python
from namel3ss import Parser, generate_backend, generate_site

# 1. Parse N3 source
source = '''
app "My App" {
    description: "A simple application"
}

page "Home" at "/" {
    show text "Hello, World!"
}
'''

parser = Parser(source)
app = parser.parse_app()

# 2. Generate code
generate_backend(app, "./backend")
generate_site(app, "./frontend")

# 3. Run development server
# namel3ss run app.ai
```

### AI Chain Development

```python
from namel3ss.providers import create_provider_from_spec
from namel3ss.providers.integration import run_chain_with_provider

# Create provider
provider = create_provider_from_spec("openai", "gpt-4")

# Define chain steps (normally from AST)
chain_steps = [
    {
        "kind": "llm",
        "target": "analyze_sentiment",
        "options": {"text": "{{input.text}}"}
    }
]

# Execute chain
result = await run_chain_with_provider(
    chain_steps=chain_steps,
    provider=provider,
    initial_input={"text": "I love this product!"}
)

print(result["response"])  # Sentiment analysis result
```

### Evaluation and Testing

```python
from namel3ss.eval import EvalSuiteRunner, create_metric

# Create metrics
latency_metric = create_metric("builtin", "latency")
quality_metric = create_metric("llm_judge", "quality", {
    "judge_model": "gpt-4",
    "rubric": {"helpfulness": {"scale": [1, 5]}}
})

# Test data
test_data = [
    {"input": "What is AI?", "expected": "Artificial Intelligence explanation"},
    {"input": "How does ML work?", "expected": "Machine Learning explanation"},
]

# Run evaluation
runner = EvalSuiteRunner(
    suite_name="qa_eval",
    dataset_rows=test_data,
    chain_executor=my_chain_executor,
    metrics=[latency_metric, quality_metric]
)

results = await runner.run_suite(batch_size=2)
print(f"Average latency: {results.aggregate_metrics['latency']['mean']}")
```

### Memory Integration

```python
from namel3ss.memory import MemoryManager, MemoryConfig

# Configure memory
config = MemoryConfig(
    provider="redis",
    redis_url="redis://localhost:6379",
    default_ttl=3600
)

memory = MemoryManager(config)

# Store conversation
await memory.add_message(
    "user:123:chat",
    {"role": "user", "content": "Hello"},
    max_messages=50
)

# Retrieve conversation
history = await memory.get_conversation("user:123:chat", limit=10)
```

This API reference provides a comprehensive overview of the main classes and functions available in the Namel3ss package, organized by module and functionality area.