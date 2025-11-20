from typing import Any, Callable
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from opentelemetry import trace
import inspect
import time

from n3_server.adapters import (
    OpenAPIAdapter,
    LangChainAdapter,
    LLMToolWrapper,
    create_llm_tool,
)

router = APIRouter()
tracer = trace.get_tracer(__name__)


class ToolMetadata(BaseModel):
    name: str
    description: str
    inputSchema: dict[str, Any]
    outputSchema: dict[str, Any] | None = None
    tags: list[str] = Field(default_factory=list)
    source: str  # function, openapi, langchain, class


class ToolExecutionRequest(BaseModel):
    name: str
    args: dict[str, Any]
    async_: bool = Field(False, alias="async")


class ToolExecutionResult(BaseModel):
    success: bool
    result: Any | None = None
    error: str | None = None
    durationMs: float


class ToolRegistry:
    """Global tool registry with adapter support."""
    
    def __init__(self):
        self._tools: dict[str, tuple[Callable, ToolMetadata]] = {}
        self.openapi_adapter = OpenAPIAdapter()
        self.langchain_adapter = LangChainAdapter()
        self.llm_wrapper = LLMToolWrapper()
    
    def register(
        self,
        func: Callable,
        description: str = "",
        tags: list[str] | None = None,
    ):
        """Register a tool function."""
        sig = inspect.signature(func)
        
        # Build input schema from signature
        input_schema = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        
        for name, param in sig.parameters.items():
            input_schema["properties"][name] = {
                "type": "string",  # Simplified
            }
            if param.default == inspect.Parameter.empty:
                input_schema["required"].append(name)
        
        metadata = ToolMetadata(
            name=func.__name__,
            description=description or func.__doc__ or "",
            inputSchema=input_schema,
            tags=tags or [],
            source="function",
        )
        
        self._tools[func.__name__] = (func, metadata)
        return func
    
    def get_tool(self, name: str) -> tuple[Callable, ToolMetadata] | None:
        """Get a registered tool."""
        return self._tools.get(name)
    
    def list_tools(self) -> list[ToolMetadata]:
        """List all registered tools."""
        return [metadata for _, metadata in self._tools.values()]
    
    async def execute(
        self,
        name: str,
        args: dict[str, Any],
    ) -> Any:
        """Execute a tool."""
        tool_entry = self.get_tool(name)
        if not tool_entry:
            raise ValueError(f"Tool not found: {name}")
        
        func, _ = tool_entry
        
        # Execute with tracing
        with tracer.start_as_current_span(f"tool.{name}") as span:
            span.set_attribute("tool.name", name)
            span.set_attribute("tool.args", str(args))
            
            if inspect.iscoroutinefunction(func):
                result = await func(**args)
            else:
                result = func(**args)
            
            return result


# Global registry instance
registry = ToolRegistry()


def tool(description: str = "", tags: list[str] | None = None):
    """Decorator to register a tool."""
    def decorator(func: Callable):
        return registry.register(func, description, tags)
    return decorator


# Example tools
@tool(description="Add two numbers", tags=["math"])
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@tool(description="Multiply two numbers", tags=["math"])
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@router.get("", response_model=list[ToolMetadata])
async def list_tools():
    """List all registered tools."""
    return registry.list_tools()


@router.post("/execute", response_model=ToolExecutionResult)
async def execute_tool(request: ToolExecutionRequest):
    """Execute a registered tool."""
    start = time.time()
    
    try:
        result = await registry.execute(request.name, request.args)
        duration = (time.time() - start) * 1000
        
        return ToolExecutionResult(
            success=True,
            result=result,
            durationMs=duration,
        )
    except Exception as e:
        duration = (time.time() - start) * 1000
        return ToolExecutionResult(
            success=False,
            error=str(e),
            durationMs=duration,
        )


class OpenAPIImportRequest(BaseModel):
    specUrl: str
    baseUrl: str | None = None
    authToken: str | None = None
    namePrefix: str = ""
    operationIds: list[str] | None = None


class OpenAPIImportResponse(BaseModel):
    success: bool
    toolsImported: int
    toolNames: list[str]
    error: str | None = None


@router.post("/import/openapi", response_model=OpenAPIImportResponse)
async def import_openapi_tools(request: OpenAPIImportRequest):
    """Import tools from OpenAPI specification."""
    try:
        # Create operation filter if specific operations requested
        operation_filter = None
        if request.operationIds:
            operation_filter = lambda op: op.operation_id in request.operationIds
        
        # Import tools
        tools = await registry.openapi_adapter.import_from_url(
            spec_url=request.specUrl,
            base_url=request.baseUrl,
            auth_token=request.authToken,
            operation_filter=operation_filter,
            name_prefix=request.namePrefix,
        )
        
        # Register imported tools
        tool_names = []
        for tool_func in tools:
            metadata = tool_func._tool_metadata
            registry.register(
                tool_func,
                description=metadata["description"],
                tags=metadata.get("tags", []),
            )
            tool_names.append(metadata["name"])
        
        return OpenAPIImportResponse(
            success=True,
            toolsImported=len(tools),
            toolNames=tool_names,
        )
    
    except Exception as e:
        return OpenAPIImportResponse(
            success=False,
            toolsImported=0,
            toolNames=[],
            error=str(e),
        )


class LLMToolCreateRequest(BaseModel):
    name: str
    description: str
    llmName: str
    systemPrompt: str | None = None
    temperature: float = 0.7
    maxTokens: int = 1024
    responseFormat: str = "text"
    outputSchema: dict[str, Any] | None = None
    inputSchema: dict[str, Any] | None = None


class LLMToolCreateResponse(BaseModel):
    success: bool
    toolName: str
    error: str | None = None


@router.post("/create/llm", response_model=LLMToolCreateResponse)
async def create_llm_tool_endpoint(request: LLMToolCreateRequest):
    """Create an LLM-powered tool."""
    try:
        # Create LLM tool
        tool_func = registry.llm_wrapper.create_tool(
            name=request.name,
            description=request.description,
            llm_name=request.llmName,
            system_prompt=request.systemPrompt,
            temperature=request.temperature,
            max_tokens=request.maxTokens,
            response_format=request.responseFormat,
            output_schema=request.outputSchema,
            input_schema=request.inputSchema,
        )
        
        # Register tool
        metadata = tool_func._tool_metadata
        registry.register(
            tool_func,
            description=metadata["description"],
            tags=metadata.get("tags", []),
        )
        
        return LLMToolCreateResponse(
            success=True,
            toolName=request.name,
        )
    
    except Exception as e:
        return LLMToolCreateResponse(
            success=False,
            toolName=request.name,
            error=str(e),
        )


@router.post("/register")
async def register_tool(spec: dict[str, Any]):
    """Register a tool from OpenAPI or LangChain spec."""
    # TODO: Implement OpenAPI and LangChain adapters
    raise HTTPException(status_code=501, detail="Not implemented")
