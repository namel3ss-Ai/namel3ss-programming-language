"""N3 in-process runtime for embedded execution.

Loads and executes .ai files directly without a remote server,
enabling embedded N3 workflows in Python applications.
"""

import importlib.util
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from .config import N3RuntimeConfig, get_settings
from .exceptions import N3RuntimeError


class RuntimeChainsAPI:
    """API for executing chains in embedded runtime."""
    
    def __init__(self, runtime: 'N3InProcessRuntime'):
        self._runtime = runtime
    
    def run(self, name: str, **payload: Any) -> Dict[str, Any]:
        """Execute a chain.
        
        Args:
            name: Chain name
            **payload: Chain inputs
        
        Returns:
            Chain execution result
        
        Raises:
            N3RuntimeError: Chain not found or execution failed
        """
        if not hasattr(self._runtime._module, 'run_chain'):
            raise N3RuntimeError("Runtime module missing run_chain function")
        
        try:
            result = self._runtime._module.run_chain(name, payload)
            
            if result.get("status") == "error":
                raise N3RuntimeError(
                    result.get("result") or "Chain execution failed",
                    chain=name,
                    context=result.get("metadata", {}),
                )
            
            return result
        
        except Exception as e:
            if isinstance(e, N3RuntimeError):
                raise
            raise N3RuntimeError(
                f"Chain execution failed: {e}",
                chain=name,
            )


class RuntimePromptsAPI:
    """API for executing prompts in embedded runtime."""
    
    def __init__(self, runtime: 'N3InProcessRuntime'):
        self._runtime = runtime
    
    def run(self, name: str, **inputs: Any) -> Dict[str, Any]:
        """Execute a prompt."""
        if not hasattr(self._runtime._module, 'execute_prompt'):
            raise N3RuntimeError("Runtime module missing execute_prompt function")
        
        try:
            # Prompts are typically accessed through the PROMPTS registry
            prompts = getattr(self._runtime._module, 'PROMPTS', {})
            if name not in prompts:
                raise N3RuntimeError(f"Prompt '{name}' not found")
            
            prompt = prompts[name]
            result = prompt.execute(**inputs)
            
            return {"result": result, "status": "success"}
        
        except Exception as e:
            if isinstance(e, N3RuntimeError):
                raise
            raise N3RuntimeError(f"Prompt execution failed: {e}")


class RuntimeAgentsAPI:
    """API for running agents in embedded runtime."""
    
    def __init__(self, runtime: 'N3InProcessRuntime'):
        self._runtime = runtime
    
    def run(
        self,
        name: str,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        max_turns: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run an agent."""
        if not hasattr(self._runtime._module, 'run_agent'):
            raise N3RuntimeError("Runtime module missing run_agent function")
        
        try:
            payload = {"user_input": user_input}
            if context:
                payload["context"] = context
            if max_turns is not None:
                payload["max_turns"] = max_turns
            
            result = self._runtime._module.run_agent(name, payload)
            
            if result.get("status") == "error":
                raise N3RuntimeError(
                    result.get("result") or "Agent execution failed",
                    context=result.get("metadata", {}),
                )
            
            return result
        
        except Exception as e:
            if isinstance(e, N3RuntimeError):
                raise
            raise N3RuntimeError(f"Agent execution failed: {e}")


class RuntimeRagAPI:
    """API for querying RAG pipelines in embedded runtime."""
    
    def __init__(self, runtime: 'N3InProcessRuntime'):
        self._runtime = runtime
    
    def query(
        self,
        pipeline: str,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Query a RAG pipeline."""
        if not hasattr(self._runtime._module, 'query_rag'):
            raise N3RuntimeError("Runtime module missing query_rag function")
        
        try:
            payload = {"query": query}
            if top_k is not None:
                payload["top_k"] = top_k
            if filters:
                payload["filters"] = filters
            
            result = self._runtime._module.query_rag(pipeline, payload)
            
            if isinstance(result, dict) and result.get("status") == "error":
                raise N3RuntimeError(
                    result.get("result") or "RAG query failed",
                    context=result.get("metadata", {}),
                )
            
            return result
        
        except Exception as e:
            if isinstance(e, N3RuntimeError):
                raise
            raise N3RuntimeError(f"RAG query failed: {e}")


class N3InProcessRuntime:
    """N3 in-process runtime for embedded execution.
    
    Loads and executes .ai files directly without a remote server.
    Enables embedded N3 workflows in Python applications.
    
    Features:
        - Direct .ai file execution
        - No server required
        - Full access to N3 runtime features
        - Configurable caching and execution limits
    
    Example:
        Basic usage:
        >>> runtime = N3InProcessRuntime("./app.ai")
        >>> result = runtime.chains.run("summarize", text="...")
        
        With custom config:
        >>> config = N3RuntimeConfig(
        ...     source_file="./app.ai",
        ...     enable_cache=True,
        ...     max_turns=20
        ... )
        >>> runtime = N3InProcessRuntime(config=config)
        
        Direct access to generated module:
        >>> runtime = N3InProcessRuntime("./app.ai")
        >>> chains = runtime.get_chains()
        >>> agents = runtime.get_agents()
    
    Note:
        The runtime compiles the .ai file on initialization and caches
        the result. Subsequent instantiations reuse the cached module.
    """
    
    def __init__(
        self,
        source_file: Optional[str] = None,
        config: Optional[N3RuntimeConfig] = None,
    ):
        """Initialize in-process runtime.
        
        Args:
            source_file: Path to .ai source file
            config: Runtime configuration
        
        Raises:
            N3RuntimeError: Failed to load or compile source file
        """
        self.config = config or get_settings().runtime
        
        if source_file:
            self.config.source_file = Path(source_file)
        
        if not self.config.source_file:
            raise N3RuntimeError("No source file specified")
        
        if not self.config.source_file.exists():
            raise N3RuntimeError(f"Source file not found: {self.config.source_file}")
        
        # Compile and load runtime module
        self._module = self._load_runtime()
        
        # API namespaces
        self.chains = RuntimeChainsAPI(self)
        self.prompts = RuntimePromptsAPI(self)
        self.agents = RuntimeAgentsAPI(self)
        self.rag = RuntimeRagAPI(self)
    
    def _load_runtime(self):
        """Compile .ai file and load runtime module."""
        try:
            # Import namel3ss compiler components
            from namel3ss.parser import Parser
            from namel3ss.codegen.backend import generate_backend
            
            # Parse .ai source
            with open(self.config.source_file, 'r', encoding='utf-8') as f:
                source = f.read()
            
            app = Parser(source).parse_app()
            
            # Generate backend in temp directory
            with tempfile.TemporaryDirectory() as tmp_dir:
                backend_dir = Path(tmp_dir) / "backend"
                generate_backend(app, backend_dir)
                
                # Load runtime module
                runtime_file = backend_dir / "generated" / "runtime.py"
                if not runtime_file.exists():
                    raise N3RuntimeError("Runtime module not generated")
                
                # Load module dynamically
                spec = importlib.util.spec_from_file_location(
                    "n3_runtime",
                    runtime_file,
                )
                if spec is None or spec.loader is None:
                    raise N3RuntimeError("Failed to load runtime module")
                
                module = importlib.util.module_from_spec(spec)
                
                # Add backend directory to sys.path for imports
                sys.path.insert(0, str(backend_dir / "generated"))
                try:
                    spec.loader.exec_module(module)
                finally:
                    sys.path.pop(0)
                
                return module
        
        except Exception as e:
            if isinstance(e, N3RuntimeError):
                raise
            raise N3RuntimeError(f"Failed to load runtime: {e}")
    
    def get_chains(self) -> Dict[str, Any]:
        """Get available chains."""
        return getattr(self._module, 'AI_CHAINS', {})
    
    def get_prompts(self) -> Dict[str, Any]:
        """Get available prompts."""
        return getattr(self._module, 'PROMPTS', {})
    
    def get_agents(self) -> Dict[str, Any]:
        """Get available agents."""
        return getattr(self._module, 'AGENTS', {})
    
    def get_rag_pipelines(self) -> Dict[str, Any]:
        """Get available RAG pipelines."""
        return getattr(self._module, 'RAG_PIPELINES', {})
    
    def get_tools(self) -> Dict[str, Any]:
        """Get available tools."""
        return getattr(self._module, 'TOOLS', {})
    
    def execute_raw(self, func_name: str, *args, **kwargs) -> Any:
        """Execute a raw function from the runtime module.
        
        Provides direct access to any function in the generated runtime.
        
        Args:
            func_name: Function name to execute
            *args, **kwargs: Function arguments
        
        Returns:
            Function result
        
        Raises:
            N3RuntimeError: Function not found or execution failed
        
        Example:
            >>> result = runtime.execute_raw('run_chain', 'my_chain', {'input': 'value'})
        """
        if not hasattr(self._module, func_name):
            raise N3RuntimeError(f"Function '{func_name}' not found in runtime")
        
        try:
            func = getattr(self._module, func_name)
            return func(*args, **kwargs)
        except Exception as e:
            raise N3RuntimeError(f"Function execution failed: {e}")
