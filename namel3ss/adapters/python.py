"""Python function adapter for calling Python code from N3.

Enables type-safe Python FFI with automatic validation and error handling.
"""

import asyncio
import importlib
import inspect
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, Field, create_model

from .base import (
    AdapterConfig,
    AdapterType,
    BaseAdapter,
    AdapterExecutionError,
    AdapterTimeoutError,
)


class PythonAdapterConfig(AdapterConfig):
    """Configuration for Python function adapter."""
    
    adapter_type: AdapterType = Field(default=AdapterType.PYTHON)
    
    # Python function reference
    module: str = Field(..., description="Python module path (e.g., 'myapp.tools.calc')")
    function: str = Field(..., description="Function name to call")
    
    # Or direct function reference
    callable: Optional[Callable] = Field(None, description="Direct Python callable")
    
    # Async support
    is_async: bool = Field(default=False, description="Whether function is async")
    
    class Config:
        arbitrary_types_allowed = True


class PythonAdapter(BaseAdapter):
    """Adapter for calling Python functions from N3.
    
    Supports both sync and async functions with automatic schema generation
    from type hints.
    
    Features:
        - Automatic schema from type hints
        - Sync and async function support
        - Timeout enforcement
        - Exception context preservation
    
    Example:
        Define Python function:
        >>> # myapp/tools.py
        >>> def calculate_tax(amount: float, rate: float) -> float:
        ...     return amount * rate
        
        Register in N3:
        ```n3
        tool "calculate_tax" {
          adapter: "python"
          module: "myapp.tools"
          function: "calculate_tax"
          version: "1.0"
        }
        ```
        
        Or programmatically:
        >>> from namel3ss.adapters.python import PythonAdapter, PythonAdapterConfig
        >>> 
        >>> config = PythonAdapterConfig(
        ...     name="tax_calc",
        ...     module="myapp.tools",
        ...     function="calculate_tax"
        ... )
        >>> adapter = PythonAdapter(config)
        >>> result = adapter.execute(amount=100, rate=0.08)
        >>> print(result.output)
        8.0
    """
    
    def __init__(self, config: PythonAdapterConfig):
        super().__init__(config)
        self.config: PythonAdapterConfig = config
        self._func: Optional[Callable] = None
        self._load_function()
        self._generate_schemas()
    
    def _load_function(self):
        """Load Python function from module or use provided callable."""
        if self.config.callable:
            self._func = self.config.callable
            return
        
        try:
            module = importlib.import_module(self.config.module)
            self._func = getattr(module, self.config.function)
            
            if not callable(self._func):
                raise AdapterExecutionError(
                    f"{self.config.function} is not callable",
                    adapter_name=self.config.name,
                    adapter_type=self.config.adapter_type,
                )
            
            # Detect if async
            if asyncio.iscoroutinefunction(self._func):
                self.config.is_async = True
        
        except ImportError as e:
            raise AdapterExecutionError(
                f"Failed to import module {self.config.module}: {e}",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
        
        except AttributeError as e:
            raise AdapterExecutionError(
                f"Function {self.config.function} not found in {self.config.module}",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
    
    def _generate_schemas(self):
        """Generate Pydantic schemas from function type hints."""
        if not self._func:
            return
        
        try:
            sig = inspect.signature(self._func)
            
            # Generate input schema from parameters
            fields = {}
            for param_name, param in sig.parameters.items():
                if param.annotation == inspect.Parameter.empty:
                    fields[param_name] = (Any, ...)
                else:
                    if param.default == inspect.Parameter.empty:
                        fields[param_name] = (param.annotation, ...)
                    else:
                        fields[param_name] = (param.annotation, param.default)
            
            if fields:
                self._input_schema = create_model(
                    f"{self.config.name}Input",
                    **fields
                )
            
            # Generate output schema from return annotation
            if sig.return_annotation != inspect.Signature.empty:
                self._output_schema = create_model(
                    f"{self.config.name}Output",
                    value=(sig.return_annotation, ...)
                )
        
        except Exception as e:
            # Schema generation failed - validation will be skipped
            pass
    
    def _execute_impl(self, **inputs: Any) -> Any:
        """Execute Python function."""
        if not self._func:
            raise AdapterExecutionError(
                "Function not loaded",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
        
        try:
            if self.config.is_async:
                # Run async function
                return self._run_async_function(inputs)
            else:
                # Run sync function with timeout
                return self._run_with_timeout(inputs)
        
        except Exception as e:
            if isinstance(e, AdapterTimeoutError):
                raise
            
            raise AdapterExecutionError(
                f"Function execution failed: {e}",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
                context={"function": f"{self.config.module}.{self.config.function}"},
            )
    
    def _run_with_timeout(self, inputs: Dict[str, Any]) -> Any:
        """Run sync function with timeout."""
        import signal
        
        def timeout_handler(signum, frame):
            raise AdapterTimeoutError(
                f"Function execution exceeded timeout of {self.config.timeout}s",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
        
        # Set timeout (Unix only - Windows will skip timeout)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.config.timeout))
        
        try:
            result = self._func(**inputs)
            return result
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)  # Cancel timeout
    
    def _run_async_function(self, inputs: Dict[str, Any]) -> Any:
        """Run async function."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            coro = self._func(**inputs)
            return loop.run_until_complete(
                asyncio.wait_for(coro, timeout=self.config.timeout)
            )
        except asyncio.TimeoutError:
            raise AdapterTimeoutError(
                f"Async function execution exceeded timeout of {self.config.timeout}s",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
