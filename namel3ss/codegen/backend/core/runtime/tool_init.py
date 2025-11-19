"""Generate tool initialization code for runtime.py"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from namel3ss.codegen.backend.state import BackendState


def render_tool_initialization_block(state: "BackendState") -> str:
    """
    Generate tool initialization code for runtime.py
    
    Creates:
    - _TOOL_INSTANCES dict
    - _initialize_tool_instances() function that creates tool instances from TOOL_REGISTRY
    
    Args:
        state: Backend state containing tool definitions
        
    Returns:
        Python code string for tool initialization
    """
    if not state.tools:
        return ""
    
    lines = [
        "",
        "# Tool initialization",
        "from namel3ss.tools import create_tool, get_registry as get_tool_registry",
        "",
        "_TOOL_INSTANCES: Dict[str, Any] = {}",
        "",
        "",
        "def _initialize_tool_instances():",
        '    """Initialize tool instances from TOOL_REGISTRY."""',
        "    for name, spec in TOOL_REGISTRY.items():",
        "        try:",
        "            tool = create_tool(",
        "                name=name,",
        "                tool_type=spec['type'],",
        "                endpoint=spec.get('endpoint'),",
        "                method=spec.get('method', 'POST'),",
        "                input_schema=spec.get('input_schema', {}),",
        "                output_schema=spec.get('output_schema', {}),",
        "                headers=spec.get('headers', {}),",
        "                timeout=spec.get('timeout', 30.0),",
        "                register=True,",
        "                **spec.get('config', {})",
        "            )",
        "            _TOOL_INSTANCES[name] = tool",
        "        except Exception as e:",
        "            logger.warning(f\"Failed to initialize tool '{name}': {e}\")",
        "            continue",
        "",
        "",
        "_initialize_tool_instances()",
        "",
    ]
    
    return "\n".join(lines)
