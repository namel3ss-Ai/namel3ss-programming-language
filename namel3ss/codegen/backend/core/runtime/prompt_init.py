"""Generate prompt initialization code for runtime.py"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from namel3ss.codegen.backend.state import BackendState


def render_prompt_initialization_block(state: "BackendState") -> str:
    """
    Generate prompt initialization code for runtime.py
    
    Creates:
    - _PROMPT_INSTANCES dict
    - _initialize_prompt_instances() function that creates prompt instances from AI_PROMPTS
    
    Args:
        state: Backend state containing prompt definitions
        
    Returns:
        Python code string for prompt initialization
    """
    if not state.prompts:
        return ""
    
    lines = [
        "",
        "# Prompt initialization",
        "from namel3ss.prompts import create_prompt, get_registry as get_prompt_registry",
        "",
        "_PROMPT_INSTANCES: Dict[str, Any] = {}",
        "",
        "",
        "def _initialize_prompt_instances():",
        '    """Initialize prompt instances from AI_PROMPTS."""',
        "    for name, spec in AI_PROMPTS.items():",
        "        try:",
        "            prompt = create_prompt(",
        "                name=name,",
        "                template=spec['template'],",
        "                model=spec.get('model'),",
        "                args=spec.get('args', {}),",
        "                register=True",
        "            )",
        "            _PROMPT_INSTANCES[name] = prompt",
        "        except Exception as e:",
        "            logger.warning(f\"Failed to initialize prompt '{name}': {e}\")",
        "            continue",
        "",
        "",
        "_initialize_prompt_instances()",
        "",
    ]
    
    return "\n".join(lines)
