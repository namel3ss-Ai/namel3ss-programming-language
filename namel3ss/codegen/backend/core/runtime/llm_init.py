"""LLM runtime initialization helpers."""

from __future__ import annotations

from typing import List

from namel3ss.codegen.backend.state import BackendState


def render_llm_initialization_block(state: BackendState) -> str:
    """
    Generate code that instantiates LLM instances from LLM_REGISTRY.
    
    This creates actual BaseLLM instances that can be used in chain execution.
    """
    if not state.llms:
        return ""
    
    lines: List[str] = []
    
    # Add import
    lines.append("# LLM Instance Initialization")
    lines.append("from namel3ss.llm import create_llm, get_registry as get_llm_registry")
    lines.append("")
    
    # Create instance dictionary
    lines.append("# Dictionary to store initialized LLM instances")
    lines.append("_LLM_INSTANCES: Dict[str, Any] = {}")
    lines.append("")
    
    # Add initialization function
    lines.append("def _initialize_llm_instances() -> None:")
    lines.append("    \"\"\"Initialize LLM instances from LLM_REGISTRY.\"\"\"")
    lines.append("    global _LLM_INSTANCES")
    lines.append("    llm_registry = get_llm_registry()")
    lines.append("    llm_registry.clear()  # Clear any previous instances")
    lines.append("")
    lines.append("    for name, spec in LLM_REGISTRY.items():")
    lines.append("        try:")
    lines.append("            provider = spec.get('provider', '')")
    lines.append("            model = spec.get('model', '')")
    lines.append("            config = spec.get('config', {})")
    lines.append("")
    lines.append("            if not provider or not model:")
    lines.append("                logger.warning(f\"Skipping LLM '{name}': missing provider or model\")")
    lines.append("                continue")
    lines.append("")
    lines.append("            # Create LLM instance and register it")
    lines.append("            llm = create_llm(name, provider, model, config, register=True)")
    lines.append("            _LLM_INSTANCES[name] = llm")
    lines.append("            logger.info(f\"Initialized LLM '{name}' ({provider}/{model})\")")
    lines.append("        except Exception as exc:")
    lines.append("            logger.error(f\"Failed to initialize LLM '{name}': {exc}\")")
    lines.append("            # Continue with other LLMs even if one fails")
    lines.append("")
    lines.append("")
    lines.append("# Initialize LLM instances at module load time")
    lines.append("_initialize_llm_instances()")
    lines.append("")
    
    return "\n".join(lines)


__all__ = ["render_llm_initialization_block"]
