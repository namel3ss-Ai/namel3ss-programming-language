"""Streaming execution for LLM connectors and chains."""

from textwrap import dedent

STREAMING = dedent(
    '''
async def stream_llm_connector(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Stream LLM connector responses token by token."""
    
    import time
    
    start_time = time.time()
    args = dict(payload or {})
    prompt_value = args.get("prompt") or args.get("input")
    prompt_text = _stringify_prompt_value(name, prompt_value) if prompt_value is not None else ""
    
    spec = AI_CONNECTORS.get(name, {})
    context_stub = {
        "env": {key: os.getenv(key) for key in ENV_KEYS},
        "vars": {},
        "app": APP,
    }
    config_raw = spec.get("config", {})
    config_resolved = _resolve_placeholders(config_raw, context_stub)
    if not isinstance(config_resolved, dict):
        config_resolved = config_raw if isinstance(config_raw, dict) else {}
    
    # Check if LLM instance exists in registry
    if '_LLM_INSTANCES' not in globals():
        yield {"error": "LLM registry not available", "status": "error"}
        return
    
    llm_instance = _LLM_INSTANCES.get(name)
    if llm_instance is None:
        yield {"error": f"LLM '{name}' not found in registry", "status": "error"}
        return
    
    try:
        from namel3ss.llm import ChatMessage
        
        # Build messages
        mode = str(config_resolved.get("mode") or "chat").lower()
        
        if mode == "chat":
            messages = []
            
            system_prompt = config_resolved.get("system") or config_resolved.get("system_prompt")
            if system_prompt:
                messages.append(ChatMessage(role="system", content=str(system_prompt)))
            
            extra_messages = config_resolved.get("messages")
            if isinstance(extra_messages, list):
                for msg in extra_messages:
                    if isinstance(msg, dict):
                        role = str(msg.get("role", "user"))
                        content = str(msg.get("content", ""))
                        messages.append(ChatMessage(role=role, content=content))
            
            user_role = str(config_resolved.get("user_role") or "user")
            messages.append(ChatMessage(role=user_role, content=prompt_text))
            
            # Stream from LLM
            chunk_count = 0
            async for chunk in llm_instance.stream_generate_chat(messages, **args):
                chunk_count += 1
                yield {
                    "chunk": chunk.text,
                    "delta": chunk.text,
                    "index": chunk_count,
                    "finish_reason": chunk.finish_reason,
                    "model": llm_instance.model,
                }
                
                if chunk.finish_reason:
                    break
        else:
            # Completion streaming
            chunk_count = 0
            async for chunk in llm_instance.stream_generate(prompt_text, **args):
                chunk_count += 1
                yield {
                    "chunk": chunk.text,
                    "delta": chunk.text,
                    "index": chunk_count,
                    "finish_reason": chunk.finish_reason,
                    "model": llm_instance.model,
                }
                
                if chunk.finish_reason:
                    break
        
        elapsed_ms = float(round((time.time() - start_time) * 1000.0, 3))
        yield {
            "status": "complete",
            "elapsed_ms": elapsed_ms,
            "chunks": chunk_count,
        }
        
    except Exception as exc:
        elapsed_ms = float(round((time.time() - start_time) * 1000.0, 3))
        yield {
            "error": f"{type(exc).__name__}: {exc}",
            "status": "error",
            "elapsed_ms": elapsed_ms,
        }


async def stream_chain(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Stream chain execution with step-by-step progress updates."""
    
    import time
    
    start_time = time.time()
    args = dict(payload or {})
    spec = AI_CHAINS.get(name)
    
    if not spec:
        yield {"error": f"Chain '{name}' not found", "status": "error"}
        return
    
    # Send initial event
    yield {
        "type": "chain_start",
        "chain": name,
        "inputs": args,
    }
    
    try:
        # Execute chain normally (we can enhance this later to stream individual steps)
        context = build_context(None)
        result = await run_chain(name, payload, context=context)
        
        # Send step events
        for step in result.get("steps", []):
            yield {
                "type": "step_complete",
                "step": step.get("step"),
                "name": step.get("name"),
                "status": step.get("status"),
                "output": step.get("output"),
            }
        
        # Send final event
        elapsed_ms = float(round((time.time() - start_time) * 1000.0, 3))
        yield {
            "type": "chain_complete",
            "status": result.get("status"),
            "result": result.get("result"),
            "elapsed_ms": elapsed_ms,
        }
        
    except Exception as exc:
        elapsed_ms = float(round((time.time() - start_time) * 1000.0, 3))
        yield {
            "type": "chain_error",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_ms": elapsed_ms,
        }
'''
).strip()

__all__ = ['STREAMING']
