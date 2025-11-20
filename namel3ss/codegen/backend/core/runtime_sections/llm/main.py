"""Main entry points."""

from textwrap import dedent

MAIN = dedent(
    '''
def run_chain(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a configured AI chain and return detailed step results."""

    start_time = time.time()
    args = dict(payload or {})
    allow_stubs = _is_truthy_env("NAMEL3SS_ALLOW_STUBS")
    spec = AI_CHAINS.get(name)
    runtime_context = context if isinstance(context, dict) else build_context(None)

    if not spec:
        response: Dict[str, Any] = {
            "status": "not_found",
            "result": None,
            "steps": [],
            "inputs": args,
            "metadata": {"elapsed_ms": float(round((time.time() - start_time) * 1000.0, 3))},
        }
        if not allow_stubs:
            response["error"] = f"Chain '{name}' is not defined"
        return response

    input_key = spec.get("input_key", "input")
    working: Any = args.get(input_key, args)
    memory_state = _ensure_memory_state(runtime_context)
    chain_scope: Dict[str, Any] = {
        "input": working,
        "steps": {},
        "locals": {},
        "loop": {},
        "counter": 0,
    }
    runtime_context["steps"] = chain_scope["steps"]
    runtime_context.setdefault("chain", {})["steps"] = chain_scope["steps"]
    runtime_context["payload"] = args

    steps_history: List[Dict[str, Any]] = []
    status, result_value, working = _execute_workflow_nodes(
        spec.get("steps", []),
        context=runtime_context,
        chain_scope=chain_scope,
        args=args,
        working=working,
        memory_state=memory_state,
        allow_stubs=allow_stubs,
        steps_history=steps_history,
        chain_name=name,
    )

    elapsed_ms = float(round((time.time() - start_time) * 1000.0, 3))
    if status != "error" and result_value is None:
        status = "partial"

    return {
        "status": status if steps_history else "partial",
        "result": result_value,
        "steps": steps_history,
        "inputs": args,
        "metadata": {"elapsed_ms": elapsed_ms},
    }
'''
).strip()

__all__ = ['MAIN']
