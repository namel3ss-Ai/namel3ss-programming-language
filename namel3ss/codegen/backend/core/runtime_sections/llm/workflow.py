"""Workflow execution engine."""

from textwrap import dedent

WORKFLOW = dedent(
    '''
async def _execute_workflow_nodes(
    nodes: List[Dict[str, Any]],
    *,
    context: Dict[str, Any],
    chain_scope: Dict[str, Any],
    args: Dict[str, Any],
    working: Any,
    memory_state: Dict[str, Dict[str, Any]],
    allow_stubs: bool,
    steps_history: List[Dict[str, Any]],
    chain_name: str,
) -> Tuple[str, Any, Any]:
    """
    Execute workflow nodes with support for parallel execution.
    
    Detects groups of parallelizable steps and executes them concurrently
    when ENABLE_PARALLEL_STEPS environment variable is set.
    """
    enable_parallel = _is_truthy_env("ENABLE_PARALLEL_STEPS")
    
    if enable_parallel:
        # Detect parallel groups
        groups = _detect_parallel_groups(nodes or [])
        
        status = "partial"
        result_value: Any = None
        current_value = working
        encountered_error = False
        
        for group in groups:
            if len(group) == 1:
                # Single step - execute normally
                node = group[0]
                node_type = str(node.get("type") or "step").lower()
                
                if node_type == "step":
                    step_status, current_value, step_result, stop_on_error = await _execute_workflow_step(
                        node,
                        context=context,
                        chain_scope=chain_scope,
                        args=args,
                        working=current_value,
                        memory_state=memory_state,
                        allow_stubs=allow_stubs,
                        steps_history=steps_history,
                        chain_name=chain_name,
                    )
                    if step_status == "error":
                        encountered_error = True
                        if stop_on_error:
                            return "error", step_result, current_value
                    elif step_status == "ok":
                        status = "ok"
                        result_value = step_result
                    elif step_status == "stub" and status != "error" and result_value is None:
                        result_value = step_result
                elif node_type == "if":
                    branch_status, branch_result, branch_value = await _execute_workflow_if(
                        node,
                        context=context,
                        chain_scope=chain_scope,
                        args=args,
                        working=current_value,
                        memory_state=memory_state,
                        allow_stubs=allow_stubs,
                        steps_history=steps_history,
                        chain_name=chain_name,
                    )
                    if branch_status == "error":
                        return "error", branch_result, branch_value
                    if branch_status == "ok" and branch_result is not None:
                        status = "ok"
                        result_value = branch_result
                    current_value = branch_value
                elif node_type == "for":
                    loop_status, loop_result, loop_value = await _execute_workflow_for(
                        node,
                        context=context,
                        chain_scope=chain_scope,
                        args=args,
                        working=current_value,
                        memory_state=memory_state,
                        allow_stubs=allow_stubs,
                        steps_history=steps_history,
                        chain_name=chain_name,
                    )
                    if loop_status == "error":
                        return "error", loop_result, loop_value
                    if loop_status == "ok" and loop_result is not None:
                        status = "ok"
                        result_value = loop_result
                    current_value = loop_value
                elif node_type == "while":
                    while_status, while_result, while_value = await _execute_workflow_while(
                        node,
                        context=context,
                        chain_scope=chain_scope,
                        args=args,
                        working=current_value,
                        memory_state=memory_state,
                        allow_stubs=allow_stubs,
                        steps_history=steps_history,
                        chain_name=chain_name,
                    )
                    if while_status == "error":
                        return "error", while_result, while_value
                    if while_status == "ok" and while_result is not None:
                        status = "ok"
                        result_value = while_result
                    current_value = while_value
            else:
                # Multiple steps - execute in parallel
                logger.info(f"Executing {len(group)} steps in parallel for chain '{chain_name}'")
                parallel_status, parallel_result, parallel_value = await _execute_parallel_steps(
                    group,
                    context=context,
                    chain_scope=chain_scope,
                    args=args,
                    working=current_value,
                    memory_state=memory_state,
                    allow_stubs=allow_stubs,
                    steps_history=steps_history,
                    chain_name=chain_name,
                )
                if parallel_status == "error":
                    return "error", parallel_result, parallel_value
                if parallel_status == "ok" and parallel_result is not None:
                    status = "ok"
                    result_value = parallel_result
                current_value = parallel_value
        
        if encountered_error and status != "error":
            status = "error"
        return status, result_value, current_value
    
    # Sequential execution (default)
    status = "partial"
    result_value: Any = None
    current_value = working
    encountered_error = False
    for node in nodes or []:
        node_type = str(node.get("type") or "step").lower()
        if node_type == "step":
            step_status, current_value, step_result, stop_on_error = await _execute_workflow_step(
                node,
                context=context,
                chain_scope=chain_scope,
                args=args,
                working=current_value,
                memory_state=memory_state,
                allow_stubs=allow_stubs,
                steps_history=steps_history,
                chain_name=chain_name,
            )
            if step_status == "error":
                encountered_error = True
                if stop_on_error:
                    return "error", step_result, current_value
            elif step_status == "ok":
                status = "ok"
                result_value = step_result
            elif step_status == "stub" and status != "error" and result_value is None:
                result_value = step_result
        elif node_type == "if":
            branch_status, branch_result, branch_value = await _execute_workflow_if(
                node,
                context=context,
                chain_scope=chain_scope,
                args=args,
                working=current_value,
                memory_state=memory_state,
                allow_stubs=allow_stubs,
                steps_history=steps_history,
                chain_name=chain_name,
            )
            if branch_status == "error":
                return "error", branch_result, branch_value
            if branch_status == "ok" and branch_result is not None:
                status = "ok"
                result_value = branch_result
            current_value = branch_value
        elif node_type == "for":
            loop_status, loop_result, loop_value = await _execute_workflow_for(
                node,
                context=context,
                chain_scope=chain_scope,
                args=args,
                working=current_value,
                memory_state=memory_state,
                allow_stubs=allow_stubs,
                steps_history=steps_history,
                chain_name=chain_name,
            )
            if loop_status == "error":
                return "error", loop_result, loop_value
            if loop_status == "ok" and loop_result is not None:
                status = "ok"
                result_value = loop_result
            current_value = loop_value
        elif node_type == "while":
            while_status, while_result, while_value = await _execute_workflow_while(
                node,
                context=context,
                chain_scope=chain_scope,
                args=args,
                working=current_value,
                memory_state=memory_state,
                allow_stubs=allow_stubs,
                steps_history=steps_history,
                chain_name=chain_name,
            )
            if while_status == "error":
                return "error", while_result, while_value
            if while_status == "ok" and while_result is not None:
                status = "ok"
                result_value = while_result
            current_value = while_value
        else:
            _record_runtime_error(
                context,
                code="workflow.unsupported_node",
                message=f"Workflow node type '{node_type}' is not supported",
                scope=chain_name,
                source="workflow",
            )
            return "error", {"error": f"Unsupported workflow node '{node_type}'"}, current_value
    if encountered_error and status != "error":
        status = "error"
    return status, result_value, current_value


async def _execute_workflow_step(
    step: Dict[str, Any],
    *,
    context: Dict[str, Any],
    chain_scope: Dict[str, Any],
    args: Dict[str, Any],
    working: Any,
    memory_state: Dict[str, Dict[str, Any]],
    allow_stubs: bool,
    steps_history: List[Dict[str, Any]],
    chain_name: str,
) -> Tuple[str, Any, Any, bool]:
    chain_scope["counter"] = chain_scope.get("counter", 0) + 1
    step_label = step.get("name") or step.get("target") or f"step_{chain_scope['counter']}"
    kind = (step.get("kind") or "").lower()
    target = step.get("target") or ""
    options = step.get("options") or {}
    resolved_options = _resolve_placeholders(options, context)
    stop_on_error = bool(step.get("stop_on_error", True))
    read_memory = _extract_memory_names(resolved_options.pop("read_memory", None))
    write_memory = _extract_memory_names(resolved_options.pop("write_memory", None))
    memory_payload = _memory_snapshot(memory_state, read_memory) if read_memory else {}
    entry: Dict[str, Any] = {
        "step": len(steps_history) + 1,
        "kind": kind,
        "name": step_label,
        "inputs": None,
        "output": None,
        "status": "partial",
    }
    steps_history.append(entry)
    current_value = working
    step_result: Any = None
    if kind == "template":
        template = AI_TEMPLATES.get(target) or {}
        prompt = template.get("prompt", "")
        template_context = {
            "input": working,
            "vars": args,
            "payload": args,
            "memory": memory_payload,
            "steps": context.get("steps", {}),
            "loop": context.get("loop", {}),
            "locals": chain_scope.get("locals", {}),
        }
        entry["inputs"] = template_context
        try:
            rendered = _render_template_value(prompt, template_context)
            entry["output"] = rendered
            entry["status"] = "ok"
            current_value = rendered
            step_result = rendered
            if write_memory:
                _write_memory_entries(
                    memory_state,
                    write_memory,
                    rendered,
                    context=context,
                    source=f"template:{step_label}",
                )
        except Exception as exc:
            entry["output"] = {"error": str(exc)}
            entry["status"] = "error"
    elif kind == "connector":
        connector_payload = dict(args)
        connector_payload.setdefault("prompt", working)
        if memory_payload:
            connector_payload.setdefault("memory", {}).update(memory_payload)
        entry["inputs"] = connector_payload
        response = await call_llm_connector(target, connector_payload)
        entry["output"] = response
        entry["status"] = response.get("status", "partial")
        step_status = response.get("status")
        if step_status == "ok":
            current_value = response.get("result")
            step_result = current_value
        elif step_status == "error":
            step_result = response
        elif step_status == "stub":
            step_result = response
        if step_status == "ok" and write_memory:
            _write_memory_entries(
                memory_state,
                write_memory,
                response.get("result"),
                context=context,
                source=f"connector:{step_label}",
            )
    elif kind == "prompt":
        prompt_payload = working if isinstance(working, dict) else working
        if isinstance(prompt_payload, dict):
            payload_data = dict(prompt_payload)
        else:
            payload_data = {"value": prompt_payload}
        if memory_payload:
            payload_data.setdefault("memory", {}).update(memory_payload)
        if read_memory:
            payload_data["read_memory"] = read_memory
        if write_memory:
            payload_data["write_memory"] = write_memory
        response = await run_prompt(target, payload_data, context=context, memory=memory_payload)
        entry["inputs"] = response.get("inputs")
        entry["output"] = response
        entry["status"] = response.get("status", "partial")
        step_status = response.get("status")
        if step_status == "ok":
            current_value = response.get("output")
            step_result = current_value
        elif step_status == "error":
            step_result = response
        elif step_status == "stub":
            step_result = response
    elif kind == "tool":
        tools_registry = context.get("tools") if isinstance(context.get("tools"), dict) else {}
        payload = dict(resolved_options)
        if working is not None and "input" not in payload:
            payload["input"] = working
        if memory_payload:
            payload.setdefault("memory", {}).update(memory_payload)
        entry["inputs"] = payload
        plugin = tools_registry.get(target)
        if plugin is None:
            message = f"Tool '{target}' is not configured"
            entry["output"] = {"error": message}
            entry["status"] = "error"
            step_result = entry["output"]
            _record_runtime_error(
                context,
                code="tool_not_found",
                message=message,
                scope=f"chain:{chain_name}",
                detail=f"step:{step_label}",
            )
        else:
            try:
                _validate_tool_payload(plugin, payload, chain_name, step_label)
                tool_result = _call_tool_plugin(plugin, context, payload)
                entry["output"] = tool_result
                entry["status"] = "ok"
                current_value = tool_result
                step_result = tool_result
                if write_memory:
                    _write_memory_entries(
                        memory_state,
                        write_memory,
                        tool_result,
                        context=context,
                        source=f"tool:{step_label}",
                    )
            except Exception as exc:
                message = str(exc)
                entry["output"] = {"error": message}
                entry["status"] = "error"
                step_result = entry["output"]
                _record_runtime_error(
                    context,
                    code="tool_step_error",
                    message=f"Tool step '{step_label}' failed in chain '{chain_name}'",
                    scope=f"chain:{chain_name}",
                    detail=message,
                )
    elif kind == "python":
        module_name = resolved_options.get("module") or target or ""
        method_name = resolved_options.get("method") or "predict"
        python_args = dict(args)
        provided_args = resolved_options.get("arguments")
        if isinstance(provided_args, dict):
            python_args.update(provided_args)
        if memory_payload:
            python_args.setdefault("memory", {}).update(memory_payload)
        entry["inputs"] = python_args
        response = call_python_model(module_name, method_name, python_args)
        entry["output"] = response
        entry["status"] = response.get("status", "partial")
        step_status = response.get("status")
        if step_status == "ok":
            current_value = response.get("result")
            step_result = current_value
        elif step_status == "error":
            step_result = response
        elif step_status == "stub":
            step_result = response
        if step_status == "ok" and write_memory:
            _write_memory_entries(
                memory_state,
                write_memory,
                response.get("result"),
                context=context,
                source=f"python:{step_label}",
            )
    elif kind == "graph":
        # Execute multi-agent graph
        graphs_registry = context.get("graphs") if isinstance(context.get("graphs"), dict) else {}
        agents_registry = context.get("agents") if isinstance(context.get("agents"), dict) else {}
        
        # Get LLM and tool registries - these are the actual instantiated objects
        llms_registry = {}
        llm_reg = context.get("llms") if isinstance(context.get("llms"), dict) else {}
        for llm_name, llm_instance in llm_reg.items():
            llms_registry[llm_name] = llm_instance
        
        tools_registry = {}
        tool_reg = context.get("tools") if isinstance(context.get("tools"), dict) else {}
        for tool_name, tool_instance in tool_reg.items():
            tools_registry[tool_name] = tool_instance
        
        graph_input = working
        if isinstance(resolved_options.get("input"), str):
            graph_input = resolved_options["input"]
        elif resolved_options.get("input") is not None:
            graph_input = resolved_options["input"]
        
        graph_context = dict(resolved_options.get("context", {}))
        if memory_payload:
            graph_context.setdefault("memory", {}).update(memory_payload)
        
        entry["inputs"] = {"input": graph_input, "context": graph_context}
        
        try:
            from namel3ss.agents.factory import run_graph_from_state
            
            response = run_graph_from_state(
                target,
                graphs_registry,
                agents_registry,
                llms_registry,
                tools_registry,
                graph_input,
                context=graph_context,
            )
            entry["output"] = response
            entry["status"] = response.get("status", "partial")
            step_status = response.get("status")
            
            if step_status == "success":
                current_value = response.get("final_response")
                step_result = current_value
                if write_memory:
                    _write_memory_entries(
                        memory_state,
                        write_memory,
                        current_value,
                        context=context,
                        source=f"graph:{step_label}",
                    )
            elif step_status == "error":
                step_result = response
                entry["status"] = "error"
        except Exception as exc:
            message = str(exc)
            entry["output"] = {"error": message}
            entry["status"] = "error"
            step_result = entry["output"]
            _record_runtime_error(
                context,
                code="graph_step_error",
                message=f"Graph step '{step_label}' failed in chain '{chain_name}'",
                scope=f"chain:{chain_name}",
                detail=message,
            )
    else:
        entry["output"] = {"error": f"Unsupported step kind '{kind}'"}
        entry["status"] = "error"
        step_result = entry["output"]
    evaluation_cfg = step.get("evaluation")
    if evaluation_cfg:
        try:
            evaluation_results = _run_step_evaluations(
                evaluation_cfg,
                entry,
                current_value,
                context,
                chain_name,
                step_label,
            )
        except Exception as exc:
            entry["output"] = {"error": str(exc)}
            entry["status"] = "error"
            step_result = entry["output"]
            entry["result"] = current_value
            chain_scope.setdefault("steps", {})[step_label] = entry
            return entry.get("status", "partial"), current_value, step_result, stop_on_error
        guardrail_name = evaluation_cfg.get("guardrail")
        if guardrail_name:
            try:
                current_value = _apply_guardrail(
                    guardrail_name,
                    evaluation_results,
                    current_value,
                    entry,
                    chain_name,
                    step_label,
                    context,
                )
                step_result = current_value
            except Exception as exc:
                entry["output"] = {"error": str(exc)}
                entry["status"] = "error"
                step_result = entry["output"]
                entry["result"] = current_value
                chain_scope.setdefault("steps", {})[step_label] = entry
                return entry.get("status", "partial"), current_value, step_result, stop_on_error
    entry["result"] = current_value
    chain_scope.setdefault("steps", {})[step_label] = entry
    return entry.get("status", "partial"), current_value, step_result, stop_on_error


async def _execute_workflow_if(
    node: Dict[str, Any],
    *,
    context: Dict[str, Any],
    chain_scope: Dict[str, Any],
    args: Dict[str, Any],
    working: Any,
    memory_state: Dict[str, Dict[str, Any]],
    allow_stubs: bool,
    steps_history: List[Dict[str, Any]],
    chain_name: str,
) -> Tuple[str, Any, Any]:
    condition = node.get("condition")
    condition_source = node.get("condition_source")
    scope = _workflow_scope(chain_scope, args, working)
    branch_nodes = node.get("then") or []
    branch_selected = False
    if _evaluate_workflow_condition(condition, condition_source, scope, context, chain_name):
        branch_selected = True
    else:
        for branch in node.get("elif", []):
            if _evaluate_workflow_condition(
                branch.get("condition"),
                branch.get("condition_source"),
                scope,
                context,
                chain_name,
            ):
                branch_nodes = branch.get("steps") or []
                branch_selected = True
                break
        if not branch_selected:
            branch_nodes = node.get("else") or []
    if not branch_nodes:
        return "partial", None, working
    return await _execute_workflow_nodes(
        branch_nodes,
        context=context,
        chain_scope=chain_scope,
        args=args,
        working=working,
        memory_state=memory_state,
        allow_stubs=allow_stubs,
        steps_history=steps_history,
        chain_name=chain_name,
    )


async def _execute_workflow_for(
    node: Dict[str, Any],
    *,
    context: Dict[str, Any],
    chain_scope: Dict[str, Any],
    args: Dict[str, Any],
    working: Any,
    memory_state: Dict[str, Dict[str, Any]],
    allow_stubs: bool,
    steps_history: List[Dict[str, Any]],
    chain_name: str,
) -> Tuple[str, Any, Any]:
    iterable = _resolve_workflow_iterable(node, context, chain_scope, args, working, chain_name)
    if not iterable:
        return "partial", None, working
    loop_var = node.get("loop_var") or "item"
    max_iterations = node.get("max_iterations")
    iterations = 0
    result_value: Any = None
    current_value = working
    loop_context = context.setdefault("loop", {})
    scope_loop = chain_scope.setdefault("loop", {})
    for item in iterable:
        if max_iterations and iterations >= max_iterations:
            break
        chain_scope["locals"][loop_var] = item
        loop_context[loop_var] = item
        scope_loop[loop_var] = item
        branch_status, branch_result, branch_value = await _execute_workflow_nodes(
            node.get("body") or [],
            context=context,
            chain_scope=chain_scope,
            args=args,
            working=current_value,
            memory_state=memory_state,
            allow_stubs=allow_stubs,
            steps_history=steps_history,
            chain_name=chain_name,
        )
        if branch_status == "error":
            chain_scope["locals"].pop(loop_var, None)
            loop_context.pop(loop_var, None)
            return "error", branch_result, branch_value
        if branch_result is not None:
            result_value = branch_result
        current_value = branch_value
        iterations += 1
    chain_scope["locals"].pop(loop_var, None)
    loop_context.pop(loop_var, None)
    scope_loop.pop(loop_var, None)
    return ("ok" if iterations else "partial"), result_value, current_value


async def _execute_workflow_while(
    node: Dict[str, Any],
    *,
    context: Dict[str, Any],
    chain_scope: Dict[str, Any],
    args: Dict[str, Any],
    working: Any,
    memory_state: Dict[str, Dict[str, Any]],
    allow_stubs: bool,
    steps_history: List[Dict[str, Any]],
    chain_name: str,
) -> Tuple[str, Any, Any]:
    max_iterations = node.get("max_iterations") or 100
    iterations = 0
    result_value: Any = None
    current_value = working
    condition = node.get("condition")
    source = node.get("condition_source")
    while iterations < max_iterations:
        scope = _workflow_scope(chain_scope, args, current_value)
        if not _evaluate_workflow_condition(condition, source, scope, context, chain_name):
            break
        branch_status, branch_result, branch_value = await _execute_workflow_nodes(
            node.get("body") or [],
            context=context,
            chain_scope=chain_scope,
            args=args,
            working=current_value,
            memory_state=memory_state,
            allow_stubs=allow_stubs,
            steps_history=steps_history,
            chain_name=chain_name,
        )
        if branch_status == "error":
            return "error", branch_result, branch_value
        if branch_result is not None:
            result_value = branch_result
        current_value = branch_value
        iterations += 1
    if iterations >= max_iterations:
        _record_runtime_error(
            context,
            code="workflow.loop_limit",
            message=f"Workflow while loop in chain '{chain_name}' exceeded iteration limit",
            scope=chain_name,
            source="workflow",
        )
        return "error", {"error": "Loop iteration limit exceeded"}, current_value
    return ("ok" if iterations else "partial"), result_value, current_value


def _evaluate_workflow_condition(
    expression: Optional[Any],
    expression_source: Optional[str],
    scope: Dict[str, Any],
    context: Dict[str, Any],
    chain_name: str,
) -> bool:
    if expression is None and not expression_source:
        return False
    try:
        if expression is None:
            return bool(_evaluate_expression_tree(expression_source or "", scope, context))
        return bool(_evaluate_expression_tree(expression, scope, context))
    except Exception as exc:
        _record_runtime_error(
            context,
            code="workflow.condition_failed",
            message=f"Workflow condition in chain '{chain_name}' failed",
            scope=chain_name,
            source="workflow",
            detail=str(exc),
        )
    return False


def _resolve_workflow_iterable(
    node: Dict[str, Any],
    context: Dict[str, Any],
    chain_scope: Dict[str, Any],
    args: Dict[str, Any],
    working: Any,
    chain_name: str,
) -> List[Any]:
    source_kind = str(node.get("source_kind") or "expression").lower()
    if source_kind == "dataset":
        name = node.get("source_name")
        dataset_rows = context.get("datasets_data", {}).get(name)
        if not dataset_rows:
            _record_runtime_error(
                context,
                code="workflow.dataset_missing",
                message=f"Dataset '{name}' is not available for workflow loop",
                scope=chain_name,
                source="workflow",
            )
            return []
        return list(dataset_rows)
    expr_payload = node.get("source_expression")
    expr_source = node.get("source_expression_source")
    value = _evaluate_workflow_expression(
        expr_payload,
        expr_source,
        chain_scope,
        args,
        working,
        context,
        chain_name,
    )
    if value is None:
        return []
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        return list(value)
    try:
        return list(value)
    except Exception:
        return []


def _evaluate_workflow_expression(
    expression: Optional[Any],
    expression_source: Optional[str],
    chain_scope: Dict[str, Any],
    args: Dict[str, Any],
    working: Any,
    context: Dict[str, Any],
    chain_name: str,
) -> Any:
    if expression is None and not expression_source:
        return None
    scope = _workflow_scope(chain_scope, args, working)
    try:
        if expression is None:
            expr_text = expression_source or ""
            if not expr_text:
                return None
            return _evaluate_expression_tree(expr_text, scope, context)
        return _evaluate_expression_tree(expression, scope, context)
    except Exception as exc:
        _record_runtime_error(
            context,
            code="workflow.expression_failed",
            message=f"Workflow expression in chain '{chain_name}' failed",
            scope=chain_name,
            source="workflow",
            detail=str(exc),
        )
    return None


def _workflow_scope(chain_scope: Dict[str, Any], args: Dict[str, Any], working: Any) -> Dict[str, Any]:
    scope = {
        "input": chain_scope.get("input"),
        "steps": chain_scope.get("steps", {}),
        "locals": chain_scope.get("locals", {}),
        "loop": chain_scope.get("loop", {}),
        "payload": args,
        "value": working,
    }
    scope.update(chain_scope.get("locals", {}))
    return scope
'''
).strip()

__all__ = ['WORKFLOW']
