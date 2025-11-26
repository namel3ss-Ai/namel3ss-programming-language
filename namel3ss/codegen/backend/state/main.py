"""Main orchestration for building backend state from AST."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from ....effects import EffectAnalyzer
from ....frames import FrameExpressionAnalyzer, FrameTypeError
from .agents import _encode_agent, _encode_graph
from .ai import (
    _encode_ai_connector,
    _encode_ai_model,
    _encode_chain,
    _encode_index,
    _encode_llm,
    _encode_memory,
    _encode_planner,
    _encode_planning_workflow,
    _encode_prompt,
    _encode_rag_pipeline,
    _encode_template,
    _encode_tool,
)
from .classes import BackendState, PageSpec
from .crud import _encode_crud_resource
from .datasets import _encode_dataset
from .evaluation import _encode_eval_suite, _encode_evaluator, _encode_guardrail, _encode_metric
from .experiments import _encode_experiment
from .expressions import set_frame_analyzer
from .frames import _encode_frame
from .insights import _encode_insight
from .logic import _encode_knowledge_module, _encode_logic_query
from .models import _encode_model
from .pages import _encode_page
from .training import _encode_training_job, _encode_tuning_job
from .variables import _encode_variable

if TYPE_CHECKING:
    from ....ast import App, Prompt


def build_backend_state(app: "App") -> BackendState:
    """Build the serializable backend state for the provided :class:`App`."""
    env_keys: Set[str] = set()
    
    # Set up frame analyzer for expression encoding
    frame_analyzer = FrameExpressionAnalyzer(app.frames) if app.frames else None
    set_frame_analyzer(frame_analyzer)
    
    try:
        # Analyze effects
        EffectAnalyzer(app).analyze()
        
        # Encode datasets
        datasets: Dict[str, Dict[str, Any]] = {}
        connectors: Dict[str, Dict[str, Any]] = {}
        for dataset in app.datasets:
            encoded = _encode_dataset(dataset, env_keys)
            datasets[dataset.name] = encoded
            if encoded.get("connector"):
                connectors[dataset.name] = encoded["connector"]
        
        # Encode frames
        frames: Dict[str, Dict[str, Any]] = {}
        for frame in app.frames:
            frames[frame.name] = _encode_frame(frame, env_keys)
        
        # Encode AI connectors
        ai_connectors: Dict[str, Dict[str, Any]] = {}
        for connector in app.connectors:
            aiconfig = _encode_ai_connector(connector, env_keys)
            aiconfig.pop("name", None)
            ai_connectors[connector.name] = aiconfig
        
        # Encode AI models
        ai_models: Dict[str, Dict[str, Any]] = {}
        for model in app.ai_models:
            ai_models[model.name] = _encode_ai_model(model, env_keys)
        
        # Encode first-class LLM blocks
        llms: Dict[str, Dict[str, Any]] = {}
        for llm in app.llms:
            llms[llm.name] = _encode_llm(llm, env_keys)
        
        # Encode first-class Tool blocks
        tools: Dict[str, Dict[str, Any]] = {}
        for tool in app.tools:
            tools[tool.name] = _encode_tool(tool, env_keys)
        
        # Encode RAG indices
        indices: Dict[str, Dict[str, Any]] = {}
        for index in app.indices:
            indices[index.name] = _encode_index(index, env_keys)
        
        # Encode RAG pipelines
        rag_pipelines: Dict[str, Dict[str, Any]] = {}
        for pipeline in app.rag_pipelines:
            rag_pipelines[pipeline.name] = _encode_rag_pipeline(pipeline, env_keys)
        
        # Encode memories
        memories: Dict[str, Dict[str, Any]] = {}
        for memory in app.memories:
            memories[memory.name] = _encode_memory(memory, env_keys)
        memory_names = set(memories.keys())
        
        # Encode prompts
        prompt_lookup: Dict[str, "Prompt"] = {}
        prompts: Dict[str, Dict[str, Any]] = {}
        for prompt in app.prompts:
            model_name = prompt.model
            # Allow prompts without models or with models that reference either ai_models or llms
            if model_name and model_name not in ai_models and model_name not in llms:
                raise ValueError(f"Prompt '{prompt.name}' references undefined model '{model_name}'")
            prompts[prompt.name] = _encode_prompt(prompt, env_keys)
            prompt_lookup[prompt.name] = prompt
        
        # Encode insights
        insights: Dict[str, Dict[str, Any]] = {}
        for insight in app.insights:
            insights[insight.name] = _encode_insight(insight, env_keys)
        
        # Encode ML models
        models: Dict[str, Dict[str, Any]] = {}
        for model in app.models:
            models[model.name] = _encode_model(model, env_keys)
        
        # Encode templates
        templates: Dict[str, Dict[str, Any]] = {}
        for template in app.templates:
            templates[template.name] = _encode_template(template, env_keys)
        
        # Encode chains
        chains: Dict[str, Dict[str, Any]] = {}
        for chain in app.chains:
            chains[chain.name] = _encode_chain(chain, env_keys, memory_names)

        # Encode planners
        planners: Dict[str, Dict[str, Any]] = {}
        planning_workflows: Dict[str, Dict[str, Any]] = {}
        
        # Check if app has planners and planning workflows
        if hasattr(app, "planners"):
            for planner in app.planners:
                planners[planner.name] = _encode_planner(planner, env_keys)
        
        if hasattr(app, "planning_workflows"):
            for workflow in app.planning_workflows:
                planning_workflows[workflow.name] = _encode_planning_workflow(workflow, env_keys)
        
        # Encode agents
        agents: Dict[str, Dict[str, Any]] = {}
        for agent in app.agents:
            agents[agent.name] = _encode_agent(agent, env_keys)
        
        # Encode multi-agent graphs
        graphs: Dict[str, Dict[str, Any]] = {}
        for graph in app.graphs:
            graphs[graph.name] = _encode_graph(graph, env_keys)
        
        # Encode experiments
        experiments: Dict[str, Dict[str, Any]] = {}
        for experiment in app.experiments:
            experiments[experiment.name] = _encode_experiment(experiment, env_keys)
        
        # Encode training jobs
        training_jobs: Dict[str, Dict[str, Any]] = {}
        for job in app.training_jobs:
            training_jobs[job.name] = _encode_training_job(job, env_keys)
        
        # Encode tuning jobs
        tuning_jobs: Dict[str, Dict[str, Any]] = {}
        for job in app.tuning_jobs:
            tuning_jobs[job.name] = _encode_tuning_job(job, env_keys)
        
        # Encode CRUD resources
        crud_resources: Dict[str, Dict[str, Any]] = {}
        for resource in app.crud_resources:
            crud_resources[resource.name] = _encode_crud_resource(resource, env_keys)
        
        # Encode evaluators
        evaluators: Dict[str, Dict[str, Any]] = {}
        for evaluator in app.evaluators:
            evaluators[evaluator.name] = _encode_evaluator(evaluator, env_keys)
        
        # Encode metrics
        metrics: Dict[str, Dict[str, Any]] = {}
        for metric in app.metrics:
            metrics[metric.name] = _encode_metric(metric)
        
        # Encode guardrails
        guardrails: Dict[str, Dict[str, Any]] = {}
        for guardrail in app.guardrails:
            guardrails[guardrail.name] = _encode_guardrail(guardrail)
        
        # Encode eval suites
        eval_suites: Dict[str, Dict[str, Any]] = {}
        for suite in app.eval_suites:
            eval_suites[suite.name] = _encode_eval_suite(suite)
        
        # Encode logic queries
        queries: Dict[str, Dict[str, Any]] = {}
        for query in app.queries:
            queries[query.name] = _encode_logic_query(query, env_keys)
        
        # Encode knowledge modules
        knowledge_modules: Dict[str, Dict[str, Any]] = {}
        for module in app.knowledge_modules:
            knowledge_modules[module.name] = _encode_knowledge_module(module, env_keys)
        
        # Encode pages
        pages: List[PageSpec] = []
        for index, page in enumerate(app.pages):
            page_spec = _encode_page(page, env_keys, prompt_lookup)
            page_spec.index = index  # Set the page index
            pages.append(page_spec)
        
        # Build app payload
        app_payload: Dict[str, Any] = {
            "name": app.name,
            "database": app.database,
            "theme": dict(app.theme.values),
            "variables": [_encode_variable(var, env_keys) for var in app.variables],
        }
        
        sorted_env_keys = sorted(env_keys)
        
        state = BackendState(
            app=app_payload,
            datasets=datasets,
            frames=frames,
            connectors=connectors,
            ai_connectors=ai_connectors,
            memories=memories,
            ai_models=ai_models,
            llms=llms,
            tools=tools,
            indices=indices,
            rag_pipelines=rag_pipelines,
            prompts=prompts,
            insights=insights,
            models=models,
            templates=templates,
            chains=chains,
            planners=planners,
            planning_workflows=planning_workflows,
            agents=agents,
            graphs=graphs,
            experiments=experiments,
            training_jobs=training_jobs,
            tuning_jobs=tuning_jobs,
            crud_resources=crud_resources,
            evaluators=evaluators,
            metrics=metrics,
            guardrails=guardrails,
            eval_suites=eval_suites,
            queries=queries,
            knowledge_modules=knowledge_modules,
            pages=pages,
            env_keys=sorted_env_keys,
        )
        
        # Store original app for IR builder access to AST
        state._original_app = app
        
        return state
    finally:
        # Clean up frame analyzer
        set_frame_analyzer(None)
