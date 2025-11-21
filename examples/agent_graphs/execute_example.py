#!/usr/bin/env python3
"""
Execute example agent graph workflows end-to-end.

This script demonstrates:
1. Loading graph JSON from file
2. Converting to N3 AST using EnhancedN3ASTConverter
3. Building RuntimeRegistry with real components
4. Executing the graph with GraphExecutor
5. Displaying results and telemetry

Usage:
    python examples/agent_graphs/execute_example.py --graph customer_support_triage
    python examples/agent_graphs/execute_example.py --graph research_pipeline --verbose
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from n3_server.converter.enhanced_converter import EnhancedN3ASTConverter
from n3_server.converter.models import GraphJSON
from n3_server.execution.registry import RuntimeRegistry
from n3_server.execution.executor import GraphExecutor, ExecutionContext
from namel3ss.llm.openai import OpenAILLM
from namel3ss.llm.registry import LLMRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def load_graph(graph_name: str) -> Dict[str, Any]:
    """Load graph JSON from file."""
    graph_file = project_root / "examples" / "agent_graphs" / f"{graph_name}.json"
    
    if not graph_file.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_file}")
    
    with open(graph_file) as f:
        return json.load(f)


async def setup_llm_registry() -> LLMRegistry:
    """Setup LLM registry with OpenAI models."""
    registry = LLMRegistry()
    
    # Register GPT-4
    gpt4 = OpenAILLM(
        name="gpt-4",
        model_name="gpt-4",
        api_key=None,  # Will use OPENAI_API_KEY env var
    )
    registry.register(gpt4)
    
    # Register GPT-4 Turbo
    gpt4_turbo = OpenAILLM(
        name="gpt-4-turbo",
        model_name="gpt-4-turbo-preview",
        api_key=None,
    )
    registry.register(gpt4_turbo)
    
    logger.info(f"Registered LLMs: {registry.list()}")
    return registry


async def execute_graph(
    graph_name: str,
    input_data: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Execute agent graph end-to-end.
    
    Args:
        graph_name: Name of graph JSON file (without .json)
        input_data: Input data for graph execution
        verbose: Enable verbose logging
    
    Returns:
        Execution result with output data and telemetry
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Step 1: Load graph JSON
    logger.info(f"Loading graph: {graph_name}")
    graph_data = await load_graph(graph_name)
    
    # Step 2: Validate and convert to AST
    logger.info("Converting graph to N3 AST...")
    converter = EnhancedN3ASTConverter()
    graph_json = GraphJSON.model_validate(graph_data)
    chain, conversion_context = converter.convert_graph_to_chain(graph_json)
    
    summary = converter.get_conversion_summary(conversion_context)
    logger.info(f"Conversion summary: {summary}")
    
    # Step 3: Setup LLM registry
    logger.info("Setting up LLM registry...")
    llm_registry = await setup_llm_registry()
    
    # Step 4: Build runtime registry
    logger.info("Building runtime registry...")
    runtime_registry = await RuntimeRegistry.from_conversion_context(
        conversion_context,
        llm_registry=llm_registry,
    )
    logger.info(f"Registry built: {len(runtime_registry.agents)} agents, "
                f"{len(runtime_registry.prompts)} prompts, "
                f"{len(runtime_registry.rag_pipelines)} RAG pipelines")
    
    # Step 5: Execute chain
    logger.info(f"Executing chain with input: {input_data}")
    executor = GraphExecutor(registry=runtime_registry)
    context = ExecutionContext()
    
    result = await executor.execute_chain(chain, input_data, context)
    
    # Step 6: Display results
    logger.info("=" * 80)
    logger.info("EXECUTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output: {json.dumps(result, indent=2)}")
    logger.info(f"Spans collected: {len(context.spans)}")
    
    # Display telemetry
    total_tokens = sum(
        span.attributes.tokens_prompt + span.attributes.tokens_completion
        for span in context.spans
        if span.attributes and span.attributes.tokens_prompt is not None
    )
    total_cost = sum(
        span.attributes.cost
        for span in context.spans
        if span.attributes and span.attributes.cost is not None
    )
    
    logger.info(f"Total tokens: {total_tokens}")
    logger.info(f"Total cost: ${total_cost:.4f}")
    
    if verbose:
        logger.debug("\nExecution Spans:")
        for span in context.spans:
            logger.debug(f"  [{span.type.value}] {span.name}: "
                        f"{span.duration_ms:.1f}ms ({span.status})")
    
    return {
        "output": result,
        "telemetry": {
            "spans": len(context.spans),
            "total_tokens": total_tokens,
            "total_cost": total_cost,
        }
    }


async def main():
    """Main execution entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Execute example agent graphs")
    parser.add_argument(
        "--graph",
        required=True,
        choices=["customer_support_triage", "research_pipeline"],
        help="Graph to execute"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Define example inputs for each graph
    example_inputs = {
        "customer_support_triage": {
            "ticket_text": "I can't log into my account and I have an important meeting in 30 minutes! "
                          "I've tried resetting my password twice but the reset email never arrives. "
                          "This is extremely urgent!",
            "customer_tier": "enterprise"
        },
        "research_pipeline": {
            "research_question": "What are the latest advances in large language model reasoning "
                                "and how do they compare to chain-of-thought prompting?"
        }
    }
    
    input_data = example_inputs[args.graph]
    
    try:
        result = await execute_graph(args.graph, input_data, args.verbose)
        logger.info("\n✅ Execution successful!")
        return 0
    except Exception as e:
        logger.error(f"\n❌ Execution failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
