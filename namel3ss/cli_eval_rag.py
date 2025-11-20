"""CLI tool for evaluating RAG systems."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import click
import pandas as pd

from namel3ss.eval.rag_eval import RAGEvaluator
from namel3ss.multimodal import (
    MultimodalEmbeddingProvider,
    HybridRetriever,
    MultimodalConfig,
)
from namel3ss.multimodal.qdrant_backend import QdrantMultimodalBackend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Namel3ss RAG Evaluation CLI."""
    pass


@cli.command()
@click.argument('dataset_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), default='evaluation_results.md',
              help='Output file for results')
@click.option('--collection', default='multimodal_docs', help='Qdrant collection name')
@click.option('--qdrant-host', default='localhost', help='Qdrant host')
@click.option('--qdrant-port', default=6333, type=int, help='Qdrant port')
@click.option('--top-k', default=10, type=int, help='Number of documents to retrieve')
@click.option('--k-values', default='1,3,5,10', help='Comma-separated k values for metrics')
@click.option('--use-llm-judge', is_flag=True, help='Use LLM to judge answer quality')
@click.option('--llm-model', default='gpt-4', help='LLM model for judging')
@click.option('--llm-api-key', envvar='OPENAI_API_KEY', help='OpenAI API key')
@click.option('--csv-output', type=click.Path(), help='Optional CSV file for detailed results')
@click.option('--device', default='cpu', help='Device: cpu, cuda, or mps')
def eval_rag(
    dataset_path: str,
    output: str,
    collection: str,
    qdrant_host: str,
    qdrant_port: int,
    top_k: int,
    k_values: str,
    use_llm_judge: bool,
    llm_model: str,
    llm_api_key: Optional[str],
    csv_output: Optional[str],
    device: str,
):
    """
    Evaluate RAG system on a labeled dataset.
    
    Dataset format (JSON or CSV):
    [
        {
            "query": "What is machine learning?",
            "relevant_docs": ["doc1", "doc2"],
            "relevance_scores": {"doc1": 1.0, "doc2": 0.8},  # optional
            "ground_truth_answer": "ML is..."  # optional
        },
        ...
    ]
    """
    asyncio.run(_eval_rag_async(
        dataset_path=dataset_path,
        output=output,
        collection=collection,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        top_k=top_k,
        k_values=k_values,
        use_llm_judge=use_llm_judge,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        csv_output=csv_output,
        device=device,
    ))


async def _eval_rag_async(
    dataset_path: str,
    output: str,
    collection: str,
    qdrant_host: str,
    qdrant_port: int,
    top_k: int,
    k_values: str,
    use_llm_judge: bool,
    llm_model: str,
    llm_api_key: Optional[str],
    csv_output: Optional[str],
    device: str,
):
    """Async implementation of RAG evaluation."""
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = _load_dataset(dataset_path)
    logger.info(f"Loaded {len(dataset)} evaluation examples")
    
    # Parse k values
    k_list = [int(k.strip()) for k in k_values.split(',')]
    
    # Initialize components
    logger.info("Initializing RAG components...")
    
    # Embedding provider
    embedding_provider = MultimodalEmbeddingProvider(device=device)
    await embedding_provider.initialize()
    
    # Vector backend
    qdrant_config = {
        "host": qdrant_host,
        "port": qdrant_port,
        "collection_name": collection,
        "text_dimension": 384,
        "image_dimension": 512,
        "audio_dimension": 384,
    }
    vector_backend = QdrantMultimodalBackend(qdrant_config)
    await vector_backend.initialize()
    
    # Retriever
    retriever = HybridRetriever(
        vector_backend=vector_backend,
        embedding_provider=embedding_provider,
        enable_sparse=True,
        enable_reranking=True,
        device=device,
    )
    await retriever.initialize()
    
    # Evaluator
    evaluator = RAGEvaluator(
        k_values=k_list,
        use_llm_judge=use_llm_judge,
        llm_judge_model=llm_model,
        llm_api_key=llm_api_key,
    )
    
    # Define retriever function
    async def retriever_fn(query: str):
        result = await retriever.search(query, top_k=top_k)
        return result.documents
    
    # Run evaluation
    logger.info("Running evaluation...")
    aggregated = await evaluator.evaluate_dataset(
        eval_examples=dataset,
        retriever_fn=retriever_fn,
    )
    
    # Format and save results
    markdown_report = evaluator.format_results(aggregated)
    
    Path(output).write_text(markdown_report)
    logger.info(f"Saved markdown report to {output}")
    
    # Optionally save CSV
    if csv_output:
        _save_csv_results(aggregated, csv_output)
        logger.info(f"Saved CSV results to {csv_output}")
    
    # Print summary to console
    click.echo("\n" + "="*60)
    click.echo("EVALUATION RESULTS")
    click.echo("="*60)
    click.echo(markdown_report)
    click.echo("="*60)


def _load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load evaluation dataset from JSON or CSV."""
    path_obj = Path(path)
    
    if path_obj.suffix == '.json':
        with open(path) as f:
            return json.load(f)
    elif path_obj.suffix == '.csv':
        df = pd.read_csv(path)
        
        # Convert to list of dicts
        dataset = []
        for _, row in df.iterrows():
            item = {
                "query": row["query"],
                "relevant_docs": json.loads(row["relevant_docs"]),
            }
            
            if "relevance_scores" in row and pd.notna(row["relevance_scores"]):
                item["relevance_scores"] = json.loads(row["relevance_scores"])
            
            if "ground_truth_answer" in row and pd.notna(row["ground_truth_answer"]):
                item["ground_truth_answer"] = row["ground_truth_answer"]
            
            dataset.append(item)
        
        return dataset
    else:
        raise ValueError(f"Unsupported file format: {path_obj.suffix}")


def _save_csv_results(aggregated: Dict[str, Any], output_path: str):
    """Save detailed results to CSV."""
    rows = []
    
    # Retrieval metrics
    for k in sorted(aggregated["precision_at_k"].keys()):
        rows.append({
            "metric": f"precision@{k}",
            "value": aggregated["precision_at_k"][k],
        })
        rows.append({
            "metric": f"recall@{k}",
            "value": aggregated["recall_at_k"][k],
        })
        rows.append({
            "metric": f"ndcg@{k}",
            "value": aggregated["ndcg_at_k"][k],
        })
    
    rows.append({"metric": "hit_rate", "value": aggregated["hit_rate"]})
    rows.append({"metric": "mrr", "value": aggregated["mrr"]})
    
    # Generation metrics
    if "faithfulness" in aggregated:
        rows.append({"metric": "faithfulness", "value": aggregated["faithfulness"]})
        rows.append({"metric": "relevance", "value": aggregated["relevance"]})
        if "correctness" in aggregated:
            rows.append({"metric": "correctness", "value": aggregated["correctness"]})
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


@cli.command()
@click.argument('queries_file', type=click.Path(exists=True))
@click.option('--collection', default='multimodal_docs', help='Qdrant collection name')
@click.option('--qdrant-host', default='localhost', help='Qdrant host')
@click.option('--qdrant-port', default=6333, type=int, help='Qdrant port')
@click.option('--top-k', default=10, type=int, help='Number of documents to retrieve')
@click.option('--device', default='cpu', help='Device: cpu, cuda, or mps')
def batch_search(
    queries_file: str,
    collection: str,
    qdrant_host: str,
    qdrant_port: int,
    top_k: int,
    device: str,
):
    """
    Perform batch search on multiple queries.
    
    Input file format (one query per line or JSON array).
    """
    asyncio.run(_batch_search_async(
        queries_file=queries_file,
        collection=collection,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        top_k=top_k,
        device=device,
    ))


async def _batch_search_async(
    queries_file: str,
    collection: str,
    qdrant_host: str,
    qdrant_port: int,
    top_k: int,
    device: str,
):
    """Async implementation of batch search."""
    # Load queries
    path = Path(queries_file)
    if path.suffix == '.json':
        with open(path) as f:
            queries = json.load(f)
    else:
        queries = path.read_text().strip().split('\n')
    
    logger.info(f"Loaded {len(queries)} queries")
    
    # Initialize components
    embedding_provider = MultimodalEmbeddingProvider(device=device)
    await embedding_provider.initialize()
    
    qdrant_config = {
        "host": qdrant_host,
        "port": qdrant_port,
        "collection_name": collection,
    }
    vector_backend = QdrantMultimodalBackend(qdrant_config)
    await vector_backend.initialize()
    
    retriever = HybridRetriever(
        vector_backend=vector_backend,
        embedding_provider=embedding_provider,
        enable_sparse=True,
        enable_reranking=True,
        device=device,
    )
    await retriever.initialize()
    
    # Process queries
    results = []
    for query in queries:
        logger.info(f"Searching: {query}")
        result = await retriever.search(query, top_k=top_k)
        results.append({
            "query": query,
            "results": result.documents,
            "scores": result.scores,
        })
    
    # Output results
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    cli()
