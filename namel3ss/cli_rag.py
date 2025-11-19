"""CLI command for building RAG indices from datasets."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

from ..loader import load_program
from ..rag import build_index, get_embedding_provider, get_vector_backend
from ..rag.backends import PgVectorBackend


async def build_rag_index(
    source_file: Path,
    index_name: str,
    dataset_name: Optional[str] = None,
    verbose: bool = False,
) -> int:
    """
    Build a RAG index from a dataset.
    
    Args:
        source_file: Path to .n3 file
        index_name: Name of the index to build
        dataset_name: Override source dataset (optional)
        verbose: Print detailed progress
    
    Returns:
        Exit code (0 for success)
    """
    try:
        # Load program
        if verbose:
            print(f"Loading program from {source_file}...")
        
        program = load_program(source_file)
        if not program.body or not hasattr(program.body[0], 'indices'):
            print(f"Error: No app found in {source_file}", file=sys.stderr)
            return 1
        
        app = program.body[0]
        
        # Find index definition
        index_def = None
        for idx in app.indices:
            if idx.name == index_name:
                index_def = idx
                break
        
        if not index_def:
            print(f"Error: Index '{index_name}' not found in app", file=sys.stderr)
            available = [idx.name for idx in app.indices]
            if available:
                print(f"Available indices: {', '.join(available)}", file=sys.stderr)
            return 1
        
        # Override dataset if specified
        source_dataset = dataset_name or index_def.source_dataset
        
        # Find dataset
        dataset_obj = None
        for ds in app.datasets:
            if ds.name == source_dataset:
                dataset_obj = ds
                break
        
        if not dataset_obj:
            print(f"Error: Dataset '{source_dataset}' not found in app", file=sys.stderr)
            return 1
        
        if verbose:
            print(f"\nBuilding index '{index_name}':")
            print(f"  Source dataset: {source_dataset}")
            print(f"  Embedding model: {index_def.embedding_model}")
            print(f"  Chunk size: {index_def.chunk_size}")
            print(f"  Overlap: {index_def.overlap}")
            print(f"  Backend: {index_def.backend}")
            print()
        
        # Initialize vector backend
        backend_config = dict(index_def.config) if index_def.config else {}
        if index_def.table_name:
            backend_config["table_name"] = index_def.table_name
        if index_def.namespace:
            backend_config["namespace"] = index_def.namespace
        if index_def.collection:
            backend_config["collection"] = index_def.collection
        
        backend = get_vector_backend(index_def.backend, backend_config)
        await backend.initialize()
        
        if verbose:
            print("Vector backend initialized")
        
        # TODO: In production, load actual documents from the dataset
        # For now, this is a placeholder that shows the structure
        print("Warning: Document loading from datasets not yet implemented", file=sys.stderr)
        print("This command shows the structure for index building", file=sys.stderr)
        
        # Example structure for when dataset loading is implemented:
        # documents = await load_dataset_documents(dataset_obj)
        documents = [
            {
                "id": "doc_1",
                "content": "Example document content. This would come from your dataset.",
                "metadata": {"source": source_dataset}
            }
        ]
        
        # Build index
        if verbose:
            print(f"\nProcessing {len(documents)} documents...")
        
        result = await build_index(
            index_name=index_name,
            documents=documents,
            embedding_model=index_def.embedding_model,
            vector_backend=backend,
            chunk_size=index_def.chunk_size,
            overlap=index_def.overlap,
            batch_size=32,
        )
        
        # Print results
        print(f"\n✓ Index '{index_name}' built successfully!")
        print(f"  Documents processed: {result.documents_processed}")
        print(f"  Chunks created: {result.chunks_created}")
        print(f"  Chunks indexed: {result.chunks_indexed}")
        if result.tokens_used:
            print(f"  Embedding tokens: {result.tokens_used}")
        print(f"  Build time: {result.metadata.get('build_time_seconds', 0):.2f}s")
        
        if result.errors:
            print(f"\n⚠ Encountered {len(result.errors)} errors:")
            for error in result.errors[:5]:  # Show first 5
                print(f"  - {error}")
        
        await backend.close()
        return 0
        
    except Exception as e:
        print(f"Error building index: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def cmd_build_index(args) -> None:
    """CLI command handler for building RAG indices."""
    source_file = Path(args.source).resolve()
    if not source_file.exists():
        print(f"Error: File not found: {source_file}", file=sys.stderr)
        sys.exit(1)
    
    exit_code = asyncio.run(build_rag_index(
        source_file=source_file,
        index_name=args.index,
        dataset_name=args.dataset,
        verbose=args.verbose,
    ))
    sys.exit(exit_code)
