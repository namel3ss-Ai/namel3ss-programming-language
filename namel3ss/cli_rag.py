"""CLI command for building RAG indices from datasets."""

import asyncio
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

from ..loader import load_program
from ..rag import (
    build_index,
    get_embedding_provider,
    get_vector_backend,
    get_dataset_loader,
    DatasetLoaderError,
    IndexStateManager,
)
from ..rag.backends import PgVectorBackend


def _parse_filters(filter_args: list) -> Dict[str, Any]:
    """
    Parse filter arguments from CLI.
    
    Args:
        filter_args: List of "key=value" strings
        
    Returns:
        Dictionary of filters
    """
    filters = {}
    for arg in filter_args:
        if "=" not in arg:
            print(f"Warning: Invalid filter format '{arg}', expected 'key=value'", file=sys.stderr)
            continue
        key, value = arg.split("=", 1)
        filters[key.strip()] = value.strip()
    return filters


class ProgressReporter:
    """Progress reporter for index building."""
    
    def __init__(self, verbose: bool = False, use_progress_bar: bool = True):
        self.verbose = verbose
        self.use_progress_bar = use_progress_bar
        self.start_time = time.time()
        self.last_update = self.start_time
        self.documents_processed = 0
        self.chunks_created = 0
        self.tokens_used = 0
        
        # Try to import tqdm for progress bars
        self.tqdm = None
        if use_progress_bar:
            try:
                from tqdm import tqdm
                self.tqdm = tqdm
            except ImportError:
                if verbose:
                    print("Note: Install tqdm for progress bars (pip install tqdm)", file=sys.stderr)
        
        self.pbar = None
    
    def start(self, total: Optional[int] = None):
        """Start progress reporting."""
        if self.tqdm and total:
            self.pbar = self.tqdm(total=total, desc="Indexing documents", unit="doc")
    
    def update(self, documents: int, chunks: int, tokens: int):
        """Update progress."""
        docs_delta = documents - self.documents_processed
        self.documents_processed = documents
        self.chunks_created = chunks
        self.tokens_used = tokens
        
        # Update progress bar
        if self.pbar:
            self.pbar.update(docs_delta)
        
        # Print periodic updates if verbose
        if self.verbose:
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                docs_per_sec = self.documents_processed / elapsed
                chunks_per_sec = self.chunks_created / elapsed
                
                # Update every 5 seconds
                if time.time() - self.last_update >= 5.0:
                    print(
                        f"  Progress: {self.documents_processed} docs, {self.chunks_created} chunks, "
                        f"{self.tokens_used} tokens "
                        f"({docs_per_sec:.1f} docs/s, {chunks_per_sec:.1f} chunks/s)",
                        file=sys.stderr,
                    )
                    self.last_update = time.time()
    
    def finish(self):
        """Finish progress reporting."""
        if self.pbar:
            self.pbar.close()
        
        elapsed = time.time() - self.start_time
        if elapsed > 0 and self.documents_processed > 0:
            docs_per_sec = self.documents_processed / elapsed
            chunks_per_sec = self.chunks_created / elapsed
            
            print(
                f"\n📊 Final: {self.documents_processed} docs, {self.chunks_created} chunks, "
                f"{self.tokens_used:,} tokens in {elapsed:.1f}s "
                f"({docs_per_sec:.1f} docs/s, {chunks_per_sec:.1f} chunks/s)"
            )


async def build_rag_index(
    source_file: Path,
    index_name: str,
    dataset_name: Optional[str] = None,
    verbose: bool = False,
    max_documents: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None,
    resume: bool = False,
    force_rebuild: bool = False,
) -> int:
    """
    Build a RAG index from a dataset.
    
    Args:
        source_file: Path to .ai file
        index_name: Name of the index to build
        dataset_name: Override source dataset (optional)
        verbose: Print detailed progress
        max_documents: Maximum number of documents to index
        filters: Metadata filters for document selection
        resume: Resume from previous run
        force_rebuild: Force rebuild (delete previous state)
    
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
            if max_documents:
                print(f"  Max documents: {max_documents}")
            if filters:
                print(f"  Filters: {filters}")
            if resume:
                print(f"  Mode: Resume from checkpoint")
            elif force_rebuild:
                print(f"  Mode: Force rebuild")
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
        
        # Initialize state manager
        state_manager = IndexStateManager()
        
        # Handle force rebuild
        if force_rebuild:
            if verbose:
                print("Deleting previous index state...")
            state_manager.delete_state(index_name, source_dataset)
            # Also clear the vector backend
            try:
                doc_count = await backend.count()
                if doc_count > 0 and verbose:
                    print(f"Warning: Vector backend still contains {doc_count} documents. Consider manually clearing it.")
            except:
                pass
        
        # Load or create index state
        index_state = None
        if resume and not force_rebuild:
            index_state = state_manager.load_state(index_name, source_dataset)
            if index_state:
                if index_state.completed:
                    print(f"Index '{index_name}' already completed. Use --force-rebuild to rebuild.", file=sys.stderr)
                    return 0
                if verbose:
                    print(f"Resuming from checkpoint: {index_state.total_documents} documents already processed")
            else:
                if verbose:
                    print("No previous state found, starting from beginning")
        
        if index_state is None:
            index_state = state_manager.create_state(
                index_name=index_name,
                dataset_name=source_dataset,
                metadata={
                    "embedding_model": index_def.embedding_model,
                    "chunk_size": index_def.chunk_size,
                    "overlap": index_def.overlap,
                },
            )
        
        # Create dataset loader
        try:
            loader = get_dataset_loader(dataset_obj, app)
        except DatasetLoaderError as e:
            print(f"Error creating dataset loader: {e}", file=sys.stderr)
            return 1
        
        if verbose:
            print(f"Dataset loader created for '{source_dataset}'")
        
        # Initialize progress reporter
        progress = ProgressReporter(verbose=verbose, use_progress_bar=True)
        progress.start(total=max_documents)
        
        # Create async iterator that wraps loader and handles state
        async def _iter_documents():
            doc_count = 0
            async for doc in loader.iter_documents(limit=max_documents, filters=filters):
                # Skip already-processed documents if resuming
                if resume and index_state.is_processed(doc["id"]):
                    continue
                
                yield doc
                doc_count += 1
        
        # Prepare progress callback
        def _progress_callback(docs, chunks, tokens):
            progress.update(docs + index_state.total_documents, chunks + index_state.total_chunks, tokens + index_state.total_tokens)
        
        # Build index with streaming
        if verbose:
            print(f"Loading and indexing documents...\n")
        
        result = await build_index(
            index_name=index_name,
            documents=_iter_documents(),
            embedding_model=index_def.embedding_model,
            vector_backend=backend,
            chunk_size=index_def.chunk_size,
            overlap=index_def.overlap,
            batch_size=32,
            progress_callback=_progress_callback,
        )
        
        # Update state
        index_state.total_documents += result.documents_processed
        index_state.total_chunks += result.chunks_created
        index_state.total_tokens += result.tokens_used
        index_state.mark_completed()
        state_manager.save_state(index_state)
        
        # Finish progress reporting
        progress.finish()
        
        # Print results
        print(f"\n✓ Index '{index_name}' built successfully!")
        print(f"  Documents processed: {result.documents_processed}")
        print(f"  Chunks created: {result.chunks_created}")
        print(f"  Chunks indexed: {result.chunks_indexed}")
        if result.tokens_used:
            print(f"  Embedding tokens: {result.tokens_used:,}")
        print(f"  Build time: {result.metadata.get('build_time_seconds', 0):.2f}s")
        
        if result.errors:
            print(f"\n⚠ Encountered {len(result.errors)} errors:")
            for error in result.errors[:5]:  # Show first 5
                print(f"  - {error}")
            if len(result.errors) > 5:
                print(f"  ... and {len(result.errors) - 5} more")
        
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
    
    # Parse filters
    filters = None
    if hasattr(args, "filter") and args.filter:
        filters = _parse_filters(args.filter)
    
    exit_code = asyncio.run(build_rag_index(
        source_file=source_file,
        index_name=args.index,
        dataset_name=getattr(args, "dataset", None),
        verbose=getattr(args, "verbose", False),
        max_documents=getattr(args, "max_documents", None),
        filters=filters,
        resume=getattr(args, "resume", False),
        force_rebuild=getattr(args, "force_rebuild", False),
    ))
    sys.exit(exit_code)

