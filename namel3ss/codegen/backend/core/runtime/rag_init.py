"""Generate RAG initialization code for multimodal pipelines."""

from __future__ import annotations

import textwrap
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ....codegen.backend.state.classes import BackendState

__all__ = ["generate_rag_initialization"]


def generate_rag_initialization(state: "BackendState") -> str:
    """Generate initialization code for RAG indices and pipelines.
    
    Generates code to initialize:
    - Multimodal ingesters (if extract_images or extract_audio enabled)
    - Embedding providers (text, image, audio)
    - Vector database backends (Qdrant with multi-vector support)
    - Hybrid retrievers (dense + sparse + reranking)
    
    Parameters
    ----------
    state : BackendState
        Compiled backend state containing indices and RAG pipelines
        
    Returns
    -------
    str
        Python code for initializing RAG components
    """
    sections = []
    
    # Check if we need multimodal support
    has_multimodal = False
    has_hybrid = False
    
    for index_data in state.indices.values():
        if index_data.get("extract_images") or index_data.get("extract_audio"):
            has_multimodal = True
    
    for pipeline_data in state.rag_pipelines.values():
        if pipeline_data.get("enable_hybrid"):
            has_hybrid = True
    
    # Generate imports
    imports = _generate_imports(has_multimodal, has_hybrid)
    sections.append(imports)
    
    # Generate ingester initialization
    if has_multimodal:
        sections.append(_generate_ingester_init())
    
    # Generate embedding provider initialization
    sections.append(_generate_embedding_providers(state))
    
    # Generate vector backend initialization
    sections.append(_generate_vector_backends(state))
    
    # Generate retriever initialization
    if has_hybrid:
        sections.append(_generate_hybrid_retrievers(state))
    
    # Generate RAG pipeline factory
    sections.append(_generate_rag_pipeline_factory(state, has_multimodal, has_hybrid))
    
    return "\n\n".join(sections)


def _generate_imports(has_multimodal: bool, has_hybrid: bool) -> str:
    """Generate import statements."""
    base_imports = '''
# RAG and vector database imports
from namel3ss.rag import VectorIndexBackend
try:
    from qdrant_client import QdrantClient
except ImportError:
    QdrantClient = None
'''
    
    multimodal_imports = '''
# Multimodal imports
try:
    from namel3ss.multimodal import (
        MultimodalIngester,
        MultimodalEmbeddingProvider,
        QdrantMultimodalBackend,
    )
except ImportError:
    MultimodalIngester = None
    MultimodalEmbeddingProvider = None
    QdrantMultimodalBackend = None
'''
    
    hybrid_imports = '''
# Hybrid search imports
try:
    from namel3ss.multimodal import HybridRetriever
except ImportError:
    HybridRetriever = None
'''
    
    parts = [textwrap.dedent(base_imports).strip()]
    if has_multimodal:
        parts.append(textwrap.dedent(multimodal_imports).strip())
    if has_hybrid:
        parts.append(textwrap.dedent(hybrid_imports).strip())
    
    return "\n\n".join(parts)


def _generate_ingester_init() -> str:
    """Generate multimodal ingester initialization."""
    template = '''
# Multimodal ingester (shared across indices)
_MULTIMODAL_INGESTER = None

def _get_multimodal_ingester():
    """Get or create the shared multimodal ingester."""
    global _MULTIMODAL_INGESTER
    if _MULTIMODAL_INGESTER is None:
        if MultimodalIngester is None:
            raise ImportError("MultimodalIngester not available. Install: pip install PyMuPDF pillow")
        _MULTIMODAL_INGESTER = MultimodalIngester(
            extract_images=True,
            extract_audio=True,
            max_image_size=(1024, 1024),
        )
    return _MULTIMODAL_INGESTER
'''
    return textwrap.dedent(template).strip()


def _generate_embedding_providers(state: "BackendState") -> str:
    """Generate embedding provider initialization."""
    # Collect unique embedding models
    text_models = set()
    image_models = set()
    audio_models = set()
    
    for index_data in state.indices.values():
        text_models.add(index_data.get("embedding_model", "all-MiniLM-L6-v2"))
        if index_data.get("image_model"):
            image_models.add(index_data["image_model"])
        if index_data.get("audio_model"):
            audio_models.add(index_data["audio_model"])
    
    for pipeline_data in state.rag_pipelines.values():
        text_models.add(pipeline_data.get("query_encoder", "all-MiniLM-L6-v2"))
    
    template = '''
# Embedding providers (cached by model name)
_EMBEDDING_PROVIDERS: Dict[str, Any] = {{}}

async def _get_embedding_provider(
    text_model: str = "all-MiniLM-L6-v2",
    image_model: Optional[str] = None,
    audio_model: Optional[str] = None,
) -> Any:
    """Get or create an embedding provider."""
    key = f"{{text_model}}|{{image_model}}|{{audio_model}}"
    if key not in _EMBEDDING_PROVIDERS:
        if MultimodalEmbeddingProvider is None:
            raise ImportError("MultimodalEmbeddingProvider not available. Install: pip install sentence-transformers transformers")
        
        provider = MultimodalEmbeddingProvider(
            text_model=text_model,
            image_model=image_model or "openai/clip-vit-base-patch32",
            audio_model=audio_model or "openai/whisper-base",
            device=os.getenv("DEVICE", "cpu"),
        )
        await provider.initialize()
        _EMBEDDING_PROVIDERS[key] = provider
    
    return _EMBEDDING_PROVIDERS[key]
'''
    return textwrap.dedent(template).strip()


def _generate_vector_backends(state: "BackendState") -> str:
    """Generate vector database backend initialization."""
    template = '''
# Vector database backends (cached by collection name)
_VECTOR_BACKENDS: Dict[str, Any] = {{}}

async def _get_vector_backend(
    collection_name: str,
    backend_type: str = "qdrant",
    use_multimodal: bool = False,
) -> Any:
    """Get or create a vector database backend."""
    if collection_name not in _VECTOR_BACKENDS:
        if backend_type == "qdrant":
            if use_multimodal:
                if QdrantMultimodalBackend is None:
                    raise ImportError("QdrantMultimodalBackend not available. Install: pip install qdrant-client")
                backend = QdrantMultimodalBackend(config={{
                    "host": os.getenv("QDRANT_HOST", "localhost"),
                    "port": int(os.getenv("QDRANT_PORT", "6333")),
                    "collection_name": collection_name,
                }})
            else:
                # Use standard VectorIndexBackend for text-only
                backend = VectorIndexBackend(
                    backend_type="qdrant",
                    config={{
                        "host": os.getenv("QDRANT_HOST", "localhost"),
                        "port": int(os.getenv("QDRANT_PORT", "6333")),
                        "collection_name": collection_name,
                    }}
                )
            await backend.initialize()
            _VECTOR_BACKENDS[collection_name] = backend
        else:
            raise ValueError(f"Unsupported backend type: {{backend_type}}")
    
    return _VECTOR_BACKENDS[collection_name]
'''
    return textwrap.dedent(template).strip()


def _generate_hybrid_retrievers(state: "BackendState") -> str:
    """Generate hybrid retriever initialization."""
    template = '''
# Hybrid retrievers (cached by pipeline name)
_HYBRID_RETRIEVERS: Dict[str, Any] = {{}}

async def _get_hybrid_retriever(
    pipeline_name: str,
    vector_backend: Any,
    embedding_provider: Any,
    sparse_model: str = "bm25",
    reranker_model: Optional[str] = None,
    reranker_type: Optional[str] = None,
) -> Any:
    """Get or create a hybrid retriever."""
    if pipeline_name not in _HYBRID_RETRIEVERS:
        if HybridRetriever is None:
            raise ImportError("HybridRetriever not available. Install multimodal dependencies")
        
        retriever = HybridRetriever(
            vector_backend=vector_backend,
            embedding_provider=embedding_provider,
            enable_sparse=(sparse_model == "bm25"),
            enable_reranking=(reranker_model is not None),
            reranker_model=reranker_model,
            reranker_type=reranker_type or "cross_encoder",
        )
        await retriever.initialize()
        _HYBRID_RETRIEVERS[pipeline_name] = retriever
    
    return _HYBRID_RETRIEVERS[pipeline_name]
'''
    return textwrap.dedent(template).strip()


def _generate_rag_pipeline_factory(
    state: "BackendState",
    has_multimodal: bool,
    has_hybrid: bool,
) -> str:
    """Generate RAG pipeline factory function."""
    template_start = '''
async def initialize_rag_pipeline(pipeline_name: str) -> Dict[str, Any]:
    """Initialize a RAG pipeline by name.
    
    Returns a dict with:
        - ingester: Optional[MultimodalIngester]
        - embedding_provider: MultimodalEmbeddingProvider
        - vector_backend: VectorIndexBackend or QdrantMultimodalBackend
        - retriever: Optional[HybridRetriever]
        - config: Dict with pipeline configuration
    """
    pipeline_configs = {'''
    
    # Generate pipeline configurations
    pipeline_configs = []
    for name, pipeline_data in state.rag_pipelines.items():
        # Find corresponding index
        index_name = pipeline_data.get("index")
        index_data = state.indices.get(index_name, {})
        
        config = {
            "name": name,
            "query_encoder": pipeline_data.get("query_encoder", "all-MiniLM-L6-v2"),
            "index": index_name,
            "collection": index_data.get("collection", f"{index_name}_collection"),
            "top_k": pipeline_data.get("top_k", 10),
            "distance_metric": pipeline_data.get("distance_metric", "cosine"),
            "extract_images": index_data.get("extract_images", False),
            "extract_audio": index_data.get("extract_audio", False),
            "image_model": index_data.get("image_model"),
            "audio_model": index_data.get("audio_model"),
            "enable_hybrid": pipeline_data.get("enable_hybrid", False),
            "sparse_model": pipeline_data.get("sparse_model"),
            "dense_weight": pipeline_data.get("dense_weight", 0.7),
            "sparse_weight": pipeline_data.get("sparse_weight", 0.3),
            "reranker": pipeline_data.get("reranker"),
            "reranker_type": pipeline_data.get("reranker_type"),
        }
        pipeline_configs.append(f'        "{name}": {config!r}')
    
    template_middle = ",\n".join(pipeline_configs)
    
    template_end = '''
    }
    
    if pipeline_name not in pipeline_configs:
        raise ValueError(f"Unknown RAG pipeline: {pipeline_name}")
    
    config = pipeline_configs[pipeline_name]
    
    # Initialize components
    result = {"config": config}
    
    # Ingester (if multimodal)
    if config["extract_images"] or config["extract_audio"]:
        result["ingester"] = _get_multimodal_ingester()
    else:
        result["ingester"] = None
    
    # Embedding provider
    result["embedding_provider"] = await _get_embedding_provider(
        text_model=config["query_encoder"],
        image_model=config.get("image_model"),
        audio_model=config.get("audio_model"),
    )
    
    # Vector backend
    use_multimodal = config["extract_images"] or config["extract_audio"]
    result["vector_backend"] = await _get_vector_backend(
        collection_name=config["collection"],
        backend_type="qdrant",
        use_multimodal=use_multimodal,
    )
    
    # Hybrid retriever (if enabled)
    if config["enable_hybrid"]:
        result["retriever"] = await _get_hybrid_retriever(
            pipeline_name=pipeline_name,
            vector_backend=result["vector_backend"],
            embedding_provider=result["embedding_provider"],
            sparse_model=config.get("sparse_model", "bm25"),
            reranker_model=config.get("reranker"),
            reranker_type=config.get("reranker_type"),
        )
    else:
        result["retriever"] = None
    
    return result
'''
    
    parts = [
        textwrap.dedent(template_start).strip(),
        template_middle,
        textwrap.dedent(template_end).strip(),
    ]
    
    return "\n".join(parts)
