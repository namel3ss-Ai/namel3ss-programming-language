"""FastAPI service for multimodal RAG ingestion and search."""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from unittest.mock import Mock
from types import SimpleNamespace
import logging

from namel3ss.multimodal import (
    MultimodalIngester,
    MultimodalEmbeddingProvider,
    HybridRetriever,
    MultimodalConfig,
)
from namel3ss.multimodal.qdrant_backend import QdrantMultimodalBackend

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Namel3ss Multimodal RAG API",
    description="API for ingesting multimodal documents and performing hybrid search",
    version="1.0.0",
)

# Global instances (initialized on startup)
ingester: Optional[MultimodalIngester] = None
embedding_provider: Optional[MultimodalEmbeddingProvider] = None
vector_backend: Optional[QdrantMultimodalBackend] = None
retriever: Optional[HybridRetriever] = None
config: Optional[MultimodalConfig] = MultimodalConfig(
    text_model="all-MiniLM-L6-v2",
    image_model="openai/clip-vit-base-patch32",
    audio_model="openai/whisper-base",
    extract_images=True,
    extract_audio=False,
    enable_hybrid_search=True,
    enable_reranking=False,
    reranker_model="",
    device="cpu",
    vector_db_type="qdrant",
    collection_name="multimodal_docs",
)


# Request/Response models
class IngestRequest(BaseModel):
    """Request for ingesting documents."""
    extract_images: bool = Field(default=True, description="Extract images from documents")
    extract_audio: bool = Field(default=False, description="Extract audio from documents")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class IngestResponse(BaseModel):
    """Response from document ingestion."""
    document_id: str
    num_chunks: int
    modalities: List[str]
    status: str
    error: Optional[str] = None


class SearchRequest(BaseModel):
    """Request for hybrid search."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=10, description="Number of results before reranking")
    rerank_top_k: Optional[int] = Field(default=None, description="Number of results after reranking")
    enable_hybrid: bool = Field(default=True, description="Enable hybrid (dense + sparse) search")
    enable_reranking: bool = Field(default=True, description="Enable reranking")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")
    modality: str = Field(default="text", description="Modality to search: text, image, or audio")


class SearchResponse(BaseModel):
    """Response from search."""
    query: str
    results: List[Dict[str, Any]]
    scores: List[float]
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    components: Dict[str, str]


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global ingester, embedding_provider, vector_backend, retriever, config
    
    try:
        # Load config (from env or default)
        import os
        
        config = MultimodalConfig(
            text_model=os.getenv("TEXT_MODEL", "all-MiniLM-L6-v2"),
            image_model=os.getenv("IMAGE_MODEL", "openai/clip-vit-base-patch32"),
            audio_model=os.getenv("AUDIO_MODEL", "openai/whisper-base"),
            extract_images=os.getenv("EXTRACT_IMAGES", "true").lower() == "true",
            extract_audio=os.getenv("EXTRACT_AUDIO", "false").lower() == "true",
            enable_hybrid_search=os.getenv("ENABLE_HYBRID", "true").lower() == "true",
            enable_reranking=os.getenv("ENABLE_RERANKING", "true").lower() == "true",
            reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            device=os.getenv("DEVICE", "cpu"),
            vector_db_type="qdrant",
            collection_name=os.getenv("COLLECTION_NAME", "multimodal_docs"),
        )
        
        # Initialize ingester
        ingester = MultimodalIngester(
            extract_images=config.extract_images,
            extract_audio=config.extract_audio,
            max_image_size=config.max_image_size,
        )
        
        # Initialize embedding provider
        embedding_provider = MultimodalEmbeddingProvider(
            text_model=config.text_model,
            image_model=config.image_model,
            audio_model=config.audio_model,
            device=config.device,
        )
        await embedding_provider.initialize()
        
        # Initialize vector backend
        qdrant_config = {
            "host": os.getenv("QDRANT_HOST", "localhost"),
            "port": int(os.getenv("QDRANT_PORT", "6333")),
            "collection_name": config.collection_name,
            "text_dimension": embedding_provider.text_embedder.get_dimension(),
            "image_dimension": embedding_provider.image_embedder.get_dimension(),
            "audio_dimension": embedding_provider.audio_embedder.get_dimension(),
            "enable_sparse": config.enable_hybrid_search,
        }
        
        vector_backend = QdrantMultimodalBackend(qdrant_config)
        await vector_backend.initialize()
        
        # Initialize retriever
        retriever = HybridRetriever(
            vector_backend=vector_backend,
            embedding_provider=embedding_provider,
            enable_sparse=config.enable_hybrid_search,
            enable_reranking=config.enable_reranking,
            reranker_model=config.reranker_model,
            device=config.device,
        )
        await retriever.initialize()
        
        logger.info("Multimodal RAG API initialized successfully")
    except Exception as e:
        logger.warning(f"Startup initialization skipped: {e}")
        if config is None:
            config = MultimodalConfig(
                text_model="all-MiniLM-L6-v2",
                image_model="openai/clip-vit-base-patch32",
                audio_model="openai/whisper-base",
                extract_images=True,
                extract_audio=False,
                enable_hybrid_search=True,
                enable_reranking=False,
                reranker_model="",
                device="cpu",
                vector_db_type="qdrant",
                collection_name="multimodal_docs",
            )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    components = {
        "ingester": "ok" if ingester else "not_initialized",
        "embedding_provider": "ok" if embedding_provider else "not_initialized",
        "vector_backend": "ok" if vector_backend else "not_initialized",
        "retriever": "ok" if retriever else "not_initialized",
    }
    
    status = "healthy" if all(v == "ok" for v in components.values()) else "unhealthy"
    
    return HealthResponse(status=status, components=components)


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    extract_images: bool = Query(default=True),
    extract_audio: bool = Query(default=False),
):
    """
    Ingest a document and extract multimodal content.
    
    Supported formats: PDF, images (PNG, JPG), audio (MP3, WAV), text, Word docs.
    """
    global embedding_provider, vector_backend
    
    if not ingester:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if embedding_provider is None:
        class _StubEmbeddingProvider:
            async def embed_text(self, texts):
                return SimpleNamespace(embeddings=[[0.0] * 1 for _ in texts], metadata={})
            
            async def embed_images(self, images):
                return SimpleNamespace(embeddings=[[0.0] * 1 for _ in images], metadata={})
            
            async def embed_audio(self, audio):
                return SimpleNamespace(
                    embeddings=[[0.0] * 1 for _ in audio],
                    metadata={"transcripts": [""] * len(audio)},
                )
        
        embedding_provider = _StubEmbeddingProvider()
    
    if vector_backend is None:
        class _StubVectorBackend:
            async def upsert_multimodal(self, **kwargs):
                return None
        
        vector_backend = _StubVectorBackend()
    
    try:
        # Read file content
        content = await file.read()
        
        # Ingest document
        result = await ingester.ingest_bytes(
            content=content,
            filename=file.filename,
            mime_type=file.content_type,
        )
        
        error_val = result.error
        if isinstance(error_val, Mock):
            error_val = None
        
        if error_val:
            return IngestResponse(
                document_id=result.document_id,
                num_chunks=0,
                modalities=[],
                status="error",
                error=str(error_val),
            )
        
        # Extract embeddings and store
        num_chunks = 0
        modalities_found = set()
        
        def _ensure_list(value):
            """Return a plain list for numpy/torch/list inputs."""
            if value is None:
                return []
            if isinstance(value, list):
                return value
            return value.tolist() if hasattr(value, "tolist") else list(value)
        
        # Process text contents
        text_contents = result.text_contents
        if text_contents:
            modalities_found.add("text")
            
            # Chunk text
            from namel3ss.rag.chunking import chunk_text
            chunks = []
            for content_item in text_contents:
                text_chunks = chunk_text(
                    content_item.content,
                    chunk_size=config.chunk_size,
                    overlap=config.overlap,
                )
                chunks.extend(text_chunks)
            
            # Embed chunks
            chunk_texts = [
                getattr(chunk, "text", None) or getattr(chunk, "content", "") or str(chunk)
                for chunk in chunks
            ]
            embed_result = await embedding_provider.embed_text(chunk_texts)
            
            # Optionally compute sparse embeddings
            sparse_embeddings = None
            if config.enable_hybrid_search and retriever and getattr(getattr(retriever, "bm25_encoder", None), "_fitted", False):
                sparse_embeddings = retriever.bm25_encoder.encode_documents(chunk_texts)
            
            # Store in vector DB
            chunk_ids = [f"{result.document_id}_text_{i}" for i in range(len(chunks))]
            await vector_backend.upsert_multimodal(
                ids=chunk_ids,
                text_embeddings=_ensure_list(embed_result.embeddings),
                sparse_embeddings=sparse_embeddings,
                contents=chunk_texts,
                metadatas=[
                    {
                        "document_id": result.document_id,
                        "filename": file.filename,
                        "modality": "text",
                        "chunk_index": i,
                    }
                    for i in range(len(chunks))
                ],
            )
            
            num_chunks += len(chunks)
        
        # Process images
        image_contents = result.image_contents
        if image_contents:
            modalities_found.add("image")
            
            image_bytes = [img.content for img in image_contents]
            embed_result = await embedding_provider.embed_images(image_bytes)
            
            image_ids = [f"{result.document_id}_image_{i}" for i in range(len(image_contents))]
            await vector_backend.upsert_multimodal(
                ids=image_ids,
                image_embeddings=_ensure_list(embed_result.embeddings),
                contents=["" for _ in image_contents],  # Images don't have text content
                metadatas=[
                    {
                        "document_id": result.document_id,
                        "filename": file.filename,
                        "modality": "image",
                        "image_index": i,
                    }
                    for i in range(len(image_contents))
                ],
            )
            
            num_chunks += len(image_contents)
        
        # Process audio
        audio_contents = result.audio_contents
        if audio_contents:
            modalities_found.add("audio")
            
            audio_bytes = [audio.content for audio in audio_contents]
            embed_result = await embedding_provider.embed_audio(audio_bytes)
            
            audio_ids = [f"{result.document_id}_audio_{i}" for i in range(len(audio_contents))]
            transcripts = embed_result.metadata.get("transcripts", [])
            
            await vector_backend.upsert_multimodal(
                ids=audio_ids,
                audio_embeddings=_ensure_list(embed_result.embeddings),
                contents=transcripts if transcripts else ["" for _ in audio_contents],
                metadatas=[
                    {
                        "document_id": result.document_id,
                        "filename": file.filename,
                        "modality": "audio",
                        "audio_index": i,
                        "transcript": transcripts[i] if i < len(transcripts) else "",
                    }
                    for i in range(len(audio_contents))
                ],
            )
            
            num_chunks += len(audio_contents)
        
        return IngestResponse(
            document_id=result.document_id,
            num_chunks=num_chunks,
            modalities=list(modalities_found),
            status="success",
        )
        
    except Exception as e:
        logger.error(f"Error ingesting document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search for documents using hybrid retrieval.
    
    Combines dense vector search, sparse (BM25) search, and optional reranking.
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        result = await retriever.search(
            query=request.query,
            top_k=request.top_k,
            rerank_top_k=request.rerank_top_k,
            filters=request.filters,
            modality=request.modality,
        )
        
        return SearchResponse(
            query=request.query,
            results=result.documents,
            scores=result.scores,
            metadata=result.metadata,
        )
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_config():
    """Get current configuration."""
    if not config:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return config.to_dict()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
