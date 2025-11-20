"""Configuration for multimodal RAG system."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class EmbeddingModelType(str, Enum):
    """Supported embedding model types."""
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    CLIP = "clip"
    IMAGEBIND = "imagebind"
    WHISPER = "whisper"
    OPENAI = "openai"


class SparseModelType(str, Enum):
    """Supported sparse retrieval models."""
    BM25 = "bm25"
    SPLADE = "splade"


class RerankerType(str, Enum):
    """Supported reranker types."""
    COLBERT_V2 = "colbert_v2"
    CROSS_ENCODER = "cross_encoder"
    COHERE = "cohere"


@dataclass
class MultimodalConfig:
    """Configuration for multimodal RAG features."""
    
    # Text embedding config
    text_model: str = "all-MiniLM-L6-v2"
    text_model_type: EmbeddingModelType = EmbeddingModelType.SENTENCE_TRANSFORMERS
    
    # Image embedding config
    image_model: str = "openai/clip-vit-base-patch32"
    image_model_type: EmbeddingModelType = EmbeddingModelType.CLIP
    extract_images: bool = False
    
    # Audio embedding config
    audio_model: str = "openai/whisper-base"
    extract_audio: bool = False
    
    # Hybrid search config
    enable_hybrid_search: bool = False
    sparse_model: SparseModelType = SparseModelType.BM25
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    
    # Reranking config
    enable_reranking: bool = False
    reranker_model: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_type: RerankerType = RerankerType.CROSS_ENCODER
    rerank_top_k: int = 20
    
    # Processing config
    chunk_size: int = 512
    overlap: int = 64
    batch_size: int = 32
    max_image_size: tuple = (224, 224)
    
    # Vector database config
    vector_db_type: str = "qdrant"
    collection_name: str = "multimodal_docs"
    
    # Device config
    device: str = "cpu"  # or "cuda", "mps"
    
    # Additional config
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "text_model": self.text_model,
            "text_model_type": self.text_model_type.value,
            "image_model": self.image_model,
            "image_model_type": self.image_model_type.value,
            "extract_images": self.extract_images,
            "audio_model": self.audio_model,
            "extract_audio": self.extract_audio,
            "enable_hybrid_search": self.enable_hybrid_search,
            "sparse_model": self.sparse_model.value,
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
            "enable_reranking": self.enable_reranking,
            "reranker_model": self.reranker_model,
            "reranker_type": self.reranker_type.value if self.reranker_type else None,
            "rerank_top_k": self.rerank_top_k,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "batch_size": self.batch_size,
            "max_image_size": self.max_image_size,
            "vector_db_type": self.vector_db_type,
            "collection_name": self.collection_name,
            "device": self.device,
            **self.config,
        }
