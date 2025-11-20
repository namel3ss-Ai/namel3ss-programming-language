"""Multimodal embedding providers for text, images, and audio."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result from embedding operation."""
    embeddings: np.ndarray  # Shape: (n_items, dimension)
    model: str
    modality: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # Ensure embeddings is numpy array
        if not isinstance(self.embeddings, np.ndarray):
            self.embeddings = np.array(self.embeddings)


class BaseModalEmbedder(ABC):
    """Base class for modality-specific embedders."""
    
    def __init__(self, model_name: str, device: str = "cpu", config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.device = device
        self.config = config or {}
        self.model = None
    
    @abstractmethod
    async def initialize(self):
        """Initialize the embedding model."""
        pass
    
    @abstractmethod
    async def embed(self, inputs: List[Any]) -> EmbeddingResult:
        """Embed inputs and return result."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        pass


class TextEmbedder(BaseModalEmbedder):
    """Text embedding using SentenceTransformers or OpenAI."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model_name, device, config)
        self.use_openai = model_name.startswith("text-embedding-")
    
    async def initialize(self):
        """Initialize text embedding model."""
        if self.use_openai:
            # OpenAI embeddings - no local model needed
            import os
            self.api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key required for text-embedding-* models")
            logger.info(f"Initialized OpenAI text embedder: {self.model_name}")
        else:
            # SentenceTransformers
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. Install with: "
                    "pip install sentence-transformers"
                )
            
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Text embedder initialized on {self.device}")
    
    async def embed(self, texts: List[str]) -> EmbeddingResult:
        """Embed text inputs."""
        if self.use_openai:
            return await self._embed_openai(texts)
        else:
            return await self._embed_sentence_transformers(texts)
    
    async def _embed_openai(self, texts: List[str]) -> EmbeddingResult:
        """Embed using OpenAI API."""
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required")
        
        client = openai.AsyncOpenAI(api_key=self.api_key)
        
        response = await client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        
        embeddings = np.array([item.embedding for item in response.data])
        
        return EmbeddingResult(
            embeddings=embeddings,
            model=self.model_name,
            modality="text",
            metadata={
                "provider": "openai",
                "tokens": response.usage.total_tokens,
            },
        )
    
    async def _embed_sentence_transformers(self, texts: List[str]) -> EmbeddingResult:
        """Embed using SentenceTransformers."""
        if self.model is None:
            await self.initialize()
        
        # Encode texts
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # Normalize for cosine similarity
        )
        
        return EmbeddingResult(
            embeddings=embeddings,
            model=self.model_name,
            modality="text",
            metadata={"provider": "sentence_transformers"},
        )
    
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "all-MiniLM-L12-v2": 384,
            "paraphrase-MiniLM-L6-v2": 384,
        }
        
        if self.model_name in dimensions:
            return dimensions[self.model_name]
        
        # Try to get from model if loaded
        if self.model is not None:
            return self.model.get_sentence_embedding_dimension()
        
        return 768  # Default


class ImageEmbedder(BaseModalEmbedder):
    """Image embedding using CLIP."""
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model_name, device, config)
        self.processor = None
    
    async def initialize(self):
        """Initialize CLIP model."""
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise ImportError(
                "transformers required. Install with: "
                "pip install transformers torch"
            )
        
        logger.info(f"Loading CLIP model: {self.model_name}")
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.eval()
        logger.info(f"Image embedder initialized on {self.device}")
    
    async def embed(self, images: List[bytes]) -> EmbeddingResult:
        """Embed image inputs (as bytes)."""
        if self.model is None:
            await self.initialize()
        
        try:
            from PIL import Image
            import torch
            import io
        except ImportError:
            raise ImportError("PIL and torch required")
        
        # Convert bytes to PIL images
        pil_images = []
        for img_bytes in images:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            pil_images.append(img)
        
        # Process images
        inputs = self.processor(
            images=pil_images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # Normalize embeddings
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        embeddings = image_features.cpu().numpy()
        
        return EmbeddingResult(
            embeddings=embeddings,
            model=self.model_name,
            modality="image",
            metadata={"provider": "clip"},
        )
    
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        dimensions = {
            "openai/clip-vit-base-patch32": 512,
            "openai/clip-vit-base-patch16": 512,
            "openai/clip-vit-large-patch14": 768,
        }
        return dimensions.get(self.model_name, 512)


class AudioEmbedder(BaseModalEmbedder):
    """Audio embedding using Whisper transcription + text embedding."""
    
    def __init__(
        self,
        model_name: str = "openai/whisper-base",
        device: str = "cpu",
        text_embedder: Optional[TextEmbedder] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model_name, device, config)
        self.text_embedder = text_embedder or TextEmbedder()
        self.processor = None
    
    async def initialize(self):
        """Initialize Whisper model."""
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
        except ImportError:
            raise ImportError(
                "transformers required. Install with: "
                "pip install transformers torch"
            )
        
        logger.info(f"Loading Whisper model: {self.model_name}")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name
        ).to(self.device)
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model.eval()
        
        # Initialize text embedder
        await self.text_embedder.initialize()
        
        logger.info(f"Audio embedder initialized on {self.device}")
    
    async def embed(self, audio_inputs: List[bytes]) -> EmbeddingResult:
        """
        Embed audio inputs by transcribing and embedding transcript.
        
        Args:
            audio_inputs: List of audio file bytes
            
        Returns:
            EmbeddingResult with text embeddings of transcripts
        """
        if self.model is None:
            await self.initialize()
        
        # Transcribe audio files
        transcripts = []
        for audio_bytes in audio_inputs:
            transcript = await self._transcribe_audio(audio_bytes)
            transcripts.append(transcript)
        
        # Embed transcripts as text
        result = await self.text_embedder.embed(transcripts)
        
        # Update metadata
        result.modality = "audio"
        result.metadata["audio_model"] = self.model_name
        result.metadata["transcripts"] = transcripts
        
        return result
    
    async def _transcribe_audio(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes to text."""
        try:
            import torch
            import librosa
            import io
            import soundfile as sf
        except ImportError:
            raise ImportError(
                "librosa and soundfile required. Install with: "
                "pip install librosa soundfile"
            )
        
        # Load audio
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
        
        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            audio_array = librosa.resample(
                audio_array,
                orig_sr=sample_rate,
                target_sr=16000
            )
            sample_rate = 16000
        
        # Process audio
        inputs = self.processor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).to(self.device)
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.model.generate(inputs.input_features)
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]
        
        return transcription
    
    def get_dimension(self) -> int:
        """Return embedding dimension (from text embedder)."""
        return self.text_embedder.get_dimension()


class MultimodalEmbeddingProvider:
    """
    Unified multimodal embedding provider.
    
    Manages text, image, and audio embedders and provides a unified interface.
    """
    
    def __init__(
        self,
        text_model: str = "all-MiniLM-L6-v2",
        image_model: str = "openai/clip-vit-base-patch32",
        audio_model: str = "openai/whisper-base",
        device: str = "cpu",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize multimodal embedding provider.
        
        Args:
            text_model: Text embedding model name
            image_model: Image embedding model name (CLIP)
            audio_model: Audio embedding model name (Whisper)
            device: Device to use ("cpu", "cuda", "mps")
            config: Additional configuration
        """
        self.config = config or {}
        
        self.text_embedder = TextEmbedder(
            model_name=text_model,
            device=device,
            config=self.config.get("text_config", {}),
        )
        
        self.image_embedder = ImageEmbedder(
            model_name=image_model,
            device=device,
            config=self.config.get("image_config", {}),
        )
        
        self.audio_embedder = AudioEmbedder(
            model_name=audio_model,
            device=device,
            text_embedder=self.text_embedder,
            config=self.config.get("audio_config", {}),
        )
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize all embedders."""
        if not self._initialized:
            await self.text_embedder.initialize()
            await self.image_embedder.initialize()
            await self.audio_embedder.initialize()
            self._initialized = True
            logger.info("Multimodal embedding provider initialized")
    
    async def embed_text(self, texts: List[str]) -> EmbeddingResult:
        """Embed text inputs."""
        if not self._initialized:
            await self.initialize()
        return await self.text_embedder.embed(texts)
    
    async def embed_images(self, images: List[bytes]) -> EmbeddingResult:
        """Embed image inputs."""
        if not self._initialized:
            await self.initialize()
        return await self.image_embedder.embed(images)
    
    async def embed_audio(self, audio: List[bytes]) -> EmbeddingResult:
        """Embed audio inputs."""
        if not self._initialized:
            await self.initialize()
        return await self.audio_embedder.embed(audio)
    
    def get_dimensions(self) -> Dict[str, int]:
        """Get embedding dimensions for each modality."""
        return {
            "text": self.text_embedder.get_dimension(),
            "image": self.image_embedder.get_dimension(),
            "audio": self.audio_embedder.get_dimension(),
        }
