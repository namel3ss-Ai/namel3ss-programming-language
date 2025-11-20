"""Unit tests for multimodal embeddings."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from namel3ss.multimodal.embeddings import (
    TextEmbedder,
    ImageEmbedder,
    AudioEmbedder,
    MultimodalEmbeddingProvider,
    EmbeddingResult,
)


@pytest.fixture
def sample_texts():
    """Sample text inputs."""
    return ["Hello world", "This is a test", "Multimodal RAG system"]


@pytest.fixture
def sample_image_bytes():
    """Sample image as bytes."""
    # Create a minimal valid PNG
    import io
    from PIL import Image
    img = Image.new('RGB', (32, 32), color='green')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf.read()


class TestTextEmbedder:
    """Test suite for TextEmbedder."""
    
    @pytest.mark.asyncio
    async def test_init_sentence_transformers(self):
        """Test initialization with SentenceTransformers."""
        embedder = TextEmbedder(model_name="all-MiniLM-L6-v2", device="cpu")
        await embedder.initialize()
        
        assert embedder.model is not None
        assert embedder.embedding_dim == 384
    
    @pytest.mark.asyncio
    async def test_embed_texts(self, sample_texts):
        """Test text embedding."""
        embedder = TextEmbedder(model_name="all-MiniLM-L6-v2", device="cpu")
        await embedder.initialize()
        
        result = await embedder.embed(sample_texts)
        
        assert isinstance(result, EmbeddingResult)
        assert result.embeddings.shape == (3, 384)
        assert np.all(np.isfinite(result.embeddings))
        assert result.metadata["model"] == "all-MiniLM-L6-v2"
    
    @pytest.mark.asyncio
    async def test_embed_single_text(self):
        """Test embedding single text."""
        embedder = TextEmbedder(model_name="all-MiniLM-L6-v2", device="cpu")
        await embedder.initialize()
        
        result = await embedder.embed(["Single text"])
        
        assert result.embeddings.shape == (1, 384)
    
    @pytest.mark.asyncio
    async def test_empty_input(self):
        """Test handling empty input."""
        embedder = TextEmbedder(model_name="all-MiniLM-L6-v2", device="cpu")
        await embedder.initialize()
        
        with pytest.raises((ValueError, IndexError)):
            await embedder.embed([])


class TestImageEmbedder:
    """Test suite for ImageEmbedder."""
    
    @pytest.mark.asyncio
    async def test_init_clip(self):
        """Test initialization with CLIP."""
        embedder = ImageEmbedder(
            model_name="openai/clip-vit-base-patch32",
            device="cpu"
        )
        await embedder.initialize()
        
        assert embedder.model is not None
        assert embedder.processor is not None
        assert embedder.embedding_dim == 512
    
    @pytest.mark.asyncio
    async def test_embed_images(self, sample_image_bytes):
        """Test image embedding."""
        embedder = ImageEmbedder(
            model_name="openai/clip-vit-base-patch32",
            device="cpu"
        )
        await embedder.initialize()
        
        result = await embedder.embed([sample_image_bytes])
        
        assert isinstance(result, EmbeddingResult)
        assert result.embeddings.shape == (1, 512)
        assert np.all(np.isfinite(result.embeddings))
        assert result.metadata["model"] == "openai/clip-vit-base-patch32"
    
    @pytest.mark.asyncio
    async def test_embed_multiple_images(self, sample_image_bytes):
        """Test embedding multiple images."""
        embedder = ImageEmbedder(
            model_name="openai/clip-vit-base-patch32",
            device="cpu"
        )
        await embedder.initialize()
        
        images = [sample_image_bytes] * 3
        result = await embedder.embed(images)
        
        assert result.embeddings.shape == (3, 512)


class TestAudioEmbedder:
    """Test suite for AudioEmbedder."""
    
    @pytest.mark.asyncio
    async def test_init_whisper(self):
        """Test initialization with Whisper."""
        embedder = AudioEmbedder(
            model_name="openai/whisper-base",
            device="cpu"
        )
        await embedder.initialize()
        
        assert embedder.whisper_processor is not None
        assert embedder.whisper_model is not None
    
    @pytest.mark.asyncio
    @patch("namel3ss.multimodal.embeddings.AudioEmbedder._transcribe_audio")
    async def test_embed_audio_with_transcription(self, mock_transcribe):
        """Test audio embedding via transcription."""
        mock_transcribe.return_value = ["This is a test transcription"]
        
        embedder = AudioEmbedder(
            model_name="openai/whisper-base",
            device="cpu",
            text_embedder_model="all-MiniLM-L6-v2"
        )
        await embedder.initialize()
        
        # Mock audio bytes
        audio_bytes = [b"fake audio data"]
        result = await embedder.embed(audio_bytes)
        
        assert isinstance(result, EmbeddingResult)
        assert "transcripts" in result.metadata
        mock_transcribe.assert_called_once()


class TestMultimodalEmbeddingProvider:
    """Test suite for MultimodalEmbeddingProvider."""
    
    @pytest.mark.asyncio
    async def test_init_all_embedders(self):
        """Test initialization of all embedders."""
        provider = MultimodalEmbeddingProvider(
            text_model="all-MiniLM-L6-v2",
            image_model="openai/clip-vit-base-patch32",
            audio_model="openai/whisper-base",
            device="cpu",
        )
        await provider.initialize()
        
        assert provider.text_embedder is not None
        assert provider.image_embedder is not None
        assert provider.audio_embedder is not None
    
    @pytest.mark.asyncio
    async def test_embed_text(self, sample_texts):
        """Test text embedding through provider."""
        provider = MultimodalEmbeddingProvider(
            text_model="all-MiniLM-L6-v2",
            device="cpu",
        )
        await provider.initialize()
        
        result = await provider.embed_text(sample_texts)
        
        assert result.embeddings.shape[0] == 3
        assert result.embeddings.shape[1] == 384
    
    @pytest.mark.asyncio
    async def test_embed_images(self, sample_image_bytes):
        """Test image embedding through provider."""
        provider = MultimodalEmbeddingProvider(
            image_model="openai/clip-vit-base-patch32",
            device="cpu",
        )
        await provider.initialize()
        
        result = await provider.embed_images([sample_image_bytes])
        
        assert result.embeddings.shape == (1, 512)
    
    @pytest.mark.asyncio
    async def test_get_dimensions(self):
        """Test getting embedding dimensions."""
        provider = MultimodalEmbeddingProvider(
            text_model="all-MiniLM-L6-v2",
            image_model="openai/clip-vit-base-patch32",
            device="cpu",
        )
        await provider.initialize()
        
        dims = provider.get_embedding_dimensions()
        
        assert dims["text"] == 384
        assert dims["image"] == 512
        assert dims["audio"] == 384  # Uses text embedder


class TestEmbeddingResult:
    """Test EmbeddingResult dataclass."""
    
    def test_create_result(self):
        """Test creating embedding result."""
        embeddings = np.random.rand(5, 128)
        result = EmbeddingResult(
            embeddings=embeddings,
            metadata={"model": "test-model"},
        )
        
        assert result.embeddings.shape == (5, 128)
        assert result.metadata["model"] == "test-model"
    
    def test_result_with_extra_metadata(self):
        """Test result with additional metadata."""
        result = EmbeddingResult(
            embeddings=np.zeros((2, 64)),
            metadata={
                "model": "model-name",
                "transcripts": ["text1", "text2"],
                "batch_size": 2,
            },
        )
        
        assert "transcripts" in result.metadata
        assert result.metadata["batch_size"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
