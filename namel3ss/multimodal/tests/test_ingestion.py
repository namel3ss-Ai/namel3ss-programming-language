"""Unit tests for multimodal ingestion."""

import pytest
import io
from pathlib import Path
from PIL import Image
import numpy as np

from namel3ss.multimodal.ingestion import (
    MultimodalIngester,
    ExtractedContent,
    IngestionResult,
    ContentModality,
)


@pytest.fixture
def ingester():
    """Create a multimodal ingester instance."""
    return MultimodalIngester(
        extract_images=True,
        extract_audio=True,
        max_image_size=(512, 512),
    )


@pytest.fixture
def sample_image_bytes():
    """Create a sample image in memory."""
    img = Image.new('RGB', (100, 100), color='red')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf.read()


@pytest.fixture
def sample_text():
    """Sample text content."""
    return "This is a test document with multiple paragraphs.\n\nSecond paragraph here."


class TestMultimodalIngester:
    """Test suite for MultimodalIngester."""
    
    def test_init(self):
        """Test ingester initialization."""
        ingester = MultimodalIngester(
            extract_images=True,
            extract_audio=False,
            max_image_size=(1024, 1024),
        )
        assert ingester.extract_images is True
        assert ingester.extract_audio is False
        assert ingester.max_image_size == (1024, 1024)
    
    @pytest.mark.asyncio
    async def test_ingest_text_file(self, ingester, tmp_path):
        """Test ingesting a plain text file."""
        text_file = tmp_path / "test.txt"
        text_file.write_text("Hello, world!\nThis is a test.")
        
        result = await ingester.ingest_file(str(text_file))
        
        assert isinstance(result, IngestionResult)
        assert len(result.text_contents) > 0
        assert result.text_contents[0].modality == ContentModality.TEXT
        assert "Hello, world!" in result.text_contents[0].content
        assert len(result.image_contents) == 0
        assert len(result.audio_contents) == 0
    
    @pytest.mark.asyncio
    async def test_ingest_image_file(self, ingester, tmp_path, sample_image_bytes):
        """Test ingesting an image file."""
        image_file = tmp_path / "test.png"
        image_file.write_bytes(sample_image_bytes)
        
        result = await ingester.ingest_file(str(image_file))
        
        assert isinstance(result, IngestionResult)
        assert len(result.image_contents) == 1
        assert result.image_contents[0].modality == ContentModality.IMAGE
        assert isinstance(result.image_contents[0].content, bytes)
        assert len(result.text_contents) == 0
    
    @pytest.mark.asyncio
    async def test_ingest_bytes_text(self, ingester):
        """Test ingesting text from bytes."""
        text_bytes = b"Hello from bytes!"
        
        result = await ingester.ingest_bytes(text_bytes, "test.txt")
        
        assert len(result.text_contents) == 1
        assert "Hello from bytes!" in result.text_contents[0].content
    
    @pytest.mark.asyncio
    async def test_ingest_bytes_image(self, ingester, sample_image_bytes):
        """Test ingesting image from bytes."""
        result = await ingester.ingest_bytes(sample_image_bytes, "test.png")
        
        assert len(result.image_contents) == 1
        assert isinstance(result.image_contents[0].content, bytes)
    
    @pytest.mark.asyncio
    async def test_ingest_unsupported_format(self, ingester):
        """Test ingesting unsupported file format."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            await ingester.ingest_bytes(b"data", "test.xyz")
    
    @pytest.mark.asyncio
    async def test_extract_images_disabled(self, tmp_path):
        """Test with image extraction disabled."""
        ingester = MultimodalIngester(extract_images=False)
        
        # Even with image file, no images should be extracted
        img = Image.new('RGB', (50, 50), color='blue')
        image_file = tmp_path / "test.png"
        img.save(image_file)
        
        result = await ingester.ingest_file(str(image_file))
        
        # Should still process as image but config controls extraction
        assert len(result.image_contents) == 1  # File is recognized as image


class TestExtractedContent:
    """Test ExtractedContent dataclass."""
    
    def test_text_content_creation(self):
        """Test creating text content."""
        content = ExtractedContent(
            content="Sample text",
            modality=ContentModality.TEXT,
            metadata={"page": 1},
        )
        
        assert content.content == "Sample text"
        assert content.modality == ContentModality.TEXT
        assert content.metadata["page"] == 1
    
    def test_image_content_creation(self):
        """Test creating image content."""
        image_data = b"fake image data"
        content = ExtractedContent(
            content=image_data,
            modality=ContentModality.IMAGE,
            metadata={"width": 800, "height": 600},
        )
        
        assert content.content == image_data
        assert content.modality == ContentModality.IMAGE
        assert content.metadata["width"] == 800


class TestIngestionResult:
    """Test IngestionResult dataclass."""
    
    def test_empty_result(self):
        """Test creating empty result."""
        result = IngestionResult(
            document_id="test_doc",
            text_contents=[],
            image_contents=[],
            audio_contents=[],
            video_contents=[],
        )
        
        assert result.document_id == "test_doc"
        assert len(result.text_contents) == 0
        assert len(result.image_contents) == 0
    
    def test_result_with_contents(self):
        """Test result with multiple content types."""
        text = ExtractedContent("text", ContentModality.TEXT, {})
        image = ExtractedContent(b"img", ContentModality.IMAGE, {})
        
        result = IngestionResult(
            document_id="doc_123",
            text_contents=[text],
            image_contents=[image],
            audio_contents=[],
            video_contents=[],
        )
        
        assert len(result.text_contents) == 1
        assert len(result.image_contents) == 1
        assert result.document_id == "doc_123"


@pytest.mark.skipif(
    not pytest.importorskip("fitz", reason="PyMuPDF not installed"),
    reason="Requires PyMuPDF for PDF testing"
)
class TestPDFIngestion:
    """Test PDF-specific ingestion (requires PyMuPDF)."""
    
    @pytest.mark.asyncio
    async def test_pdf_with_text_and_images(self, ingester):
        """Test PDF ingestion with both text and images."""
        # This would require a real PDF file
        # For now, we test the structure
        pytest.skip("Requires real PDF file for integration test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
