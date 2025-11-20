"""Multimodal document ingestion supporting text, images, and audio."""

from __future__ import annotations

import logging
import mimetypes
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, BinaryIO
import io

logger = logging.getLogger(__name__)


class ModalityType(str, Enum):
    """Types of modalities supported."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class ExtractedContent:
    """Content extracted from a document."""
    modality: ModalityType
    content: Union[str, bytes]
    metadata: Dict[str, Any] = field(default_factory=dict)
    page_number: Optional[int] = None
    position: Optional[Dict[str, float]] = None  # x, y, width, height for images


@dataclass
class IngestionResult:
    """Result from document ingestion."""
    document_id: str
    contents: List[ExtractedContent]
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    @property
    def text_contents(self) -> List[ExtractedContent]:
        """Get all text contents."""
        return [c for c in self.contents if c.modality == ModalityType.TEXT]
    
    @property
    def image_contents(self) -> List[ExtractedContent]:
        """Get all image contents."""
        return [c for c in self.contents if c.modality == ModalityType.IMAGE]
    
    @property
    def audio_contents(self) -> List[ExtractedContent]:
        """Get all audio contents."""
        return [c for c in self.contents if c.modality == ModalityType.AUDIO]


class MultimodalIngester:
    """
    Multimodal document ingester.
    
    Extracts text, images, and audio from various document formats including
    PDFs, Word documents, images, and audio files.
    """
    
    def __init__(
        self,
        extract_images: bool = True,
        extract_audio: bool = True,
        max_image_size: tuple = (1024, 1024),
        audio_sample_rate: int = 16000,
    ):
        """
        Initialize ingester.
        
        Args:
            extract_images: Whether to extract images from documents
            extract_audio: Whether to extract audio from documents/videos
            max_image_size: Maximum image dimensions (width, height)
            audio_sample_rate: Sample rate for audio extraction
        """
        self.extract_images = extract_images
        self.extract_audio = extract_audio
        self.max_image_size = max_image_size
        self.audio_sample_rate = audio_sample_rate
    
    async def ingest_file(self, file_path: Union[str, Path]) -> IngestionResult:
        """
        Ingest a file and extract all modalities.
        
        Args:
            file_path: Path to file
            
        Returns:
            IngestionResult with extracted contents
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return IngestionResult(
                document_id=str(file_path),
                contents=[],
                error=f"File not found: {file_path}",
            )
        
        # Determine file type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.pdf' or (mime_type and 'pdf' in mime_type):
                return await self._ingest_pdf(file_path)
            elif suffix in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
                return await self._ingest_image(file_path)
            elif suffix in ['.mp3', '.wav', '.flac', '.m4a', '.ogg']:
                return await self._ingest_audio(file_path)
            elif suffix in ['.mp4', '.avi', '.mov', '.mkv']:
                return await self._ingest_video(file_path)
            elif suffix in ['.txt', '.md', '.rst']:
                return await self._ingest_text(file_path)
            elif suffix in ['.docx', '.doc']:
                return await self._ingest_word(file_path)
            else:
                # Try as plain text
                return await self._ingest_text(file_path)
        except Exception as e:
            logger.error(f"Error ingesting {file_path}: {e}")
            return IngestionResult(
                document_id=str(file_path),
                contents=[],
                error=str(e),
            )
    
    async def ingest_bytes(
        self,
        content: bytes,
        filename: str,
        mime_type: Optional[str] = None,
    ) -> IngestionResult:
        """
        Ingest raw bytes.
        
        Args:
            content: Raw file content
            filename: Original filename
            mime_type: Optional MIME type
            
        Returns:
            IngestionResult with extracted contents
        """
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(filename)
        
        suffix = Path(filename).suffix.lower()
        
        try:
            if suffix == '.pdf' or (mime_type and 'pdf' in mime_type):
                return await self._ingest_pdf_bytes(content, filename)
            elif suffix in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                return await self._ingest_image_bytes(content, filename)
            elif suffix in ['.mp3', '.wav', '.flac']:
                return await self._ingest_audio_bytes(content, filename)
            else:
                # Try as text
                return await self._ingest_text_bytes(content, filename)
        except Exception as e:
            logger.error(f"Error ingesting bytes for {filename}: {e}")
            return IngestionResult(
                document_id=filename,
                contents=[],
                error=str(e),
            )
    
    async def _ingest_pdf(self, file_path: Path) -> IngestionResult:
        """Extract text and images from PDF using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF processing. "
                "Install with: pip install PyMuPDF"
            )
        
        contents = []
        doc_id = str(file_path)
        
        try:
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text
                text = page.get_text()
                if text.strip():
                    contents.append(ExtractedContent(
                        modality=ModalityType.TEXT,
                        content=text,
                        page_number=page_num + 1,
                        metadata={"source": "pdf_text"},
                    ))
                
                # Extract images if enabled
                if self.extract_images:
                    image_list = page.get_images(full=True)
                    for img_index, img_info in enumerate(image_list):
                        try:
                            xref = img_info[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]
                            
                            # Resize if needed
                            image_bytes = self._resize_image_bytes(
                                image_bytes,
                                self.max_image_size
                            )
                            
                            contents.append(ExtractedContent(
                                modality=ModalityType.IMAGE,
                                content=image_bytes,
                                page_number=page_num + 1,
                                metadata={
                                    "source": "pdf_image",
                                    "image_index": img_index,
                                    "format": image_ext,
                                },
                            ))
                        except Exception as e:
                            logger.warning(
                                f"Failed to extract image {img_index} "
                                f"from page {page_num + 1}: {e}"
                            )
            
            doc.close()
            
            return IngestionResult(
                document_id=doc_id,
                contents=contents,
                metadata={
                    "file_type": "pdf",
                    "num_pages": len(doc),
                },
            )
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return IngestionResult(
                document_id=doc_id,
                contents=[],
                error=str(e),
            )
    
    async def _ingest_pdf_bytes(self, content: bytes, filename: str) -> IngestionResult:
        """Extract text and images from PDF bytes."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF required for PDF processing")
        
        contents = []
        
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text
                text = page.get_text()
                if text.strip():
                    contents.append(ExtractedContent(
                        modality=ModalityType.TEXT,
                        content=text,
                        page_number=page_num + 1,
                        metadata={"source": "pdf_text"},
                    ))
                
                # Extract images if enabled
                if self.extract_images:
                    image_list = page.get_images(full=True)
                    for img_index, img_info in enumerate(image_list):
                        try:
                            xref = img_info[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            
                            image_bytes = self._resize_image_bytes(
                                image_bytes,
                                self.max_image_size
                            )
                            
                            contents.append(ExtractedContent(
                                modality=ModalityType.IMAGE,
                                content=image_bytes,
                                page_number=page_num + 1,
                                metadata={
                                    "source": "pdf_image",
                                    "image_index": img_index,
                                },
                            ))
                        except Exception as e:
                            logger.warning(f"Failed to extract image: {e}")
            
            doc.close()
            
            return IngestionResult(
                document_id=filename,
                contents=contents,
                metadata={"file_type": "pdf"},
            )
        except Exception as e:
            return IngestionResult(
                document_id=filename,
                contents=[],
                error=str(e),
            )
    
    async def _ingest_image(self, file_path: Path) -> IngestionResult:
        """Ingest an image file."""
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow required for image processing")
        
        try:
            img = Image.open(file_path)
            
            # Resize if needed
            if img.size[0] > self.max_image_size[0] or img.size[1] > self.max_image_size[1]:
                img.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format=img.format or 'PNG')
            image_bytes = img_byte_arr.getvalue()
            
            contents = [ExtractedContent(
                modality=ModalityType.IMAGE,
                content=image_bytes,
                metadata={
                    "source": "image_file",
                    "format": img.format,
                    "size": img.size,
                },
            )]
            
            return IngestionResult(
                document_id=str(file_path),
                contents=contents,
                metadata={"file_type": "image"},
            )
        except Exception as e:
            return IngestionResult(
                document_id=str(file_path),
                contents=[],
                error=str(e),
            )
    
    async def _ingest_image_bytes(self, content: bytes, filename: str) -> IngestionResult:
        """Ingest image from bytes."""
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow required")
        
        try:
            img = Image.open(io.BytesIO(content))
            
            if img.size[0] > self.max_image_size[0] or img.size[1] > self.max_image_size[1]:
                img.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
            
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format=img.format or 'PNG')
            image_bytes = img_byte_arr.getvalue()
            
            contents = [ExtractedContent(
                modality=ModalityType.IMAGE,
                content=image_bytes,
                metadata={"source": "image_bytes", "format": img.format},
            )]
            
            return IngestionResult(
                document_id=filename,
                contents=contents,
                metadata={"file_type": "image"},
            )
        except Exception as e:
            return IngestionResult(document_id=filename, contents=[], error=str(e))
    
    async def _ingest_audio(self, file_path: Path) -> IngestionResult:
        """Ingest audio file and optionally transcribe."""
        contents = []
        
        # Read audio bytes
        with open(file_path, 'rb') as f:
            audio_bytes = f.read()
        
        contents.append(ExtractedContent(
            modality=ModalityType.AUDIO,
            content=audio_bytes,
            metadata={
                "source": "audio_file",
                "filename": file_path.name,
            },
        ))
        
        return IngestionResult(
            document_id=str(file_path),
            contents=contents,
            metadata={"file_type": "audio"},
        )
    
    async def _ingest_audio_bytes(self, content: bytes, filename: str) -> IngestionResult:
        """Ingest audio from bytes."""
        contents = [ExtractedContent(
            modality=ModalityType.AUDIO,
            content=content,
            metadata={"source": "audio_bytes"},
        )]
        
        return IngestionResult(
            document_id=filename,
            contents=contents,
            metadata={"file_type": "audio"},
        )
    
    async def _ingest_video(self, file_path: Path) -> IngestionResult:
        """Extract audio from video file."""
        contents = []
        
        if not self.extract_audio:
            return IngestionResult(
                document_id=str(file_path),
                contents=[],
                metadata={"file_type": "video", "skipped": "audio extraction disabled"},
            )
        
        # Extract audio using ffmpeg
        try:
            import subprocess
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
                audio_path = tmp_audio.name
            
            # Extract audio track
            cmd = [
                'ffmpeg',
                '-i', str(file_path),
                '-vn',  # No video
                '-acodec', 'pcm_s16le',
                '-ar', str(self.audio_sample_rate),
                '-ac', '1',  # Mono
                audio_path,
                '-y',  # Overwrite
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Read extracted audio
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            
            contents.append(ExtractedContent(
                modality=ModalityType.AUDIO,
                content=audio_bytes,
                metadata={
                    "source": "video_audio_track",
                    "sample_rate": self.audio_sample_rate,
                },
            ))
            
            # Cleanup
            Path(audio_path).unlink()
            
        except Exception as e:
            logger.warning(f"Failed to extract audio from video: {e}")
        
        return IngestionResult(
            document_id=str(file_path),
            contents=contents,
            metadata={"file_type": "video"},
        )
    
    async def _ingest_text(self, file_path: Path) -> IngestionResult:
        """Ingest plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            contents = [ExtractedContent(
                modality=ModalityType.TEXT,
                content=text,
                metadata={"source": "text_file"},
            )]
            
            return IngestionResult(
                document_id=str(file_path),
                contents=contents,
                metadata={"file_type": "text"},
            )
        except Exception as e:
            return IngestionResult(
                document_id=str(file_path),
                contents=[],
                error=str(e),
            )
    
    async def _ingest_text_bytes(self, content: bytes, filename: str) -> IngestionResult:
        """Ingest text from bytes."""
        try:
            text = content.decode('utf-8', errors='ignore')
            
            contents = [ExtractedContent(
                modality=ModalityType.TEXT,
                content=text,
                metadata={"source": "text_bytes"},
            )]
            
            return IngestionResult(
                document_id=filename,
                contents=contents,
                metadata={"file_type": "text"},
            )
        except Exception as e:
            return IngestionResult(document_id=filename, contents=[], error=str(e))
    
    async def _ingest_word(self, file_path: Path) -> IngestionResult:
        """Ingest Word document."""
        try:
            import docx
        except ImportError:
            raise ImportError(
                "python-docx required for Word processing. "
                "Install with: pip install python-docx"
            )
        
        try:
            doc = docx.Document(file_path)
            
            contents = []
            
            # Extract paragraphs
            text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
            
            if text:
                contents.append(ExtractedContent(
                    modality=ModalityType.TEXT,
                    content=text,
                    metadata={"source": "word_doc"},
                ))
            
            # Extract images if enabled
            if self.extract_images:
                for rel in doc.part.rels.values():
                    if "image" in rel.target_ref:
                        try:
                            image_bytes = rel.target_part.blob
                            image_bytes = self._resize_image_bytes(
                                image_bytes,
                                self.max_image_size
                            )
                            
                            contents.append(ExtractedContent(
                                modality=ModalityType.IMAGE,
                                content=image_bytes,
                                metadata={"source": "word_image"},
                            ))
                        except Exception as e:
                            logger.warning(f"Failed to extract Word image: {e}")
            
            return IngestionResult(
                document_id=str(file_path),
                contents=contents,
                metadata={"file_type": "docx"},
            )
        except Exception as e:
            return IngestionResult(
                document_id=str(file_path),
                contents=[],
                error=str(e),
            )
    
    def _resize_image_bytes(self, image_bytes: bytes, max_size: tuple) -> bytes:
        """Resize image bytes to max dimensions."""
        try:
            from PIL import Image
            
            img = Image.open(io.BytesIO(image_bytes))
            
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format=img.format or 'PNG')
            return img_byte_arr.getvalue()
        except Exception as e:
            logger.warning(f"Failed to resize image: {e}")
            return image_bytes
