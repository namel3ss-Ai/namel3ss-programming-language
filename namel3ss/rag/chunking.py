"""Text chunking utilities for RAG indexing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class TextChunk:
    """A chunk of text with metadata."""
    content: str
    start_pos: int
    end_pos: int
    chunk_id: str
    metadata: Dict[str, Any]
    
    def __len__(self) -> int:
        return len(self.content)


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    metadata: Optional[Dict[str, Any]] = None,
    separator: str = "\n\n",
) -> List[TextChunk]:
    """
    Chunk text into overlapping segments.
    
    Args:
        text: Text to chunk
        chunk_size: Target size for each chunk in characters
        overlap: Number of overlapping characters between chunks
        metadata: Optional metadata to attach to each chunk
        separator: Preferred separator for splitting (default: paragraph breaks)
        
    Returns:
        List of TextChunk objects
        
    Raises:
        ValueError: If overlap >= chunk_size or negative values provided
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    
    metadata = metadata or {}
    chunks = []
    
    if not text or not text.strip():
        return chunks
    
    # Split by preferred separator first
    if separator and separator in text:
        segments = text.split(separator)
    else:
        # Fall back to sentence splitting
        segments = _split_into_sentences(text)
    
    current_chunk = []
    current_size = 0
    start_pos = 0
    
    for i, segment in enumerate(segments):
        segment = segment.strip()
        if not segment:
            continue
        
        segment_size = len(segment)
        
        # If single segment exceeds chunk_size, split it further
        if segment_size > chunk_size:
            # First, add current chunk if any
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(TextChunk(
                    content=chunk_text,
                    start_pos=start_pos,
                    end_pos=start_pos + len(chunk_text),
                    chunk_id=f"chunk_{len(chunks)}",
                    metadata={**metadata, "chunk_index": len(chunks)}
                ))
                current_chunk = []
                current_size = 0
            
            # Split large segment
            sub_chunks = _split_large_segment(segment, chunk_size, overlap)
            for sub_chunk in sub_chunks:
                chunks.append(TextChunk(
                    content=sub_chunk,
                    start_pos=start_pos,
                    end_pos=start_pos + len(sub_chunk),
                    chunk_id=f"chunk_{len(chunks)}",
                    metadata={**metadata, "chunk_index": len(chunks)}
                ))
                start_pos += len(sub_chunk) - overlap if overlap else len(sub_chunk)
            
            continue
        
        # Check if adding this segment exceeds chunk_size
        if current_size + segment_size + (1 if current_chunk else 0) > chunk_size:
            # Create chunk from current segments
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(TextChunk(
                    content=chunk_text,
                    start_pos=start_pos,
                    end_pos=start_pos + len(chunk_text),
                    chunk_id=f"chunk_{len(chunks)}",
                    metadata={**metadata, "chunk_index": len(chunks)}
                ))
                
                # Start new chunk with overlap
                if overlap > 0:
                    # Keep last part for overlap
                    overlap_text = chunk_text[-overlap:]
                    current_chunk = [overlap_text, segment]
                    current_size = len(overlap_text) + segment_size + 1
                    start_pos = start_pos + len(chunk_text) - overlap
                else:
                    current_chunk = [segment]
                    current_size = segment_size
                    start_pos = start_pos + len(chunk_text)
            else:
                current_chunk = [segment]
                current_size = segment_size
        else:
            # Add segment to current chunk
            current_chunk.append(segment)
            current_size += segment_size + (1 if len(current_chunk) > 1 else 0)
    
    # Add final chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(TextChunk(
            content=chunk_text,
            start_pos=start_pos,
            end_pos=start_pos + len(chunk_text),
            chunk_id=f"chunk_{len(chunks)}",
            metadata={**metadata, "chunk_index": len(chunks)}
        ))
    
    return chunks


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using simple heuristics."""
    # Simple sentence splitter - can be improved with nltk/spacy
    pattern = r'(?<=[.!?])\s+'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def _split_large_segment(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split a large segment that exceeds chunk_size."""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        
        # Try to break at word boundary
        if end < text_len:
            # Look for last space before end
            last_space = text.rfind(' ', start, end)
            if last_space > start:
                end = last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap if overlap and end < text_len else end
        
        # Avoid infinite loop on very long words
        if chunks and start <= (end - len(chunks[-1])):
            start = end
    
    return chunks


__all__ = [
    "TextChunk",
    "chunk_text",
]
