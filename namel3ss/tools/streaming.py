"""
Streaming support for tool adapters.

This module provides utilities and base classes for implementing streaming tools
that return results incrementally (e.g., LLM completions, file processing,
real-time data feeds).

Key Components:
    - StreamBuffer: Buffer for accumulating chunks
    - StreamAggregator: Aggregate chunks into final result
    - rate_limit_stream: Rate limiting for streams
    - batch_stream: Batch chunks for efficiency

Architecture:
    Tool implements StreamingToolAdapter protocol
    ↓
    invoke_stream() yields chunks (TChunk models)
    ↓
    Chunks buffered/aggregated as needed
    ↓
    Final result constructed from chunks

Example:
    from namel3ss.tools.streaming import StreamBuffer, StreamingContext
    from namel3ss.tools.schemas import ToolChunkModel
    from pydantic import Field
    
    class TokenChunk(ToolChunkModel):
        token: str = Field(..., description="Generated token")
    
    async def process_stream():
        buffer = StreamBuffer[TokenChunk]()
        
        async for chunk in tool.invoke_stream(input, context):
            buffer.add(chunk)
            if buffer.should_flush():
                result = buffer.flush()
                yield result

Thread Safety:
    StreamBuffer is NOT thread-safe. Use per-coroutine instances.
    StreamAggregator IS thread-safe for concurrent chunk processing.

Performance:
    - Chunks should be small (low latency)
    - Buffer/batch for network efficiency
    - Use backpressure to prevent memory overflow
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
)

from namel3ss.tools.schemas import ToolChunkModel


T = TypeVar("T", bound=ToolChunkModel)
TResult = TypeVar("TResult")


@dataclass
class StreamingContext:
    """
    Context for streaming operations.
    
    Provides configuration and state for streaming:
    - Buffer size limits
    - Flush intervals
    - Backpressure handling
    - Progress tracking
    
    Attributes:
        max_buffer_size: Maximum chunks to buffer before flushing
        flush_interval: Time in seconds between flushes
        enable_backpressure: Whether to apply backpressure
        max_queue_size: Maximum queue size before blocking
    
    Example:
        >>> context = StreamingContext(
        ...     max_buffer_size=100,
        ...     flush_interval=1.0,
        ...     enable_backpressure=True
        ... )
    """
    
    max_buffer_size: int = 100
    flush_interval: float = 1.0
    enable_backpressure: bool = True
    max_queue_size: int = 1000
    
    # Runtime state
    _chunks_processed: int = field(default=0, init=False)
    _bytes_processed: int = field(default=0, init=False)
    _start_time: float = field(default_factory=time.time, init=False)
    
    def increment_chunks(self, count: int = 1) -> None:
        """Increment chunks processed counter."""
        self._chunks_processed += count
    
    def increment_bytes(self, count: int) -> None:
        """Increment bytes processed counter."""
        self._bytes_processed += count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        elapsed = time.time() - self._start_time
        return {
            "chunks_processed": self._chunks_processed,
            "bytes_processed": self._bytes_processed,
            "elapsed_seconds": elapsed,
            "chunks_per_second": self._chunks_processed / elapsed if elapsed > 0 else 0,
            "bytes_per_second": self._bytes_processed / elapsed if elapsed > 0 else 0,
        }


class StreamBuffer(Generic[T]):
    """
    Buffer for accumulating stream chunks.
    
    Provides buffering with automatic flushing based on:
    - Buffer size limits
    - Time intervals
    - Custom flush conditions
    
    Type-safe with Generic[T] for chunk type.
    
    Example:
        >>> buffer = StreamBuffer[TokenChunk](max_size=100)
        >>> buffer.add(TokenChunk(token="hello"))
        >>> buffer.add(TokenChunk(token=" "))
        >>> buffer.add(TokenChunk(token="world"))
        >>> 
        >>> if buffer.should_flush():
        ...     chunks = buffer.flush()
        ...     text = "".join(c.token for c in chunks)
        ...     print(text)  # "hello world"
    
    Flush Conditions:
        - Buffer reaches max_size
        - Time since last flush exceeds flush_interval
        - Final chunk received (is_final=True)
        - Custom condition via should_flush_fn
    
    Thread Safety:
        NOT thread-safe. Use one instance per coroutine.
    """
    
    def __init__(
        self,
        max_size: int = 100,
        flush_interval: float = 1.0,
        should_flush_fn: Optional[Callable[[List[T]], bool]] = None,
    ):
        """
        Initialize buffer.
        
        Args:
            max_size: Maximum chunks before auto-flush
            flush_interval: Seconds between flushes
            should_flush_fn: Custom flush condition function
        """
        self._buffer: Deque[T] = deque()
        self._max_size = max_size
        self._flush_interval = flush_interval
        self._should_flush_fn = should_flush_fn
        self._last_flush_time = time.time()
    
    def add(self, chunk: T) -> None:
        """
        Add chunk to buffer.
        
        Args:
            chunk: Chunk to add
        """
        self._buffer.append(chunk)
    
    def should_flush(self) -> bool:
        """
        Check if buffer should be flushed.
        
        Returns:
            True if flush conditions met
        """
        # Size limit
        if len(self._buffer) >= self._max_size:
            return True
        
        # Time interval
        if time.time() - self._last_flush_time >= self._flush_interval:
            return True
        
        # Final chunk
        if self._buffer and self._buffer[-1].is_final:
            return True
        
        # Custom condition
        if self._should_flush_fn and self._should_flush_fn(list(self._buffer)):
            return True
        
        return False
    
    def flush(self) -> List[T]:
        """
        Flush buffer and return chunks.
        
        Returns:
            List of buffered chunks
        """
        chunks = list(self._buffer)
        self._buffer.clear()
        self._last_flush_time = time.time()
        return chunks
    
    def peek(self) -> List[T]:
        """
        Peek at buffer without flushing.
        
        Returns:
            List of buffered chunks (copy)
        """
        return list(self._buffer)
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self._buffer)
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self._buffer) == 0


class StreamAggregator(Generic[T, TResult]):
    """
    Aggregate stream chunks into final result.
    
    Provides:
    - Chunk accumulation
    - Result construction
    - Progress tracking
    - Error handling
    
    Type-safe: Generic[T, TResult]
    - T: Chunk type (extends ToolChunkModel)
    - TResult: Final result type
    
    Example:
        >>> from pydantic import BaseModel
        >>> 
        >>> class TokenChunk(ToolChunkModel):
        ...     token: str
        >>> 
        >>> class TextResult(BaseModel):
        ...     text: str
        ...     token_count: int
        >>> 
        >>> def aggregate_tokens(chunks: List[TokenChunk]) -> TextResult:
        ...     text = "".join(c.token for c in chunks)
        ...     return TextResult(text=text, token_count=len(chunks))
        >>> 
        >>> aggregator = StreamAggregator[TokenChunk, TextResult](
        ...     aggregate_fn=aggregate_tokens
        ... )
        >>> 
        >>> async for chunk in stream:
        ...     aggregator.add(chunk)
        >>> 
        >>> result = aggregator.get_result()
        >>> print(result.text)
    
    Thread Safety:
        Thread-safe for concurrent chunk addition.
        Use asyncio.Lock for async environments.
    """
    
    def __init__(
        self,
        aggregate_fn: Callable[[List[T]], TResult],
        max_chunks: Optional[int] = None,
    ):
        """
        Initialize aggregator.
        
        Args:
            aggregate_fn: Function to construct result from chunks
            max_chunks: Maximum chunks to store (None = unlimited)
        """
        self._chunks: List[T] = []
        self._aggregate_fn = aggregate_fn
        self._max_chunks = max_chunks
        self._lock = asyncio.Lock()
    
    async def add(self, chunk: T) -> None:
        """
        Add chunk (thread-safe).
        
        Args:
            chunk: Chunk to add
        
        Raises:
            ValueError: If max_chunks exceeded
        """
        async with self._lock:
            if self._max_chunks and len(self._chunks) >= self._max_chunks:
                raise ValueError(f"Max chunks ({self._max_chunks}) exceeded")
            self._chunks.append(chunk)
    
    def add_sync(self, chunk: T) -> None:
        """
        Add chunk (sync version).
        
        Args:
            chunk: Chunk to add
        """
        if self._max_chunks and len(self._chunks) >= self._max_chunks:
            raise ValueError(f"Max chunks ({self._max_chunks}) exceeded")
        self._chunks.append(chunk)
    
    async def get_result(self) -> TResult:
        """
        Get aggregated result (thread-safe).
        
        Returns:
            Aggregated result
        """
        async with self._lock:
            return self._aggregate_fn(self._chunks)
    
    def get_result_sync(self) -> TResult:
        """
        Get aggregated result (sync version).
        
        Returns:
            Aggregated result
        """
        return self._aggregate_fn(self._chunks)
    
    def get_chunks(self) -> List[T]:
        """Get all chunks (copy)."""
        return self._chunks.copy()
    
    def chunk_count(self) -> int:
        """Get number of chunks."""
        return len(self._chunks)
    
    def clear(self) -> None:
        """Clear all chunks."""
        self._chunks.clear()


# Utility functions for stream manipulation

async def rate_limit_stream(
    stream: AsyncIterator[T],
    max_per_second: float,
) -> AsyncIterator[T]:
    """
    Rate limit a stream to max items per second.
    
    Args:
        stream: Input stream
        max_per_second: Maximum items per second
    
    Yields:
        Rate-limited chunks
    
    Example:
        >>> limited = rate_limit_stream(stream, max_per_second=10)
        >>> async for chunk in limited:
        ...     process(chunk)
    """
    interval = 1.0 / max_per_second
    last_yield = time.time()
    
    async for chunk in stream:
        now = time.time()
        elapsed = now - last_yield
        
        if elapsed < interval:
            await asyncio.sleep(interval - elapsed)
        
        yield chunk
        last_yield = time.time()


async def batch_stream(
    stream: AsyncIterator[T],
    batch_size: int,
    timeout: Optional[float] = None,
) -> AsyncIterator[List[T]]:
    """
    Batch stream chunks into lists.
    
    Args:
        stream: Input stream
        batch_size: Chunks per batch
        timeout: Max seconds to wait for batch (None = wait forever)
    
    Yields:
        Batches of chunks
    
    Example:
        >>> batched = batch_stream(stream, batch_size=10, timeout=1.0)
        >>> async for batch in batched:
        ...     process_batch(batch)  # process 10 chunks at once
    """
    batch: List[T] = []
    batch_start = time.time()
    
    async for chunk in stream:
        batch.append(chunk)
        
        # Yield when batch full
        if len(batch) >= batch_size:
            yield batch
            batch = []
            batch_start = time.time()
        
        # Yield on timeout
        elif timeout and (time.time() - batch_start >= timeout):
            if batch:
                yield batch
                batch = []
            batch_start = time.time()
    
    # Yield remaining
    if batch:
        yield batch


async def filter_stream(
    stream: AsyncIterator[T],
    predicate: Callable[[T], bool],
) -> AsyncIterator[T]:
    """
    Filter stream chunks by predicate.
    
    Args:
        stream: Input stream
        predicate: Filter function
    
    Yields:
        Chunks matching predicate
    
    Example:
        >>> def is_complete(chunk):
        ...     return len(chunk.token) > 0
        >>> 
        >>> filtered = filter_stream(stream, is_complete)
        >>> async for chunk in filtered:
        ...     print(chunk)
    """
    async for chunk in stream:
        if predicate(chunk):
            yield chunk


async def map_stream(
    stream: AsyncIterator[T],
    mapper: Callable[[T], T],
) -> AsyncIterator[T]:
    """
    Map stream chunks through transformation.
    
    Args:
        stream: Input stream
        mapper: Transformation function
    
    Yields:
        Transformed chunks
    
    Example:
        >>> def uppercase_token(chunk):
        ...     chunk.token = chunk.token.upper()
        ...     return chunk
        >>> 
        >>> mapped = map_stream(stream, uppercase_token)
        >>> async for chunk in mapped:
        ...     print(chunk.token)
    """
    async for chunk in stream:
        yield mapper(chunk)


async def collect_stream(stream: AsyncIterator[T]) -> List[T]:
    """
    Collect all chunks from stream into list.
    
    Args:
        stream: Input stream
    
    Returns:
        List of all chunks
    
    Example:
        >>> chunks = await collect_stream(stream)
        >>> print(f"Collected {len(chunks)} chunks")
    
    Warning:
        Can consume unbounded memory for infinite streams.
    """
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    return chunks


async def take_stream(
    stream: AsyncIterator[T],
    count: int,
) -> AsyncIterator[T]:
    """
    Take first N chunks from stream.
    
    Args:
        stream: Input stream
        count: Number of chunks to take
    
    Yields:
        First N chunks
    
    Example:
        >>> first_10 = take_stream(stream, 10)
        >>> async for chunk in first_10:
        ...     print(chunk)
    """
    taken = 0
    async for chunk in stream:
        if taken >= count:
            break
        yield chunk
        taken += 1


async def merge_streams(
    *streams: AsyncIterator[T],
) -> AsyncIterator[T]:
    """
    Merge multiple streams into single stream.
    
    Args:
        *streams: Input streams
    
    Yields:
        Chunks from all streams (in arrival order)
    
    Example:
        >>> stream1 = get_stream_1()
        >>> stream2 = get_stream_2()
        >>> merged = merge_streams(stream1, stream2)
        >>> async for chunk in merged:
        ...     print(chunk)
    
    Note:
        Uses asyncio.Queue for merging.
        All streams run concurrently.
    """
    queue: asyncio.Queue[Optional[T]] = asyncio.Queue()
    
    async def consume(stream: AsyncIterator[T]) -> None:
        async for chunk in stream:
            await queue.put(chunk)
    
    # Start all consumers
    tasks = [asyncio.create_task(consume(stream)) for stream in streams]
    
    # Monitor completion
    async def monitor() -> None:
        await asyncio.gather(*tasks)
        await queue.put(None)  # Sentinel
    
    asyncio.create_task(monitor())
    
    # Yield from queue
    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        yield chunk
