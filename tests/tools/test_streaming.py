"""Unit tests for Tool Adapter Framework streaming support."""

import asyncio
import pytest
from typing import AsyncIterator, List

from namel3ss.tools.streaming import (
    StreamBuffer,
    StreamAggregator,
    StreamingContext,
    rate_limit_stream,
    batch_stream,
    filter_stream,
    map_stream,
    collect_stream,
    take_stream,
    merge_streams,
)
from namel3ss.tools.schemas import ToolChunkModel
from pydantic import Field


# Test chunk model

class TestChunk(ToolChunkModel):
    """Test chunk for streaming."""
    token: str = Field(..., description="Token string")
    value: int = Field(0, description="Token value")


# Helper functions

async def create_stream(tokens: List[str], delay: float = 0.01) -> AsyncIterator[TestChunk]:
    """Create test stream."""
    for i, token in enumerate(tokens):
        chunk = TestChunk(
            token=token,
            value=i,
            sequence=i,
            is_final=(i == len(tokens) - 1),
        )
        yield chunk
        await asyncio.sleep(delay)


# Tests

@pytest.mark.asyncio
async def test_streaming_context():
    """Test StreamingContext."""
    context = StreamingContext(
        max_buffer_size=100,
        flush_interval=1.0,
        enable_backpressure=True,
        max_queue_size=1000,
    )
    
    assert context.max_buffer_size == 100
    assert context.flush_interval == 1.0
    assert context.enable_backpressure is True
    assert context.max_queue_size == 1000
    
    # Test counters
    context.increment_chunks(5)
    context.increment_bytes(100)
    
    stats = context.get_stats()
    assert stats["chunks_processed"] == 5
    assert stats["bytes_processed"] == 100
    assert "elapsed_seconds" in stats
    assert "chunks_per_second" in stats


@pytest.mark.asyncio
async def test_stream_buffer_basic():
    """Test StreamBuffer basic functionality."""
    buffer = StreamBuffer[TestChunk](max_size=5, flush_interval=10.0)
    
    # Add chunks
    for i in range(3):
        chunk = TestChunk(token=f"token{i}", value=i)
        buffer.add(chunk)
    
    assert buffer.size() == 3
    assert not buffer.is_empty()
    
    # Peek without flushing
    peeked = buffer.peek()
    assert len(peeked) == 3
    assert buffer.size() == 3  # Still buffered
    
    # Flush
    flushed = buffer.flush()
    assert len(flushed) == 3
    assert buffer.size() == 0
    assert buffer.is_empty()


@pytest.mark.asyncio
async def test_stream_buffer_auto_flush_size():
    """Test StreamBuffer auto-flush on size limit."""
    buffer = StreamBuffer[TestChunk](max_size=3, flush_interval=10.0)
    
    # Add chunks
    for i in range(2):
        buffer.add(TestChunk(token=f"token{i}", value=i))
    
    assert not buffer.should_flush()  # Not at limit
    
    # Add one more to reach limit
    buffer.add(TestChunk(token="token2", value=2))
    assert buffer.should_flush()  # At limit


@pytest.mark.asyncio
async def test_stream_buffer_auto_flush_final():
    """Test StreamBuffer auto-flush on final chunk."""
    buffer = StreamBuffer[TestChunk](max_size=10, flush_interval=10.0)
    
    buffer.add(TestChunk(token="token1", value=1, is_final=False))
    assert not buffer.should_flush()
    
    buffer.add(TestChunk(token="token2", value=2, is_final=True))
    assert buffer.should_flush()  # Final chunk


@pytest.mark.asyncio
async def test_stream_buffer_custom_flush():
    """Test StreamBuffer with custom flush condition."""
    def should_flush(chunks: List[TestChunk]) -> bool:
        # Flush when sum of values exceeds 10
        return sum(c.value for c in chunks) > 10
    
    buffer = StreamBuffer[TestChunk](
        max_size=100,
        flush_interval=10.0,
        should_flush_fn=should_flush,
    )
    
    buffer.add(TestChunk(token="t1", value=5))
    assert not buffer.should_flush()
    
    buffer.add(TestChunk(token="t2", value=6))
    assert buffer.should_flush()  # Sum = 11


@pytest.mark.asyncio
async def test_stream_aggregator_basic():
    """Test StreamAggregator basic functionality."""
    def aggregate_fn(chunks: List[TestChunk]) -> str:
        return " ".join(c.token for c in chunks)
    
    aggregator = StreamAggregator[TestChunk, str](aggregate_fn=aggregate_fn)
    
    # Add chunks
    await aggregator.add(TestChunk(token="hello", value=1))
    await aggregator.add(TestChunk(token="world", value=2))
    
    assert aggregator.chunk_count() == 2
    
    # Get result
    result = await aggregator.get_result()
    assert result == "hello world"


@pytest.mark.asyncio
async def test_stream_aggregator_max_chunks():
    """Test StreamAggregator max_chunks limit."""
    def aggregate_fn(chunks: List[TestChunk]) -> int:
        return len(chunks)
    
    aggregator = StreamAggregator[TestChunk, int](
        aggregate_fn=aggregate_fn,
        max_chunks=3,
    )
    
    # Add up to limit
    await aggregator.add(TestChunk(token="t1", value=1))
    await aggregator.add(TestChunk(token="t2", value=2))
    await aggregator.add(TestChunk(token="t3", value=3))
    
    # Should fail on exceeding limit
    with pytest.raises(ValueError) as exc_info:
        await aggregator.add(TestChunk(token="t4", value=4))
    assert "Max chunks" in str(exc_info.value)


@pytest.mark.asyncio
async def test_stream_aggregator_sync():
    """Test StreamAggregator sync methods."""
    def aggregate_fn(chunks: List[TestChunk]) -> int:
        return len(chunks)
    
    aggregator = StreamAggregator[TestChunk, int](aggregate_fn=aggregate_fn)
    
    # Sync add
    aggregator.add_sync(TestChunk(token="t1", value=1))
    aggregator.add_sync(TestChunk(token="t2", value=2))
    
    # Sync result
    result = aggregator.get_result_sync()
    assert result == 2


@pytest.mark.asyncio
async def test_rate_limit_stream():
    """Test rate_limit_stream()."""
    tokens = ["a", "b", "c", "d", "e"]
    stream = create_stream(tokens, delay=0.0)  # No delay
    
    limited = rate_limit_stream(stream, max_per_second=10.0)
    
    import time
    start = time.time()
    chunks = []
    async for chunk in limited:
        chunks.append(chunk)
    elapsed = time.time() - start
    
    assert len(chunks) == 5
    # Should take at least 0.4 seconds (5 items at 10/sec = 0.4s minimum)
    assert elapsed >= 0.4


@pytest.mark.asyncio
async def test_batch_stream():
    """Test batch_stream()."""
    tokens = ["a", "b", "c", "d", "e", "f", "g"]
    stream = create_stream(tokens, delay=0.01)
    
    batched = batch_stream(stream, batch_size=3)
    
    batches = []
    async for batch in batched:
        batches.append(batch)
    
    assert len(batches) == 3  # 3 + 3 + 1
    assert len(batches[0]) == 3
    assert len(batches[1]) == 3
    assert len(batches[2]) == 1


@pytest.mark.asyncio
async def test_batch_stream_timeout():
    """Test batch_stream() with timeout."""
    # Slow stream
    async def slow_stream() -> AsyncIterator[TestChunk]:
        for i in range(5):
            yield TestChunk(token=f"t{i}", value=i)
            await asyncio.sleep(0.2)  # 200ms delay
    
    batched = batch_stream(slow_stream(), batch_size=10, timeout=0.3)
    
    batches = []
    async for batch in batched:
        batches.append(batch)
    
    # Should create batches due to timeout (not size)
    assert len(batches) > 1


@pytest.mark.asyncio
async def test_filter_stream():
    """Test filter_stream()."""
    tokens = ["a", "b", "c", "d", "e"]
    stream = create_stream(tokens, delay=0.01)
    
    # Filter: keep only even indices
    filtered = filter_stream(stream, lambda c: c.value % 2 == 0)
    
    chunks = []
    async for chunk in filtered:
        chunks.append(chunk)
    
    assert len(chunks) == 3  # Indices 0, 2, 4
    assert chunks[0].token == "a"
    assert chunks[1].token == "c"
    assert chunks[2].token == "e"


@pytest.mark.asyncio
async def test_map_stream():
    """Test map_stream()."""
    tokens = ["a", "b", "c"]
    stream = create_stream(tokens, delay=0.01)
    
    # Map: uppercase tokens
    def uppercase(chunk: TestChunk) -> TestChunk:
        chunk.token = chunk.token.upper()
        return chunk
    
    mapped = map_stream(stream, uppercase)
    
    chunks = []
    async for chunk in mapped:
        chunks.append(chunk)
    
    assert len(chunks) == 3
    assert chunks[0].token == "A"
    assert chunks[1].token == "B"
    assert chunks[2].token == "C"


@pytest.mark.asyncio
async def test_collect_stream():
    """Test collect_stream()."""
    tokens = ["a", "b", "c", "d", "e"]
    stream = create_stream(tokens, delay=0.01)
    
    chunks = await collect_stream(stream)
    
    assert len(chunks) == 5
    assert all(isinstance(c, TestChunk) for c in chunks)
    assert [c.token for c in chunks] == tokens


@pytest.mark.asyncio
async def test_take_stream():
    """Test take_stream()."""
    tokens = ["a", "b", "c", "d", "e"]
    stream = create_stream(tokens, delay=0.01)
    
    limited = take_stream(stream, count=3)
    
    chunks = []
    async for chunk in limited:
        chunks.append(chunk)
    
    assert len(chunks) == 3
    assert chunks[0].token == "a"
    assert chunks[1].token == "b"
    assert chunks[2].token == "c"


@pytest.mark.asyncio
async def test_merge_streams():
    """Test merge_streams()."""
    stream1 = create_stream(["a", "b", "c"], delay=0.02)
    stream2 = create_stream(["x", "y", "z"], delay=0.02)
    
    merged = merge_streams(stream1, stream2)
    
    chunks = []
    async for chunk in merged:
        chunks.append(chunk)
    
    assert len(chunks) == 6
    # Tokens should be mixed from both streams
    tokens = [c.token for c in chunks]
    assert "a" in tokens
    assert "x" in tokens


@pytest.mark.asyncio
async def test_stream_buffer_clear():
    """Test StreamBuffer clear()."""
    buffer = StreamBuffer[TestChunk]()
    
    buffer.add(TestChunk(token="t1", value=1))
    buffer.add(TestChunk(token="t2", value=2))
    
    assert buffer.size() == 2
    
    buffer.flush()
    assert buffer.is_empty()


@pytest.mark.asyncio
async def test_stream_aggregator_get_chunks():
    """Test StreamAggregator get_chunks()."""
    def aggregate_fn(chunks: List[TestChunk]) -> str:
        return ""
    
    aggregator = StreamAggregator[TestChunk, str](aggregate_fn=aggregate_fn)
    
    await aggregator.add(TestChunk(token="t1", value=1))
    await aggregator.add(TestChunk(token="t2", value=2))
    
    chunks = aggregator.get_chunks()
    assert len(chunks) == 2
    assert chunks[0].token == "t1"
    assert chunks[1].token == "t2"


@pytest.mark.asyncio
async def test_stream_aggregator_clear():
    """Test StreamAggregator clear()."""
    def aggregate_fn(chunks: List[TestChunk]) -> int:
        return len(chunks)
    
    aggregator = StreamAggregator[TestChunk, int](aggregate_fn=aggregate_fn)
    
    await aggregator.add(TestChunk(token="t1", value=1))
    await aggregator.add(TestChunk(token="t2", value=2))
    
    assert aggregator.chunk_count() == 2
    
    aggregator.clear()
    assert aggregator.chunk_count() == 0


@pytest.mark.asyncio
async def test_complex_stream_pipeline():
    """Test complex stream processing pipeline."""
    tokens = list("abcdefghij")
    stream = create_stream(tokens, delay=0.01)
    
    # Pipeline: filter (keep even) -> map (uppercase) -> take 3
    filtered = filter_stream(stream, lambda c: c.value % 2 == 0)
    mapped = map_stream(filtered, lambda c: TestChunk(
        token=c.token.upper(),
        value=c.value,
        sequence=c.sequence,
        is_final=c.is_final,
    ))
    limited = take_stream(mapped, count=3)
    
    chunks = []
    async for chunk in limited:
        chunks.append(chunk)
    
    assert len(chunks) == 3
    assert chunks[0].token == "A"  # Index 0
    assert chunks[1].token == "C"  # Index 2
    assert chunks[2].token == "E"  # Index 4
