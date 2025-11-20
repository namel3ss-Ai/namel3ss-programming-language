# Safety Event Persistence - Quick Reference

## Quick Start

```python
from namel3ss.safety.logging import SafetyEventLogger, SafetyEvent, get_global_logger
from namel3ss.safety.runtime import SafetyAction
from datetime import datetime

# Option 1: Use global logger (simplest)
logger = get_global_logger()
await logger.start()

event = SafetyEvent(
    event_id="evt-001",
    timestamp=datetime.utcnow(),
    policy_name="pii_policy",
    action=SafetyAction.REDACT,
    direction="input",
    latency_ms=10.5,
)

await logger.log(event)
await logger.stop()

# Option 2: Custom logger
logger = SafetyEventLogger(
    target_dataset="my_safety_events",
    buffer_size=200,
    flush_interval_seconds=5.0,
    overflow_strategy="block",
)
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_dataset` | `"safety_events"` | Dataset name for persistence |
| `buffer_size` | `100` | Max events before auto-flush |
| `flush_interval_seconds` | `10.0` | Time between periodic flushes |
| `overflow_strategy` | `"drop_oldest"` | `"drop_oldest"`, `"drop_newest"`, `"block"` |
| `adapter_registry` | `AdapterRegistry()` | Custom adapter registry |
| `fallback_path` | `"/tmp/namel3ss_safety_fallback"` | Path for fallback files |

## Overflow Strategies

- **`drop_oldest`**: Remove oldest event when buffer full (default)
- **`drop_newest`**: Ignore new events when buffer full  
- **`block`**: Force flush to make room (may block execution)

## Statistics

```python
stats = logger.get_stats()
print(f"""
  Total Logged: {stats['total_logged']}
  Total Flushed: {stats['total_flushed']}
  Total Dropped: {stats['total_dropped']}
  Current Buffer: {stats['current_buffer_size']}/{stats['max_buffer_size']}
  Dataset: {stats['target_dataset']}
""")
```

## Advanced: Custom Sink

```python
from namel3ss.safety.persistence import SafetyEventSink

sink = SafetyEventSink(
    dataset_name="prod_safety",
    adapter_registry=my_registry,
    fallback_path="/var/log/safety",
    max_retries=5,
    retry_delay_seconds=2.0,
)

logger = SafetyEventLogger(
    event_sink=sink,
    buffer_size=500,
)
```

## Database Writer (Optional)

```python
from namel3ss.safety.persistence import SafetyEventDatabaseWriter

writer = SafetyEventDatabaseWriter(
    connection_string="postgresql://localhost/safety",
    table_name="safety_events",
)

await writer.write_events(rows)
```

## Object Store Writer (Optional)

```python
from namel3ss.safety.persistence import SafetyEventObjectStoreWriter

writer = SafetyEventObjectStoreWriter(
    bucket_name="my-safety-bucket",
    prefix="events/safety",
    format="parquet",  # or "json", "csv"
)

await writer.write_events(rows)
```

## Error Handling

The logger **never crashes your application**:

- Failed writes are retried (default 3 times)
- After max retries, events written to fallback files
- All errors logged but not raised
- Statistics track dropped events

## Fallback Files

When all persistence fails:
- Files written to: `{fallback_path}/safety_events_YYYYMMDD_HHMMSS_{count}.json`
- JSON format for easy recovery
- Critical log entries generated
- Can be reprocessed later

## Integration with Datasets

Events automatically persist to configured datasets via adapters:

```python
# CSV adapter
adapter_registry.register("safety_events", CSVAdapter("/data/safety.csv"))

# SQL adapter  
adapter_registry.register("safety_events", SQLAdapter(engine, "safety_events"))

# JSON adapter
adapter_registry.register("safety_events", JSONAdapter("/data/safety.json"))
```

## Best Practices

1. **Use global logger** for simple cases
2. **Configure buffer size** based on event volume
3. **Set fallback path** for production (persistent storage)
4. **Monitor statistics** to detect issues
5. **Use "block" strategy** for critical events
6. **Flush on shutdown** with `await logger.stop()`

## Common Patterns

### High-Volume Logging
```python
logger = SafetyEventLogger(
    buffer_size=1000,
    flush_interval_seconds=5.0,
    overflow_strategy="drop_oldest",
)
```

### Critical Events Only
```python
logger = SafetyEventLogger(
    buffer_size=10,
    flush_interval_seconds=1.0,
    overflow_strategy="block",
)
```

### Testing/Development
```python
logger = SafetyEventLogger(
    buffer_size=5,
    flush_interval_seconds=0.5,
    fallback_path="/tmp/test_safety",
)
```

## Troubleshooting

**Events not persisting?**
- Check adapter is registered: `adapter_registry.get("safety_events")`
- Check logs for retry failures
- Look for fallback files in fallback_path

**Buffer filling up?**
- Increase buffer_size
- Decrease flush_interval_seconds
- Check adapter write performance

**Memory growing?**
- Verify flush_interval_seconds isn't too high
- Check overflow_strategy is set
- Monitor stats['current_buffer_size']

## Testing

```python
# Run persistence tests
pytest tests/safety/test_persistence.py -v

# Run integration tests
pytest tests/safety/test_safety_event_logging.py -v

# Run specific test
pytest tests/safety/test_persistence.py::TestSafetyEventSink::test_write_events_uses_insert_method -v
```

## Dependencies

Core functionality (no additional deps):
- Python 3.10+
- asyncio

Optional (for specific writers):
- `sqlalchemy` - for SafetyEventDatabaseWriter
- `pyarrow` - for Parquet format in ObjectStoreWriter
- `boto3` - for AWS S3 (not yet implemented)
- `google-cloud-storage` - for GCS (not yet implemented)

## Architecture

```
SafetyEventLogger (logging.py)
    ↓
SafetyEventSink (persistence.py)
    ↓
AdapterRegistry (logic_adapters.py)
    ↓
DatasetAdapter (CSV/JSON/SQL/etc)
    ↓
Persistent Storage
```

## Performance Characteristics

- **Buffering**: O(1) append to buffer
- **Flush**: O(n) where n = buffer size
- **Retry**: Exponential backoff (1s, 2s, 4s...)
- **Fallback**: Async file write, doesn't block
- **Memory**: Max = buffer_size * ~1KB per event

## Safety Guarantees

✅ Non-blocking - never blocks main execution  
✅ Never crashes - all exceptions caught  
✅ Data loss prevention - fallback persistence  
✅ Statistics - visibility into operations  
✅ Thread-safe - async lock on buffer  

## Support

- Documentation: `SAFETY_PERSISTENCE_IMPLEMENTATION.md`
- Tests: `tests/safety/test_*.py`
- Code: `namel3ss/safety/persistence.py`, `logging.py`
