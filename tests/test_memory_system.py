"""
Tests for the production memory system integration.

Verifies:
- Memory definitions parse correctly
- Memory registry works with different scopes
- Prompts can reference memory
- Chains can read/write memory
- Memory functions work correctly
"""

import pytest
import asyncio
from namel3ss.ast import Memory, Prompt, Chain, ChainStep, PromptArgument, OutputSchema, OutputField, OutputFieldType
from namel3ss.codegen.backend.core.runtime.memory import (
    MemoryRegistry,
    MemoryHandle,
    MemorySpec,
    InMemoryBackend,
    MemoryError,
    MemoryNotFoundError,
)
from namel3ss.codegen.backend.core.runtime.memory_functions import (
    read_memory,
    write_memory,
    append_memory,
    clear_memory,
    set_scope_context,
)
from namel3ss.prompts.runtime import PromptProgram


@pytest.fixture
def memory_registry():
    """Create a fresh memory registry for each test."""
    backend = InMemoryBackend()
    registry = MemoryRegistry(backend)
    return registry


@pytest.fixture
def session_context():
    """Standard session scope context."""
    return {"session_id": "test-session-123", "user_id": "user-456"}


@pytest.mark.asyncio
async def test_memory_registry_basic(memory_registry, session_context):
    """Test basic memory registration and retrieval."""
    # Register a memory
    memory_registry.register({
        "name": "conversation",
        "scope": "session",
        "kind": "list",
        "max_items": 10,
        "config": {},
        "metadata": {"description": "Chat history"},
    })
    
    # Get handle
    handle = memory_registry.get("conversation", scope_context=session_context)
    assert handle is not None
    
    # Initially empty
    content = await handle.read()
    assert content == []
    
    # Append items
    await handle.append({"role": "user", "content": "Hello"})
    await handle.append({"role": "assistant", "content": "Hi there!"})
    
    # Read back
    content = await handle.read()
    assert len(content) == 2
    assert content[0]["role"] == "user"
    assert content[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_memory_max_items_enforcement(memory_registry, session_context):
    """Test that max_items is enforced with proper eviction."""
    memory_registry.register({
        "name": "limited_list",
        "scope": "session",
        "kind": "list",
        "max_items": 3,
        "config": {},
        "metadata": {},
    })
    
    handle = memory_registry.get("limited_list", scope_context=session_context)
    
    # Append more items than limit
    for i in range(5):
        await handle.append(f"item-{i}")
    
    # Should only keep last 3
    content = await handle.read()
    assert len(content) == 3
    assert content == ["item-2", "item-3", "item-4"]


@pytest.mark.asyncio
async def test_memory_scope_isolation(memory_registry):
    """Test that different scopes are isolated."""
    memory_registry.register({
        "name": "user_prefs",
        "scope": "user",
        "kind": "key_value",
        "config": {},
        "metadata": {},
    })
    
    # Two different users
    user1_context = {"user_id": "alice"}
    user2_context = {"user_id": "bob"}
    
    handle1 = memory_registry.get("user_prefs", scope_context=user1_context)
    handle2 = memory_registry.get("user_prefs", scope_context=user2_context)
    
    # Write for alice
    await handle1.write({"theme": "dark", "language": "en"})
    
    # Write for bob
    await handle2.write({"theme": "light", "language": "es"})
    
    # Verify isolation
    alice_prefs = await handle1.read()
    bob_prefs = await handle2.read()
    
    assert alice_prefs["theme"] == "dark"
    assert bob_prefs["theme"] == "light"
    assert alice_prefs["language"] == "en"
    assert bob_prefs["language"] == "es"


@pytest.mark.asyncio
async def test_memory_key_value_operations(memory_registry, session_context):
    """Test key-value memory operations."""
    memory_registry.register({
        "name": "kv_store",
        "scope": "global",
        "kind": "key_value",
        "config": {},
        "metadata": {},
    })
    
    handle = memory_registry.get("kv_store", scope_context=session_context)
    
    # Set individual keys
    await handle.set_key("api_key", "sk-12345")
    await handle.set_key("endpoint", "https://api.example.com")
    
    # Get keys
    api_key = await handle.get_key("api_key")
    endpoint = await handle.get_key("endpoint")
    missing = await handle.get_key("nonexistent", default="default-value")
    
    assert api_key == "sk-12345"
    assert endpoint == "https://api.example.com"
    assert missing == "default-value"


@pytest.mark.asyncio
async def test_memory_functions_global_registry(session_context):
    """Test built-in memory functions with global registry."""
    # Import and set up global registry
    from namel3ss.codegen.backend.core.runtime.memory import get_memory_registry, reset_memory_registry
    
    # Reset to get fresh registry
    reset_memory_registry()
    registry = get_memory_registry()
    
    # Register memory
    registry.register({
        "name": "chat_history",
        "scope": "session",
        "kind": "list",
        "max_items": 50,
        "config": {},
        "metadata": {},
    })
    
    # Set scope context
    set_scope_context(session_context)
    
    # Use built-in functions
    await append_memory("chat_history", {"role": "user", "content": "Test message"})
    await append_memory("chat_history", {"role": "assistant", "content": "Test response"})
    
    # Read back
    history = await read_memory("chat_history")
    assert len(history) == 2
    assert history[0]["content"] == "Test message"
    
    # Clear
    await clear_memory("chat_history")
    history = await read_memory("chat_history")
    assert history == []


@pytest.mark.asyncio
async def test_prompt_with_memory_reference(memory_registry, session_context):
    """Test that prompts can reference memory in templates."""
    # Register memory
    memory_registry.register({
        "name": "conversation_history",
        "scope": "session",
        "kind": "list",
        "max_items": 100,
        "config": {},
        "metadata": {},
    })
    
    # Populate memory
    handle = memory_registry.get("conversation_history", scope_context=session_context)
    await handle.append({"role": "user", "content": "What is 2+2?"})
    await handle.append({"role": "assistant", "content": "2+2 equals 4."})
    
    # Create prompt with memory reference
    prompt = Prompt(
        name="chat_reply",
        model="gpt4",
        template="""You are a helpful assistant.

Conversation history:
{memory.conversation_history}

User: {user_input}
Assistant:""",
        args=[
            PromptArgument(name="user_input", arg_type="string", required=True),
        ],
        output_schema=None,
        parameters={},
        metadata={},
    )
    
    # Create prompt program with memory
    program = PromptProgram(
        definition=prompt,
        memory_registry=memory_registry,
        scope_context=session_context,
    )
    
    # Render with memory
    rendered = await program.render_prompt({"user_input": "What is 3+3?"})
    
    # Verify memory was inserted
    assert "What is 2+2?" in rendered
    assert "2+2 equals 4" in rendered
    assert "What is 3+3?" in rendered


@pytest.mark.asyncio
async def test_prompt_memory_limit(memory_registry, session_context):
    """Test memory reference with limit in prompt."""
    # Register and populate memory
    memory_registry.register({
        "name": "recent_context",
        "scope": "session",
        "kind": "list",
        "config": {},
        "metadata": {},
    })
    
    handle = memory_registry.get("recent_context", scope_context=session_context)
    for i in range(10):
        await handle.append(f"Event {i}")
    
    # Prompt with limited memory reference
    prompt = Prompt(
        name="summarize",
        model="gpt4",
        template="Recent events (last 3):\n{memory.recent_context:3}\n\nSummarize:",
        args=[],
        output_schema=None,
        parameters={},
        metadata={},
    )
    
    program = PromptProgram(
        definition=prompt,
        memory_registry=memory_registry,
        scope_context=session_context,
    )
    
    rendered = await program.render_prompt({})
    
    # Should only have last 3 events
    assert "Event 7" in rendered
    assert "Event 8" in rendered
    assert "Event 9" in rendered
    assert "Event 0" not in rendered


@pytest.mark.asyncio
async def test_memory_not_found_error(memory_registry, session_context):
    """Test error when accessing undefined memory."""
    with pytest.raises(MemoryNotFoundError) as exc_info:
        memory_registry.get("nonexistent", scope_context=session_context)
    
    assert "not found in registry" in str(exc_info.value)


@pytest.mark.asyncio
async def test_memory_type_validation(memory_registry, session_context):
    """Test that memory operations validate types."""
    memory_registry.register({
        "name": "list_mem",
        "scope": "session",
        "kind": "list",
        "config": {},
        "metadata": {},
    })
    
    handle = memory_registry.get("list_mem", scope_context=session_context)
    
    # Writing non-list to list memory should fail
    with pytest.raises(MemoryError) as exc_info:
        await handle.write("not a list")
    
    assert "expected list" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_memory_update_function(memory_registry, session_context):
    """Test memory update with transformation function."""
    memory_registry.register({
        "name": "counter",
        "scope": "session",
        "kind": "key_value",
        "config": {},
        "metadata": {},
    })
    
    handle = memory_registry.get("counter", scope_context=session_context)
    
    # Initialize
    await handle.write({"count": 0, "items": []})
    
    # Update with function
    await handle.update(lambda data: {
        "count": data["count"] + 1,
        "items": data["items"] + ["new_item"],
    })
    
    # Verify
    data = await handle.read()
    assert data["count"] == 1
    assert len(data["items"]) == 1


@pytest.mark.asyncio
async def test_prompt_memory_read_write_methods(memory_registry, session_context):
    """Test PromptProgram read/write memory methods."""
    memory_registry.register({
        "name": "test_memory",
        "scope": "session",
        "kind": "list",
        "config": {},
        "metadata": {},
    })
    
    prompt = Prompt(
        name="test",
        model="gpt4",
        template="Test",
        args=[],
        output_schema=None,
        parameters={},
        metadata={},
    )
    
    program = PromptProgram(
        definition=prompt,
        memory_registry=memory_registry,
        scope_context=session_context,
    )
    
    # Write through program
    await program.write_memory("test_memory", ["item1", "item2"])
    
    # Read through program
    content = await program.read_memory("test_memory")
    assert content == ["item1", "item2"]
    
    # Append through program
    await program.append_memory("test_memory", "item3")
    content = await program.read_memory("test_memory")
    assert len(content) == 3


@pytest.mark.asyncio
async def test_global_scope_shared_across_sessions(memory_registry):
    """Test that global scope is shared across different sessions."""
    memory_registry.register({
        "name": "global_config",
        "scope": "global",
        "kind": "key_value",
        "config": {},
        "metadata": {},
    })
    
    # Two different sessions
    session1 = {"session_id": "session-1"}
    session2 = {"session_id": "session-2"}
    
    handle1 = memory_registry.get("global_config", scope_context=session1)
    handle2 = memory_registry.get("global_config", scope_context=session2)
    
    # Write from session 1
    await handle1.write({"feature_flag": True})
    
    # Read from session 2 - should see same data
    config = await handle2.read()
    assert config["feature_flag"] is True


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
