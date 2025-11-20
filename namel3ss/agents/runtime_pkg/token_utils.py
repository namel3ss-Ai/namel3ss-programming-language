from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .data_models import AgentMessage


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.
    
    Uses a simple but accurate heuristic: ~4 characters per token for English text.
    This is conservative and works well for GPT models.
    
    Args:
        text: Text to estimate tokens for
    
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    # ~4 chars per token is a good heuristic for English
    # Add small overhead for special tokens
    return max(1, len(text) // 4 + 1)


def estimate_messages_tokens(messages: List['AgentMessage']) -> int:
    """
    Estimate total token count for a list of messages.
    
    Args:
        messages: List of AgentMessage objects
    
    Returns:
        Estimated total token count
    """
    total = 0
    for msg in messages:
        # Count content tokens
        total += estimate_tokens(msg.content)
        
        # Add overhead for role and structure (~4 tokens per message)
        total += 4
        
        # Add tokens for tool calls/results if present
        if msg.tool_call:
            total += estimate_tokens(json.dumps(msg.tool_call))
        if msg.tool_result:
            total += estimate_tokens(json.dumps(msg.tool_result))
    
    return total
