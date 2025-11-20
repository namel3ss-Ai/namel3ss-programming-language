from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import List, Optional

from namel3ss.ast.agents import MemoryConfig
from namel3ss.observability.metrics import record_metric

from .data_models import AgentMessage
from .token_utils import estimate_messages_tokens

logger = logging.getLogger(__name__)


class BaseMemory:
    """
    Production-grade agent memory with rolling summarization support.
    
    Supports multiple memory policies:
    - none: No message history
    - full_history: All messages
    - conversation_window: Sliding window of recent messages
    - summary: Incremental summarization of older messages + recent window
    
    For summary policy:
    - Maintains a rolling summary of older conversation segments
    - Keeps recent messages in full for immediate context
    - Triggers summarization based on message count or token thresholds
    - Falls back to windowed memory if summarization fails
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig(policy="conversation_window")
        self.messages: List[AgentMessage] = []
        
        # Summarization state
        self.summary: Optional[str] = None
        self.last_summarized_index: int = 0
        self._summarizer = None
        self._summarization_failures: int = 0
        self._last_summarization_attempt: float = 0.0
        
        # Configuration defaults for summary policy
        self._max_summary_tokens = self.config.max_summary_tokens or 512
        self._trigger_messages = self.config.summary_trigger_messages or 20
        self._trigger_tokens = self.config.summary_trigger_tokens or 4000
        self._recent_window = self.config.summary_recent_window or 5
    
    def add_message(self, message: AgentMessage) -> None:
        """Add a message to memory."""
        self.messages.append(message)
    
    def get_messages(self) -> List[AgentMessage]:
        """
        Get messages for context (applying windowing/summarization).
        
        For summary policy:
        - Returns synthetic summary message (if exists) + recent unsummarized messages
        - Falls back to windowed memory if summarization is unavailable
        """
        policy = self.config.policy
        
        if policy == "none":
            return []
        elif policy == "full_history":
            return self.messages
        elif policy == "conversation_window":
            window_size = self.config.window_size or 10
            return self.messages[-window_size:]
        elif policy == "summary":
            return self._get_summarized_messages()
        else:
            return self.messages
    
    def _get_summarized_messages(self) -> List[AgentMessage]:
        """
        Get messages with summarization applied.
        
        Returns:
            List containing summary message (if exists) + recent unsummarized messages
        """
        messages = []
        
        # Add synthetic summary message if available
        if self.summary:
            summary_msg = AgentMessage(
                role="system",
                content=f"[Previous conversation summary]: {self.summary}",
                metadata={"type": "summary", "summarized_up_to_index": self.last_summarized_index}
            )
            messages.append(summary_msg)
        
        # Add recent unsummarized messages
        recent_start = max(self.last_summarized_index, len(self.messages) - self._recent_window)
        recent_messages = self.messages[recent_start:]
        messages.extend(recent_messages)
        
        # Fallback: if no summary and too many messages, use windowing
        if not self.summary and len(messages) > self._recent_window * 2:
            logger.warning(
                f"No summary available but {len(messages)} messages in memory. "
                "Falling back to windowed memory."
            )
            window_size = self.config.window_size or 10
            return self.messages[-window_size:]
        
        return messages
    
    async def maybe_summarize(self) -> None:
        """
        Check if summarization should be triggered and perform if needed.
        
        Summarization is triggered when:
        - Number of unsummarized messages exceeds summary_trigger_messages
        - OR estimated token count of unsummarized messages exceeds summary_trigger_tokens
        
        Implements:
        - Incremental summarization (builds on existing summary)
        - Exponential backoff on repeated failures
        - Robust error handling with fallback behavior
        """
        # Only applicable for summary policy
        if self.config.policy != "summary":
            return
        
        # Check if we have summarizer configured
        if not self.config.summarizer:
            logger.warning(
                "Summary policy enabled but no summarizer configured. "
                "Set memory.summarizer (e.g., 'openai/gpt-4o-mini')"
            )
            return
        
        # Calculate unsummarized messages
        unsummarized_count = len(self.messages) - self.last_summarized_index
        
        # Check if we need to summarize
        should_summarize = False
        reason = ""
        
        if unsummarized_count >= self._trigger_messages:
            should_summarize = True
            reason = f"message count ({unsummarized_count} >= {self._trigger_messages})"
        else:
            # Estimate tokens in unsummarized messages
            unsummarized_msgs = self.messages[self.last_summarized_index:]
            estimated_tokens = estimate_messages_tokens(unsummarized_msgs)
            
            if estimated_tokens >= self._trigger_tokens:
                should_summarize = True
                reason = f"token count (~{estimated_tokens} >= {self._trigger_tokens})"
        
        if not should_summarize:
            return
        
        # Check backoff after failures
        if self._summarization_failures > 0:
            backoff_delay = min(60.0 * (2 ** (self._summarization_failures - 1)), 3600.0)
            time_since_last = time.time() - self._last_summarization_attempt
            
            if time_since_last < backoff_delay:
                logger.debug(
                    f"Skipping summarization due to backoff "
                    f"({self._summarization_failures} failures, "
                    f"{backoff_delay - time_since_last:.1f}s remaining)"
                )
                return
        
        logger.info(
            f"Triggering summarization: {reason}, "
            f"{unsummarized_count} messages to summarize"
        )
        
        self._last_summarization_attempt = time.time()
        
        try:
            await self._perform_summarization()
            
            # Reset failure counter on success
            self._summarization_failures = 0
            
            record_metric(
                "agent.memory.summarization.success",
                1,
                tags={"policy": "summary"}
            )
        
        except Exception as e:
            self._summarization_failures += 1
            
            logger.error(
                f"Summarization failed (attempt {self._summarization_failures}): {e}",
                exc_info=True
            )
            
            record_metric(
                "agent.memory.summarization.error",
                1,
                tags={
                    "policy": "summary",
                    "error_type": type(e).__name__,
                    "failures": str(self._summarization_failures)
                }
            )
            
            # Log fallback behavior
            logger.warning(
                f"Falling back to windowed memory after summarization failure. "
                f"Will retry after backoff period."
            )
    
    async def _perform_summarization(self) -> None:
        """
        Perform the actual summarization of unsummarized messages.
        
        Raises:
            Exception: If summarization fails
        """
        from namel3ss.agents.summarization import get_summarizer
        
        # Lazy load summarizer
        if self._summarizer is None:
            self._summarizer = get_summarizer(
                self.config.summarizer,
                config=self.config.config
            )
        
        # Get messages to summarize (all except recent window)
        end_index = len(self.messages) - self._recent_window
        if end_index <= self.last_summarized_index:
            logger.debug("No new messages to summarize after reserving recent window")
            return
        
        messages_to_summarize = self.messages[self.last_summarized_index:end_index]
        
        if not messages_to_summarize:
            logger.debug("No messages to summarize")
            return
        
        # Perform summarization
        logger.info(
            f"Summarizing {len(messages_to_summarize)} messages "
            f"(indices {self.last_summarized_index}-{end_index})"
        )
        
        new_summary = await self._summarizer.summarize(
            messages_to_summarize,
            existing_summary=self.summary,
            max_tokens=self._max_summary_tokens
        )
        
        # Update state
        self.summary = new_summary
        self.last_summarized_index = end_index
        
        logger.info(
            f"Summarization complete: {len(new_summary)} chars, "
            f"now tracking from index {self.last_summarized_index}"
        )
    
    def clear(self) -> None:
        """Clear all messages and summary from memory."""
        self.messages.clear()
        self.summary = None
        self.last_summarized_index = 0
        self._summarization_failures = 0
