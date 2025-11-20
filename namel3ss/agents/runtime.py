"""Agent runtime execution engine with LLM, tools, memory, and goal-based behavior."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import json
import logging
import time

from namel3ss.ast.agents import AgentDefinition, MemoryConfig
from namel3ss.llm.base import BaseLLM, ChatMessage, LLMResponse
from namel3ss.observability.metrics import record_metric
from namel3ss.templates import get_default_engine, TemplateError

logger = logging.getLogger(__name__)


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


@dataclass
class AgentMessage:
    """A message in agent conversation history."""
    
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_call: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTurn:
    """A complete turn of agent execution (prompt + response + tool calls)."""
    
    messages: List[AgentMessage]
    llm_response: Optional[LLMResponse] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result from agent execution."""
    
    status: str  # "success", "error", "max_turns", "goal_achieved"
    final_response: str
    turns: List[AgentTurn]
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


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


class AgentRuntime:
    """
    Runtime execution engine for agents.
    
    Manages agent execution with:
    - LLM interaction for reasoning and response generation
    - Tool invocation for actions
    - Memory management for conversation context
    - Goal-based execution with max turns
    """
    
    def __init__(
        self,
        agent_def: AgentDefinition,
        llm_instance: BaseLLM,
        tool_registry: Optional[Dict[str, Callable]] = None,
    ):
        """
        Initialize agent runtime.
        
        Args:
            agent_def: AgentDefinition from AST
            llm_instance: Initialized LLM instance (BaseLLM)
            tool_registry: Dict mapping tool names to callable functions
        """
        self.agent_def = agent_def
        self.llm = llm_instance
        self.tool_registry = tool_registry or {}
        self.memory = BaseMemory(agent_def.memory_config)
    
    async def aact(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        max_turns: Optional[int] = None,
    ) -> AgentResult:
        """
        Execute agent asynchronously for user input with multi-turn reasoning.
        
        Supports async summarization for memory management.
        
        Args:
            user_input: User's input message
            context: Additional context variables
            max_turns: Maximum reasoning turns (overrides agent default)
        
        Returns:
            AgentResult with final response and execution trace
        """
        start_time = time.time()
        context = context or {}
        max_turns = max_turns or self.agent_def.max_turns or 10
        
        # Record execution start
        record_metric("agent.execution.start", 1, tags={"agent": self.agent_def.name})
        
        turns: List[AgentTurn] = []
        
        try:
            # Add user message to memory
            user_msg = AgentMessage(role="user", content=user_input)
            self.memory.add_message(user_msg)
            
            # Execute turns until goal achieved or max turns reached
            for turn_num in range(max_turns):
                logger.debug(f"Agent {self.agent_def.name} turn {turn_num + 1}/{max_turns}")
                
                # Check if summarization is needed before this turn
                await self.memory.maybe_summarize()
                
                # Record turn start
                record_metric("agent.turn.start", 1, tags={
                    "agent": self.agent_def.name,
                    "turn": str(turn_num + 1)
                })
                
                turn = self._execute_turn(context)
                turns.append(turn)
                
                # Record turn completion
                record_metric("agent.turn.complete", 1, tags={
                    "agent": self.agent_def.name,
                    "turn": str(turn_num + 1),
                    "tool_calls": str(len(turn.tool_calls))
                })
                
                # Check if agent is done (no tool calls or explicit finish)
                if not turn.tool_calls:
                    final_response = turn.llm_response.text if turn.llm_response else ""
                    elapsed_ms = (time.time() - start_time) * 1000
                    
                    # Final summarization after completion
                    await self.memory.maybe_summarize()
                    
                    # Record successful completion
                    record_metric("agent.execution.complete", elapsed_ms, tags={
                        "agent": self.agent_def.name,
                        "turns": str(len(turns)),
                        "status": "success"
                    })
                    
                    return AgentResult(
                        status="success",
                        final_response=final_response,
                        turns=turns,
                        metadata={"total_turns": len(turns), "elapsed_ms": elapsed_ms},
                    )
            
            # Max turns reached
            elapsed_ms = (time.time() - start_time) * 1000
            final_response = turns[-1].llm_response.text if turns and turns[-1].llm_response else ""
            
            # Final summarization
            await self.memory.maybe_summarize()
            
            record_metric("agent.execution.max_turns", 1, tags={
                "agent": self.agent_def.name,
                "turns": str(max_turns)
            })
            
            return AgentResult(
                status="max_turns",
                final_response=final_response,
                turns=turns,
                metadata={"total_turns": len(turns), "max_turns": max_turns},
            )
        
        except Exception as e:
            logger.error(f"Agent {self.agent_def.name} error: {e}", exc_info=True)
            record_metric("agent.execution.error", 1, tags={
                "agent": self.agent_def.name,
                "error_type": type(e).__name__
            })
            return AgentResult(
                status="error",
                final_response="",
                turns=turns,
                metadata={"total_turns": len(turns)},
                error=str(e),
            )
    
    def act(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        max_turns: Optional[int] = None,
    ) -> AgentResult:
        """
        Execute agent for user input with multi-turn reasoning (sync wrapper).
        
        For production use with summarization, prefer aact() for proper async execution.
        This method will use asyncio.run() to execute the async version.
        
        Args:
            user_input: User's input message
            context: Additional context variables
            max_turns: Maximum reasoning turns (overrides agent default)
        
        Returns:
            AgentResult with final response and execution trace
        """
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, should use aact instead
            logger.warning(
                "act() called from async context. Use aact() for proper async execution."
            )
            # Create a task but don't await it (not ideal but maintains compatibility)
            return asyncio.run(self.aact(user_input, context, max_turns))
        except RuntimeError:
            # No running loop, we can create one
            return asyncio.run(self.aact(user_input, context, max_turns))
    
    def act_sync_legacy(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        max_turns: Optional[int] = None,
    ) -> AgentResult:
        """
        Legacy synchronous execution without summarization support.
        
        Note: This method does not support memory summarization.
        Use aact() for full feature support.
        """
        start_time = time.time()
        context = context or {}
        max_turns = max_turns or self.agent_def.max_turns or 10
        
        # Record execution start
        record_metric("agent.execution.start", 1, tags={"agent": self.agent_def.name})
        
        turns: List[AgentTurn] = []
        
        try:
            # Add user message to memory
            user_msg = AgentMessage(role="user", content=user_input)
            self.memory.add_message(user_msg)
            
            # Execute turns until goal achieved or max turns reached
            for turn_num in range(max_turns):
                logger.debug(f"Agent {self.agent_def.name} turn {turn_num + 1}/{max_turns}")
                
                # Record turn start
                record_metric("agent.turn.start", 1, tags={
                    "agent": self.agent_def.name,
                    "turn": str(turn_num + 1)
                })
                
                turn = self._execute_turn(context)
                turns.append(turn)
                
                # Record turn completion
                record_metric("agent.turn.complete", 1, tags={
                    "agent": self.agent_def.name,
                    "turn": str(turn_num + 1),
                    "tool_calls": str(len(turn.tool_calls))
                })
                
                # Check if agent is done (no tool calls or explicit finish)
                if not turn.tool_calls:
                    final_response = turn.llm_response.text if turn.llm_response else ""
                    elapsed_ms = (time.time() - start_time) * 1000
                    
                    # Record successful completion
                    record_metric("agent.execution.complete", elapsed_ms, tags={
                        "agent": self.agent_def.name,
                        "turns": str(len(turns)),
                        "status": "success"
                    })
                    
                    return AgentResult(
                        status="success",
                        final_response=final_response,
                        turns=turns,
                        metadata={"total_turns": len(turns), "elapsed_ms": elapsed_ms},
                    )
            
            # Max turns reached
            elapsed_ms = (time.time() - start_time) * 1000
            final_response = turns[-1].llm_response.text if turns and turns[-1].llm_response else ""
            
            record_metric("agent.execution.max_turns", 1, tags={
                "agent": self.agent_def.name,
                "turns": str(max_turns)
            })
            
            return AgentResult(
                status="max_turns",
                final_response=final_response,
                turns=turns,
                metadata={"total_turns": len(turns), "max_turns": max_turns},
            )
        
        except Exception as e:
            logger.error(f"Agent {self.agent_def.name} error: {e}", exc_info=True)
            record_metric("agent.execution.error", 1, tags={
                "agent": self.agent_def.name,
                "error_type": type(e).__name__
            })
            return AgentResult(
                status="error",
                final_response="",
                turns=turns,
                metadata={"total_turns": len(turns)},
                error=str(e),
            )
    
    def _execute_turn(self, context: Dict[str, Any]) -> AgentTurn:
        """Execute a single reasoning turn."""
        # Build messages for LLM
        messages = self._build_messages(context)
        
        # Generate LLM response
        llm_kwargs = {}
        if self.agent_def.temperature is not None:
            llm_kwargs["temperature"] = self.agent_def.temperature
        
        llm_response = self.llm.generate_chat(messages, **llm_kwargs)
        
        # Parse tool calls from response
        tool_calls = self._parse_tool_calls(llm_response.text)
        
        # Execute tools
        tool_results = []
        if tool_calls:
            tool_results = self._execute_tools(tool_calls)
            
            # Add tool results to memory
            for result in tool_results:
                tool_msg = AgentMessage(
                    role="tool",
                    content=json.dumps(result),
                    tool_result=result,
                )
                self.memory.add_message(tool_msg)
        else:
            # Add assistant response to memory
            assistant_msg = AgentMessage(
                role="assistant",
                content=llm_response.text,
            )
            self.memory.add_message(assistant_msg)
        
        return AgentTurn(
            messages=[m for m in messages],
            llm_response=llm_response,
            tool_calls=tool_calls,
            tool_results=tool_results,
            metadata={
                "tokens_used": llm_response.total_tokens,
                "finish_reason": llm_response.finish_reason,
            },
        )
    
    def _build_messages(self, context: Dict[str, Any]) -> List[ChatMessage]:
        """Build chat messages from system prompt, goal, and memory."""
        messages = []
        
        # System message with agent configuration
        system_parts = []
        
        if self.agent_def.system_prompt:
            system_parts.append(self.agent_def.system_prompt)
        
        if self.agent_def.goal:
            system_parts.append(f"\nYour goal: {self.agent_def.goal}")
        
        if self.agent_def.tool_names:
            tools_desc = self._format_tools_description()
            system_parts.append(f"\nAvailable tools: {tools_desc}")
        
        if system_parts:
            messages.append(ChatMessage(
                role="system",
                content="\n".join(system_parts),
            ))
        
        # Add conversation history from memory
        for mem_msg in self.memory.get_messages():
            messages.append(ChatMessage(
                role=mem_msg.role,
                content=mem_msg.content,
            ))
        
        return messages
    
    def _format_tools_description(self) -> str:
        """
        Format available tools for system prompt using template engine.
        
        Returns:
            Formatted tool description string
        """
        engine = get_default_engine()
        template_source = """
To use a tool, respond with:
TOOL_CALL: tool_name(arg1="value1", arg2="value2")

Available tools: {{ tool_names }}"""
        
        try:
            compiled = engine.compile(template_source, name="agent_tools_description")
            tool_names = ", ".join(self.agent_def.tool_names)
            return compiled.render({"tool_names": tool_names})
        except TemplateError as e:
            # Fallback to simple string if template fails (shouldn't happen with this simple template)
            logger.warning(f"Template rendering failed for tools description: {e}")
            tool_names = ", ".join(self.agent_def.tool_names)
            return f"\nTo use a tool, respond with:\nTOOL_CALL: tool_name(arg1=\"value1\", arg2=\"value2\")\n\nAvailable tools: {tool_names}"
    
    def _parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Parse tool calls from LLM response text."""
        tool_calls = []
        
        # Look for TOOL_CALL: pattern
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("TOOL_CALL:"):
                try:
                    call_str = line[10:].strip()  # Remove "TOOL_CALL:"
                    
                    # Parse tool_name(args)
                    if "(" in call_str and call_str.endswith(")"):
                        tool_name = call_str[:call_str.index("(")].strip()
                        args_str = call_str[call_str.index("(") + 1:-1]
                        
                        # Parse arguments (simplified - would need proper parser for complex cases)
                        args = self._parse_tool_args(args_str)
                        
                        tool_calls.append({
                            "tool": tool_name,
                            "args": args,
                        })
                except Exception as e:
                    logger.warning(f"Failed to parse tool call: {line}, error: {e}")
        
        return tool_calls
    
    def _parse_tool_args(self, args_str: str) -> Dict[str, Any]:
        """Parse tool arguments from string."""
        args = {}
        
        if not args_str.strip():
            return args
        
        # Simple parser for key="value" pairs
        import re
        pattern = r'(\w+)=("[^"]*"|\'[^\']*\'|\S+)'
        matches = re.findall(pattern, args_str)
        
        for key, value in matches:
            # Remove quotes if present
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            args[key] = value
        
        return args
    
    def _execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute tool calls and return results."""
        results = []
        
        for call in tool_calls:
            tool_name = call["tool"]
            args = call["args"]
            
            if tool_name not in self.tool_registry:
                logger.warning(f"Tool {tool_name} not found in registry")
                record_metric("agent.tool.error", 1, tags={
                    "agent": self.agent_def.name,
                    "tool": tool_name,
                    "error": "not_found"
                })
                results.append({
                    "tool": tool_name,
                    "status": "error",
                    "error": f"Tool {tool_name} not available",
                })
                continue
            
            try:
                tool_fn = self.tool_registry[tool_name]
                result = tool_fn(**args)
                
                record_metric("agent.tool.success", 1, tags={
                    "agent": self.agent_def.name,
                    "tool": tool_name
                })
                
                results.append({
                    "tool": tool_name,
                    "status": "success",
                    "result": result,
                })
            except Exception as e:
                logger.error(f"Tool {tool_name} execution failed: {e}", exc_info=True)
                record_metric("agent.tool.error", 1, tags={
                    "agent": self.agent_def.name,
                    "tool": tool_name,
                    "error": "execution_failed"
                })
                results.append({
                    "tool": tool_name,
                    "status": "error",
                    "error": str(e),
                })
        
        return results
    
    def reset(self) -> None:
        """Reset agent state (clear memory)."""
        self.memory.clear()
