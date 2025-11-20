from __future__ import annotations

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

from .data_models import AgentMessage, AgentTurn, AgentResult
from .memory import BaseMemory
from .token_utils import estimate_tokens

logger = logging.getLogger(__name__)


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
