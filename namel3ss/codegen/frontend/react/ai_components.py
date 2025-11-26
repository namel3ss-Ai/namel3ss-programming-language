"""
React AI semantic component generators.

This module generates TypeScript React components for AI-specific UI:
- ChatThread: Display AI conversation threads with streaming support
- AgentPanel: Show agent state, metrics, and performance
- ToolCallView: Display tool invocations with inputs/outputs
- LogView: Structured log/trace viewing with filtering
- EvaluationResult: Eval metrics, histograms, error analysis
- DiffView: Side-by-side or unified text/code diffs
"""

import textwrap
from pathlib import Path

from .utils import write_file


def write_chat_thread_component(components_dir: Path) -> None:
    """Generate ChatThread.tsx for AI conversation display."""
    content = textwrap.dedent(
        """
        import { useEffect, useRef, useState } from "react";

        export interface Message {
          role: string;
          content: string;
          timestamp?: string;
          tokens?: number;
          [key: string]: any;
        }

        export interface ChatThreadProps {
          thread_id: string;
          messages_binding: string;
          data?: { messages?: Message[] };
          group_by?: string;
          show_timestamps?: boolean;
          show_avatar?: boolean;
          reverse_order?: boolean;
          auto_scroll?: boolean;
          max_height?: string;
          streaming_enabled?: boolean;
          streaming_source?: string;
          show_role_labels?: boolean;
          show_token_count?: boolean;
          enable_copy?: boolean;
          enable_regenerate?: boolean;
          variant?: string;
        }

        export default function ChatThread({
          thread_id,
          messages_binding,
          data,
          group_by = "role",
          show_timestamps = true,
          show_avatar = true,
          reverse_order = false,
          auto_scroll = true,
          max_height,
          streaming_enabled = false,
          streaming_source,
          show_role_labels = true,
          show_token_count = false,
          enable_copy = true,
          enable_regenerate = false,
          variant = "default",
        }: ChatThreadProps) {
          const messagesEndRef = useRef<HTMLDivElement>(null);
          const [messages, setMessages] = useState<Message[]>(data?.messages || []);
          const [streamingContent, setStreamingContent] = useState<string>("");

          useEffect(() => {
            if (data?.messages) {
              setMessages(data.messages);
            }
          }, [data]);

          useEffect(() => {
            if (auto_scroll && messagesEndRef.current) {
              messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
            }
          }, [messages, streamingContent, auto_scroll]);

          const handleCopy = (content: string) => {
            navigator.clipboard.writeText(content);
          };

          const displayMessages = reverse_order ? [...messages].reverse() : messages;

          return (
            <div
              className={`chat-thread chat-thread--${variant}`}
              data-thread-id={thread_id}
              style={{ maxHeight: max_height, overflowY: "auto" }}
            >
              {displayMessages.map((message, idx) => (
                <div
                  key={idx}
                  className={`chat-message chat-message--${message.role}`}
                >
                  {show_avatar && (
                    <div className="chat-avatar">
                      {message.role.charAt(0).toUpperCase()}
                    </div>
                  )}
                  <div className="chat-message-content">
                    {show_role_labels && (
                      <div className="chat-message-header">
                        <span className="chat-role">{message.role}</span>
                        {show_timestamps && message.timestamp && (
                          <span className="chat-timestamp">{message.timestamp}</span>
                        )}
                        {show_token_count && message.tokens && (
                          <span className="chat-tokens">{message.tokens} tokens</span>
                        )}
                      </div>
                    )}
                    <div className="chat-message-text">{message.content}</div>
                    {enable_copy && (
                      <button
                        className="chat-copy-btn"
                        onClick={() => handleCopy(message.content)}
                        title="Copy message"
                      >
                        📋
                      </button>
                    )}
                  </div>
                </div>
              ))}
              {streaming_enabled && streamingContent && (
                <div className="chat-message chat-message--assistant chat-message--streaming">
                  {show_avatar && <div className="chat-avatar">A</div>}
                  <div className="chat-message-content">
                    <div className="chat-message-text">{streamingContent}</div>
                    <span className="chat-streaming-indicator">●</span>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          );
        }
        """
    ).strip() + "\n"
    write_file(components_dir / "ChatThread.tsx", content)


def write_agent_panel_component(components_dir: Path) -> None:
    """Generate AgentPanel.tsx for agent state and metrics display."""
    content = textwrap.dedent(
        """
        export interface AgentData {
          name?: string;
          status?: string;
          model?: string;
          tokens?: number;
          cost?: number;
          latency?: number;
          last_error?: string;
          tools?: string[];
          [key: string]: any;
        }

        export interface AgentPanelProps {
          panel_id: string;
          agent_binding: string;
          data?: { agent?: AgentData; metrics?: any };
          metrics_binding?: string;
          show_status?: boolean;
          show_metrics?: boolean;
          show_profile?: boolean;
          show_limits?: boolean;
          show_last_error?: boolean;
          show_tools?: boolean;
          show_tokens?: boolean;
          show_cost?: boolean;
          show_latency?: boolean;
          show_model?: boolean;
          variant?: string;
          compact?: boolean;
        }

        export default function AgentPanel({
          panel_id,
          agent_binding,
          data,
          metrics_binding,
          show_status = true,
          show_metrics = true,
          show_profile = false,
          show_limits = false,
          show_last_error = false,
          show_tools = false,
          show_tokens = true,
          show_cost = true,
          show_latency = true,
          show_model = true,
          variant = "card",
          compact = false,
        }: AgentPanelProps) {
          const agent = data?.agent || {};
          const metrics = data?.metrics || {};

          return (
            <div
              className={`agent-panel agent-panel--${variant} ${compact ? 'agent-panel--compact' : ''}`}
              data-panel-id={panel_id}
            >
              <div className="agent-panel-header">
                <h3>{agent.name || "Agent"}</h3>
                {show_status && agent.status && (
                  <span className={`agent-status agent-status--${agent.status}`}>
                    {agent.status}
                  </span>
                )}
              </div>

              <div className="agent-panel-body">
                {show_model && agent.model && (
                  <div className="agent-field">
                    <span className="agent-field-label">Model:</span>
                    <span className="agent-field-value">{agent.model}</span>
                  </div>
                )}

                {show_metrics && (
                  <div className="agent-metrics">
                    {show_tokens && agent.tokens !== undefined && (
                      <div className="agent-metric">
                        <span className="agent-metric-label">Tokens:</span>
                        <span className="agent-metric-value">{agent.tokens.toLocaleString()}</span>
                      </div>
                    )}
                    {show_cost && agent.cost !== undefined && (
                      <div className="agent-metric">
                        <span className="agent-metric-label">Cost:</span>
                        <span className="agent-metric-value">${agent.cost.toFixed(4)}</span>
                      </div>
                    )}
                    {show_latency && agent.latency !== undefined && (
                      <div className="agent-metric">
                        <span className="agent-metric-label">Latency:</span>
                        <span className="agent-metric-value">{agent.latency}ms</span>
                      </div>
                    )}
                  </div>
                )}

                {show_tools && agent.tools && agent.tools.length > 0 && (
                  <div className="agent-tools">
                    <div className="agent-field-label">Tools:</div>
                    <div className="agent-tools-list">
                      {agent.tools.map((tool: string, idx: number) => (
                        <span key={idx} className="agent-tool-badge">{tool}</span>
                      ))}
                    </div>
                  </div>
                )}

                {show_last_error && agent.last_error && (
                  <div className="agent-error">
                    <strong>Last Error:</strong>
                    <pre>{agent.last_error}</pre>
                  </div>
                )}
              </div>
            </div>
          );
        }
        """
    ).strip() + "\n"
    write_file(components_dir / "AgentPanel.tsx", content)


def write_tool_call_view_component(components_dir: Path) -> None:
    """Generate ToolCallView.tsx for tool invocation display."""
    content = textwrap.dedent(
        """
        import { useState } from "react";

        export interface ToolCall {
          id?: string;
          name: string;
          inputs?: any;
          outputs?: any;
          status?: string;
          duration?: number;
          timestamp?: string;
          error?: string;
          [key: string]: any;
        }

        export interface ToolCallViewProps {
          view_id: string;
          calls_binding: string;
          data?: { calls?: ToolCall[] };
          show_inputs?: boolean;
          show_outputs?: boolean;
          show_timing?: boolean;
          show_status?: boolean;
          show_raw_payload?: boolean;
          filter_tool_name?: string;
          filter_status?: string;
          variant?: string;
          expandable?: boolean;
          max_height?: string;
          enable_retry?: boolean;
          enable_copy?: boolean;
        }

        export default function ToolCallView({
          view_id,
          calls_binding,
          data,
          show_inputs = true,
          show_outputs = true,
          show_timing = true,
          show_status = true,
          show_raw_payload = false,
          filter_tool_name,
          filter_status,
          variant = "list",
          expandable = true,
          max_height,
          enable_retry = false,
          enable_copy = true,
        }: ToolCallViewProps) {
          const [expandedCalls, setExpandedCalls] = useState<Set<string>>(new Set());
          
          const calls = data?.calls || [];
          const filteredCalls = calls.filter(call => {
            if (filter_tool_name && call.name !== filter_tool_name) return false;
            if (filter_status && call.status !== filter_status) return false;
            return true;
          });

          const toggleExpanded = (callId: string) => {
            setExpandedCalls(prev => {
              const next = new Set(prev);
              if (next.has(callId)) {
                next.delete(callId);
              } else {
                next.add(callId);
              }
              return next;
            });
          };

          const handleCopy = (content: any) => {
            navigator.clipboard.writeText(JSON.stringify(content, null, 2));
          };

          return (
            <div
              className={`tool-call-view tool-call-view--${variant}`}
              data-view-id={view_id}
              style={{ maxHeight: max_height, overflowY: "auto" }}
            >
              {filteredCalls.length === 0 ? (
                <div className="tool-call-empty">No tool calls to display</div>
              ) : (
                filteredCalls.map((call, idx) => {
                  const callId = call.id || `call-${idx}`;
                  const isExpanded = expandable && expandedCalls.has(callId);

                  return (
                    <div key={callId} className="tool-call-item">
                      <div className="tool-call-header">
                        {expandable && (
                          <button
                            className="tool-call-toggle"
                            onClick={() => toggleExpanded(callId)}
                          >
                            {isExpanded ? "▼" : "▶"}
                          </button>
                        )}
                        <span className="tool-call-name">{call.name}</span>
                        {show_status && call.status && (
                          <span className={`tool-call-status tool-call-status--${call.status}`}>
                            {call.status}
                          </span>
                        )}
                        {show_timing && call.duration !== undefined && (
                          <span className="tool-call-duration">{call.duration}ms</span>
                        )}
                        {enable_copy && (
                          <button
                            className="tool-call-copy"
                            onClick={() => handleCopy(call)}
                            title="Copy tool call"
                          >
                            📋
                          </button>
                        )}
                      </div>

                      {(!expandable || isExpanded) && (
                        <div className="tool-call-body">
                          {show_inputs && call.inputs && (
                            <div className="tool-call-section">
                              <strong>Inputs:</strong>
                              <pre>{JSON.stringify(call.inputs, null, 2)}</pre>
                            </div>
                          )}
                          {show_outputs && call.outputs && (
                            <div className="tool-call-section">
                              <strong>Outputs:</strong>
                              <pre>{JSON.stringify(call.outputs, null, 2)}</pre>
                            </div>
                          )}
                          {call.error && (
                            <div className="tool-call-error">
                              <strong>Error:</strong>
                              <pre>{call.error}</pre>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          );
        }
        """
    ).strip() + "\n"
    write_file(components_dir / "ToolCallView.tsx", content)


def write_log_view_component(components_dir: Path) -> None:
    """Generate LogView.tsx for structured log/trace viewing."""
    content = textwrap.dedent(
        """
        import { useState, useMemo, useEffect, useRef } from "react";

        export interface LogEntry {
          timestamp?: string;
          level: string;
          message: string;
          metadata?: any;
          source?: string;
          [key: string]: any;
        }

        export interface LogViewProps {
          view_id: string;
          logs_binding: string;
          data?: { logs?: LogEntry[] };
          level_filter?: string[];
          search_enabled?: boolean;
          search_placeholder?: string;
          show_timestamp?: boolean;
          show_level?: boolean;
          show_metadata?: boolean;
          show_source?: boolean;
          auto_scroll?: boolean;
          auto_refresh?: boolean;
          refresh_interval?: number;
          max_entries?: number;
          variant?: string;
          max_height?: string;
          virtualized?: boolean;
          enable_copy?: boolean;
          enable_download?: boolean;
        }

        export default function LogView({
          view_id,
          logs_binding,
          data,
          level_filter,
          search_enabled = true,
          search_placeholder = "Search logs...",
          show_timestamp = true,
          show_level = true,
          show_metadata = false,
          show_source = false,
          auto_scroll = true,
          auto_refresh = false,
          refresh_interval = 5000,
          max_entries = 1000,
          variant = "default",
          max_height,
          virtualized = true,
          enable_copy = true,
          enable_download = false,
        }: LogViewProps) {
          const [searchQuery, setSearchQuery] = useState("");
          const [selectedLevels, setSelectedLevels] = useState<Set<string>>(
            new Set(level_filter || ["info", "warn", "error"])
          );
          const logsEndRef = useRef<HTMLDivElement>(null);

          const logs = useMemo(() => {
            let entries = data?.logs || [];
            
            // Apply level filter
            entries = entries.filter(log => selectedLevels.has(log.level));
            
            // Apply search
            if (searchQuery) {
              const query = searchQuery.toLowerCase();
              entries = entries.filter(log =>
                log.message.toLowerCase().includes(query) ||
                (log.source && log.source.toLowerCase().includes(query))
              );
            }
            
            // Limit entries
            if (entries.length > max_entries) {
              entries = entries.slice(-max_entries);
            }
            
            return entries;
          }, [data, selectedLevels, searchQuery, max_entries]);

          useEffect(() => {
            if (auto_scroll && logsEndRef.current) {
              logsEndRef.current.scrollIntoView({ behavior: "smooth" });
            }
          }, [logs, auto_scroll]);

          const toggleLevel = (level: string) => {
            setSelectedLevels(prev => {
              const next = new Set(prev);
              if (next.has(level)) {
                next.delete(level);
              } else {
                next.add(level);
              }
              return next;
            });
          };

          const handleCopy = () => {
            const logText = logs.map(log => 
              `[${log.timestamp || ''}] ${log.level.toUpperCase()}: ${log.message}`
            ).join('\\n');
            navigator.clipboard.writeText(logText);
          };

          const handleDownload = () => {
            const logText = logs.map(log => 
              JSON.stringify(log)
            ).join('\\n');
            const blob = new Blob([logText], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `logs-${Date.now()}.jsonl`;
            a.click();
            URL.revokeObjectURL(url);
          };

          return (
            <div className={`log-view log-view--${variant}`} data-view-id={view_id}>
              <div className="log-view-toolbar">
                {search_enabled && (
                  <input
                    type="search"
                    className="log-search"
                    placeholder={search_placeholder}
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                )}
                <div className="log-level-filters">
                  {["debug", "info", "warn", "error"].map(level => (
                    <label key={level} className="log-level-filter">
                      <input
                        type="checkbox"
                        checked={selectedLevels.has(level)}
                        onChange={() => toggleLevel(level)}
                      />
                      <span className={`log-level-label log-level-label--${level}`}>
                        {level}
                      </span>
                    </label>
                  ))}
                </div>
                <div className="log-actions">
                  {enable_copy && (
                    <button className="log-action-btn" onClick={handleCopy} title="Copy logs">
                      📋
                    </button>
                  )}
                  {enable_download && (
                    <button className="log-action-btn" onClick={handleDownload} title="Download logs">
                      💾
                    </button>
                  )}
                </div>
              </div>

              <div
                className="log-entries"
                style={{ maxHeight: max_height, overflowY: "auto" }}
              >
                {logs.length === 0 ? (
                  <div className="log-empty">No logs to display</div>
                ) : (
                  logs.map((log, idx) => (
                    <div key={idx} className={`log-entry log-entry--${log.level}`}>
                      {show_timestamp && log.timestamp && (
                        <span className="log-timestamp">{log.timestamp}</span>
                      )}
                      {show_level && (
                        <span className={`log-level log-level--${log.level}`}>
                          {log.level.toUpperCase()}
                        </span>
                      )}
                      {show_source && log.source && (
                        <span className="log-source">{log.source}</span>
                      )}
                      <span className="log-message">{log.message}</span>
                      {show_metadata && log.metadata && (
                        <pre className="log-metadata">
                          {JSON.stringify(log.metadata, null, 2)}
                        </pre>
                      )}
                    </div>
                  ))
                )}
                <div ref={logsEndRef} />
              </div>
            </div>
          );
        }
        """
    ).strip() + "\n"
    write_file(components_dir / "LogView.tsx", content)


def write_evaluation_result_component(components_dir: Path) -> None:
    """Generate EvaluationResult.tsx for eval metrics and error analysis."""
    content = textwrap.dedent(
        """
        export interface EvalRun {
          id: string;
          name?: string;
          status?: string;
          metrics?: Record<string, number>;
          errors?: Array<{ id: string; input: any; expected: any; actual: any; score: number; }>;
          metadata?: any;
          [key: string]: any;
        }

        export interface EvaluationResultProps {
          result_id: string;
          eval_run_binding: string;
          data?: { eval_run?: EvalRun };
          show_summary?: boolean;
          show_histograms?: boolean;
          show_error_table?: boolean;
          show_metadata?: boolean;
          metrics_to_show?: string[];
          primary_metric?: string;
          filter_metric?: string;
          filter_min_score?: number;
          filter_max_score?: number;
          filter_status?: string;
          show_error_distribution?: boolean;
          show_error_examples?: boolean;
          max_error_examples?: number;
          variant?: string;
          comparison_run_binding?: string;
        }

        export default function EvaluationResult({
          result_id,
          eval_run_binding,
          data,
          show_summary = true,
          show_histograms = true,
          show_error_table = true,
          show_metadata = false,
          metrics_to_show,
          primary_metric,
          filter_metric,
          filter_min_score,
          filter_max_score,
          filter_status,
          show_error_distribution = true,
          show_error_examples = true,
          max_error_examples = 10,
          variant = "dashboard",
          comparison_run_binding,
        }: EvaluationResultProps) {
          const evalRun = data?.eval_run || {};
          const metrics = evalRun.metrics || {};
          const errors = evalRun.errors || [];

          const displayMetrics = metrics_to_show
            ? Object.fromEntries(
                Object.entries(metrics).filter(([key]) => metrics_to_show.includes(key))
              )
            : metrics;

          const filteredErrors = errors.filter(error => {
            if (filter_min_score !== undefined && error.score < filter_min_score) return false;
            if (filter_max_score !== undefined && error.score > filter_max_score) return false;
            return true;
          }).slice(0, max_error_examples);

          return (
            <div
              className={`evaluation-result evaluation-result--${variant}`}
              data-result-id={result_id}
            >
              {show_summary && (
                <div className="eval-summary">
                  <h3>{evalRun.name || "Evaluation Results"}</h3>
                  {evalRun.status && (
                    <span className={`eval-status eval-status--${evalRun.status}`}>
                      {evalRun.status}
                    </span>
                  )}
                </div>
              )}

              {show_summary && Object.keys(displayMetrics).length > 0 && (
                <div className="eval-metrics">
                  {Object.entries(displayMetrics).map(([key, value]) => (
                    <div
                      key={key}
                      className={`eval-metric ${key === primary_metric ? 'eval-metric--primary' : ''}`}
                    >
                      <div className="eval-metric-label">{key}</div>
                      <div className="eval-metric-value">
                        {typeof value === 'number' ? value.toFixed(3) : value}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {show_error_table && filteredErrors.length > 0 && (
                <div className="eval-errors">
                  <h4>Error Analysis ({filteredErrors.length} examples)</h4>
                  <table className="eval-errors-table">
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>Input</th>
                        <th>Expected</th>
                        <th>Actual</th>
                        <th>Score</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredErrors.map(error => (
                        <tr key={error.id}>
                          <td>{error.id}</td>
                          <td><pre>{JSON.stringify(error.input, null, 2)}</pre></td>
                          <td><pre>{JSON.stringify(error.expected, null, 2)}</pre></td>
                          <td><pre>{JSON.stringify(error.actual, null, 2)}</pre></td>
                          <td className={error.score < 0.5 ? 'eval-score-bad' : ''}>
                            {error.score.toFixed(3)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {show_metadata && evalRun.metadata && (
                <div className="eval-metadata">
                  <h4>Metadata</h4>
                  <pre>{JSON.stringify(evalRun.metadata, null, 2)}</pre>
                </div>
              )}
            </div>
          );
        }
        """
    ).strip() + "\n"
    write_file(components_dir / "EvaluationResult.tsx", content)


def write_diff_view_component(components_dir: Path) -> None:
    """Generate DiffView.tsx for side-by-side or unified text/code diffs."""
    content = textwrap.dedent(
        """
        import { useMemo } from "react";

        export interface DiffViewProps {
          view_id: string;
          left_binding: string;
          right_binding: string;
          data?: { left?: string; right?: string };
          mode?: "split" | "unified";
          content_type?: "text" | "code";
          language?: string;
          ignore_whitespace?: boolean;
          ignore_case?: boolean;
          context_lines?: number;
          show_line_numbers?: boolean;
          highlight_inline_changes?: boolean;
          show_legend?: boolean;
          max_height?: string;
          enable_copy?: boolean;
          enable_download?: boolean;
        }

        interface DiffLine {
          type: "added" | "removed" | "unchanged";
          leftNumber?: number;
          rightNumber?: number;
          content: string;
        }

        function computeDiff(
          left: string,
          right: string,
          ignoreWhitespace: boolean,
          ignoreCase: boolean
        ): DiffLine[] {
          // Simple line-by-line diff (in production, use a proper diff library)
          const leftLines = left.split('\\n');
          const rightLines = right.split('\\n');
          const lines: DiffLine[] = [];

          const normalize = (s: string) => {
            let normalized = s;
            if (ignoreWhitespace) normalized = normalized.trim();
            if (ignoreCase) normalized = normalized.toLowerCase();
            return normalized;
          };

          const maxLines = Math.max(leftLines.length, rightLines.length);
          for (let i = 0; i < maxLines; i++) {
            const leftLine = leftLines[i] || "";
            const rightLine = rightLines[i] || "";

            if (normalize(leftLine) === normalize(rightLine)) {
              lines.push({
                type: "unchanged",
                leftNumber: i + 1,
                rightNumber: i + 1,
                content: leftLine,
              });
            } else if (i >= leftLines.length) {
              lines.push({
                type: "added",
                rightNumber: i + 1,
                content: rightLine,
              });
            } else if (i >= rightLines.length) {
              lines.push({
                type: "removed",
                leftNumber: i + 1,
                content: leftLine,
              });
            } else {
              lines.push({
                type: "removed",
                leftNumber: i + 1,
                content: leftLine,
              });
              lines.push({
                type: "added",
                rightNumber: i + 1,
                content: rightLine,
              });
            }
          }

          return lines;
        }

        export default function DiffView({
          view_id,
          left_binding,
          right_binding,
          data,
          mode = "split",
          content_type = "text",
          language,
          ignore_whitespace = false,
          ignore_case = false,
          context_lines = 3,
          show_line_numbers = true,
          highlight_inline_changes = true,
          show_legend = true,
          max_height,
          enable_copy = true,
          enable_download = false,
        }: DiffViewProps) {
          const leftContent = data?.left || "";
          const rightContent = data?.right || "";

          const diffLines = useMemo(
            () => computeDiff(leftContent, rightContent, ignore_whitespace, ignore_case),
            [leftContent, rightContent, ignore_whitespace, ignore_case]
          );

          const handleCopy = (side: 'left' | 'right') => {
            const content = side === 'left' ? leftContent : rightContent;
            navigator.clipboard.writeText(content);
          };

          const handleDownload = () => {
            const content = `--- Left\\n${leftContent}\\n\\n+++ Right\\n${rightContent}`;
            const blob = new Blob([content], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `diff-${Date.now()}.txt`;
            a.click();
            URL.revokeObjectURL(url);
          };

          return (
            <div
              className={`diff-view diff-view--${mode}`}
              data-view-id={view_id}
            >
              {show_legend && (
                <div className="diff-legend">
                  <span className="diff-legend-item diff-legend-item--added">+ Added</span>
                  <span className="diff-legend-item diff-legend-item--removed">- Removed</span>
                  <span className="diff-legend-item diff-legend-item--unchanged">Unchanged</span>
                </div>
              )}

              <div className="diff-toolbar">
                {enable_copy && (
                  <>
                    <button onClick={() => handleCopy('left')} title="Copy left">
                      📋 Left
                    </button>
                    <button onClick={() => handleCopy('right')} title="Copy right">
                      📋 Right
                    </button>
                  </>
                )}
                {enable_download && (
                  <button onClick={handleDownload} title="Download diff">
                    💾 Download
                  </button>
                )}
              </div>

              <div
                className="diff-content"
                style={{ maxHeight: max_height, overflowY: "auto" }}
              >
                {mode === "split" ? (
                  <div className="diff-split">
                    <div className="diff-pane diff-pane--left">
                      {diffLines.filter(line => line.type !== 'added').map((line, idx) => (
                        <div key={idx} className={`diff-line diff-line--${line.type}`}>
                          {show_line_numbers && (
                            <span className="diff-line-number">{line.leftNumber || ''}</span>
                          )}
                          <span className="diff-line-content">{line.content}</span>
                        </div>
                      ))}
                    </div>
                    <div className="diff-pane diff-pane--right">
                      {diffLines.filter(line => line.type !== 'removed').map((line, idx) => (
                        <div key={idx} className={`diff-line diff-line--${line.type}`}>
                          {show_line_numbers && (
                            <span className="diff-line-number">{line.rightNumber || ''}</span>
                          )}
                          <span className="diff-line-content">{line.content}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="diff-unified">
                    {diffLines.map((line, idx) => (
                      <div key={idx} className={`diff-line diff-line--${line.type}`}>
                        {show_line_numbers && (
                          <>
                            <span className="diff-line-number">{line.leftNumber || ''}</span>
                            <span className="diff-line-number">{line.rightNumber || ''}</span>
                          </>
                        )}
                        <span className="diff-line-prefix">
                          {line.type === 'added' ? '+' : line.type === 'removed' ? '-' : ' '}
                        </span>
                        <span className="diff-line-content">{line.content}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          );
        }
        """
    ).strip() + "\n"
    write_file(components_dir / "DiffView.tsx", content)


__all__ = [
    "write_chat_thread_component",
    "write_agent_panel_component",
    "write_tool_call_view_component",
    "write_log_view_component",
    "write_evaluation_result_component",
    "write_diff_view_component",
    "write_all_ai_components",
]


def write_all_ai_components(components_dir: Path) -> None:
    """
    Generate all AI semantic component files.
    
    Writes all 6 AI-specific React components to the components directory:
    - ChatThread.tsx
    - AgentPanel.tsx
    - ToolCallView.tsx
    - LogView.tsx
    - EvaluationResult.tsx
    - DiffView.tsx
    """
    write_chat_thread_component(components_dir)
    write_agent_panel_component(components_dir)
    write_tool_call_view_component(components_dir)
    write_log_view_component(components_dir)
    write_evaluation_result_component(components_dir)
    write_diff_view_component(components_dir)
