import { useState } from 'react';
import { Play, RotateCcw, AlertCircle, CheckCircle2, Clock, DollarSign, Zap } from 'lucide-react';
import { useExecution } from '../hooks/useExecution';
import type { ExecutionSpan } from '../types/graph';

interface ExecutionPanelProps {
  projectId: string;
}

export default function ExecutionPanel({ projectId }: ExecutionPanelProps) {
  const { state, execute, reset, getTotalCost, getTotalTokens, getDuration } = useExecution();
  const [inputData, setInputData] = useState('{}');
  const [entryNode, setEntryNode] = useState('start-1');
  const [expandedSpans, setExpandedSpans] = useState<Set<string>>(new Set());

  const handleExecute = async () => {
    try {
      const input = JSON.parse(inputData);
      await execute(projectId, {
        entry: entryNode,
        input,
      });
    } catch (err) {
      console.error('Execution failed:', err);
    }
  };

  const handleReset = () => {
    reset();
    setInputData('{}');
  };

  const toggleSpan = (spanId: string) => {
    const newExpanded = new Set(expandedSpans);
    if (newExpanded.has(spanId)) {
      newExpanded.delete(spanId);
    } else {
      newExpanded.add(spanId);
    }
    setExpandedSpans(newExpanded);
  };

  const formatDuration = (ms: number): string => {
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const getSpanIcon = (type: string) => {
    switch (type) {
      case 'llm.call': return '🤖';
      case 'tool.call': return '🔧';
      case 'rag.retrieve': return '📚';
      case 'agent.step': return '🎯';
      case 'chain.step': return '⛓️';
      default: return '📝';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ok': return 'text-green-600';
      case 'error': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const renderSpan = (span: ExecutionSpan, level: number = 0) => {
    const isExpanded = expandedSpans.has(span.spanId);
    const children = state.result?.trace.filter(s => s.parentSpanId === span.spanId) || [];
    const hasChildren = children.length > 0;

    return (
      <div key={span.spanId} style={{ marginLeft: `${level * 20}px` }} className="mb-2">
        <div
          className="flex items-center gap-2 rounded border border-border bg-card p-2 hover:bg-accent cursor-pointer"
          onClick={() => toggleSpan(span.spanId)}
        >
          {hasChildren && (
            <span className="text-xs">{isExpanded ? '▼' : '▶'}</span>
          )}
          <span>{getSpanIcon(span.type)}</span>
          <span className="flex-1 text-sm font-medium">{span.name}</span>
          <span className={`text-xs ${getStatusColor(span.status)}`}>
            {span.status}
          </span>
          <span className="text-xs text-muted-foreground">
            {formatDuration(span.durationMs)}
          </span>
          {span.attributes?.cost && (
            <span className="text-xs text-muted-foreground">
              ${span.attributes.cost.toFixed(4)}
            </span>
          )}
        </div>
        
        {isExpanded && (
          <div className="ml-6 mt-2 space-y-2">
            {span.attributes && Object.keys(span.attributes).length > 0 && (
              <div className="rounded border border-border bg-muted p-2 text-xs">
                <div className="font-semibold mb-1">Attributes:</div>
                {Object.entries(span.attributes).map(([key, value]) => (
                  <div key={key} className="flex gap-2">
                    <span className="text-muted-foreground">{key}:</span>
                    <span>{JSON.stringify(value)}</span>
                  </div>
                ))}
              </div>
            )}
            {span.input && (
              <div className="rounded border border-border bg-muted p-2 text-xs">
                <div className="font-semibold mb-1">Input:</div>
                <pre className="whitespace-pre-wrap">{JSON.stringify(span.input, null, 2)}</pre>
              </div>
            )}
            {span.output && (
              <div className="rounded border border-border bg-muted p-2 text-xs">
                <div className="font-semibold mb-1">Output:</div>
                <pre className="whitespace-pre-wrap">{JSON.stringify(span.output, null, 2)}</pre>
              </div>
            )}
          </div>
        )}
        
        {isExpanded && children.map(child => renderSpan(child, level + 1))}
      </div>
    );
  };

  const tokens = getTotalTokens();
  const totalCost = getTotalCost();
  const duration = getDuration();
  const rootSpans = state.result?.trace.filter(s => !s.parentSpanId) || [];

  return (
    <div className="flex h-full flex-col overflow-hidden border-b border-border">
      {/* Header */}
      <div className="border-b border-border bg-muted/50 p-4">
        <h2 className="mb-3 text-lg font-semibold">Execution</h2>
        
        {/* Input Controls */}
        <div className="space-y-3">
          <div>
            <label className="mb-1 block text-xs font-medium text-muted-foreground">
              Entry Node ID
            </label>
            <input
              type="text"
              value={entryNode}
              onChange={(e) => setEntryNode(e.target.value)}
              disabled={state.isExecuting}
              className="w-full rounded border border-input bg-background px-3 py-2 text-sm"
              placeholder="start-1"
            />
          </div>
          
          <div>
            <label className="mb-1 block text-xs font-medium text-muted-foreground">
              Input Data (JSON)
            </label>
            <textarea
              value={inputData}
              onChange={(e) => setInputData(e.target.value)}
              disabled={state.isExecuting}
              className="w-full rounded border border-input bg-background px-3 py-2 text-sm font-mono"
              rows={4}
              placeholder='{"key": "value"}'
            />
          </div>

          <div className="flex gap-2">
            <button
              onClick={handleExecute}
              disabled={state.isExecuting}
              className="flex items-center gap-2 rounded bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
            >
              <Play className="h-4 w-4" />
              {state.isExecuting ? 'Executing...' : 'Execute'}
            </button>
            
            {state.result && (
              <button
                onClick={handleReset}
                disabled={state.isExecuting}
                className="flex items-center gap-2 rounded border border-input bg-background px-4 py-2 text-sm font-medium hover:bg-accent"
              >
                <RotateCcw className="h-4 w-4" />
                Reset
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Results */}
      <div className="flex-1 overflow-auto p-4">
        {state.error && (
          <div className="mb-4 flex items-start gap-2 rounded border border-red-200 bg-red-50 p-3 text-sm text-red-800">
            <AlertCircle className="h-5 w-5 flex-shrink-0" />
            <div>
              <div className="font-semibold">Execution Failed</div>
              <div className="mt-1">{state.error}</div>
            </div>
          </div>
        )}

        {state.isExecuting && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
            Executing graph...
          </div>
        )}

        {state.result && (
          <div className="space-y-4">
            {/* Summary Stats */}
            <div className="grid grid-cols-4 gap-3">
              <div className="rounded border border-border bg-card p-3">
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <CheckCircle2 className="h-4 w-4" />
                  Status
                </div>
                <div className="mt-1 text-lg font-semibold text-green-600">Success</div>
              </div>
              
              <div className="rounded border border-border bg-card p-3">
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Clock className="h-4 w-4" />
                  Duration
                </div>
                <div className="mt-1 text-lg font-semibold">{formatDuration(duration)}</div>
              </div>
              
              <div className="rounded border border-border bg-card p-3">
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Zap className="h-4 w-4" />
                  Tokens
                </div>
                <div className="mt-1 text-lg font-semibold">
                  {(tokens.prompt + tokens.completion).toLocaleString()}
                </div>
                <div className="text-xs text-muted-foreground">
                  {tokens.prompt.toLocaleString()} in / {tokens.completion.toLocaleString()} out
                </div>
              </div>
              
              <div className="rounded border border-border bg-card p-3">
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <DollarSign className="h-4 w-4" />
                  Cost
                </div>
                <div className="mt-1 text-lg font-semibold">${totalCost.toFixed(4)}</div>
              </div>
            </div>

            {/* Final Result */}
            <div>
              <h3 className="mb-2 text-sm font-semibold">Final Result</h3>
              <div className="rounded border border-border bg-muted p-3">
                <pre className="text-xs whitespace-pre-wrap">
                  {JSON.stringify(state.result.result, null, 2)}
                </pre>
              </div>
            </div>

            {/* Execution Trace */}
            <div>
              <h3 className="mb-2 text-sm font-semibold">
                Execution Trace ({state.result.trace.length} spans)
              </h3>
              <div className="space-y-1">
                {rootSpans.map(span => renderSpan(span))}
              </div>
            </div>
          </div>
        )}

        {!state.isExecuting && !state.result && !state.error && (
          <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
            Configure input data and click Execute to run the graph
          </div>
        )}
      </div>
    </div>
  );
}

