import { useState, useCallback } from 'react';
import { graphApi } from '../lib/api';
import type { ExecutionRequest, ExecutionResult, ExecutionSpan } from '../types/graph';

export interface ExecutionState {
  isExecuting: boolean;
  result: ExecutionResult | null;
  error: string | null;
  progress: number;
}

export interface UseExecutionReturn {
  state: ExecutionState;
  execute: (projectId: string, request: ExecutionRequest) => Promise<void>;
  reset: () => void;
  getSpansByType: (type: string) => ExecutionSpan[];
  getTotalCost: () => number;
  getTotalTokens: () => { prompt: number; completion: number };
  getDuration: () => number;
}

/**
 * Hook for managing graph execution state and operations.
 * 
 * Features:
 * - Execute graphs with input data
 * - Track execution progress and status
 * - Parse and aggregate telemetry data
 * - Calculate costs and token usage
 * - Filter spans by type
 * 
 * @example
 * ```tsx
 * const { state, execute, getTotalCost } = useExecution();
 * 
 * await execute(projectId, {
 *   entry: 'start-1',
 *   input: { query: 'test' }
 * });
 * 
 * const cost = getTotalCost();
 * ```
 */
export function useExecution(): UseExecutionReturn {
  const [state, setState] = useState<ExecutionState>({
    isExecuting: false,
    result: null,
    error: null,
    progress: 0,
  });

  const execute = useCallback(async (projectId: string, request: ExecutionRequest) => {
    setState({
      isExecuting: true,
      result: null,
      error: null,
      progress: 0,
    });

    try {
      // Start execution
      const result = await graphApi.executeGraph(projectId, request);

      setState({
        isExecuting: false,
        result,
        error: null,
        progress: 100,
      });
    } catch (err) {
      setState({
        isExecuting: false,
        result: null,
        error: err instanceof Error ? err.message : 'Execution failed',
        progress: 0,
      });
      throw err;
    }
  }, []);

  const reset = useCallback(() => {
    setState({
      isExecuting: false,
      result: null,
      error: null,
      progress: 0,
    });
  }, []);

  const getSpansByType = useCallback((type: string): ExecutionSpan[] => {
    if (!state.result) return [];
    return state.result.trace.filter((span: ExecutionSpan) => span.type === type);
  }, [state.result]);

  const getTotalCost = useCallback((): number => {
    if (!state.result) return 0;
    return state.result.trace.reduce((total: number, span: ExecutionSpan) => {
      return total + (span.attributes?.cost || 0);
    }, 0);
  }, [state.result]);

  const getTotalTokens = useCallback((): { prompt: number; completion: number } => {
    if (!state.result) return { prompt: 0, completion: 0 };
    
    return state.result.trace.reduce((totals: { prompt: number; completion: number }, span: ExecutionSpan) => {
      return {
        prompt: totals.prompt + (span.attributes?.tokensPrompt || 0),
        completion: totals.completion + (span.attributes?.tokensCompletion || 0),
      };
    }, { prompt: 0, completion: 0 });
  }, [state.result]);

  const getDuration = useCallback((): number => {
    if (!state.result || state.result.trace.length === 0) return 0;
    
    // Find root spans (no parent)
    const rootSpans = state.result.trace.filter((span: ExecutionSpan) => !span.parentSpanId);
    if (rootSpans.length === 0) return 0;
    
    // Sum durations of all root spans
    return rootSpans.reduce((total: number, span: ExecutionSpan) => total + span.durationMs, 0);
  }, [state.result]);

  return {
    state,
    execute,
    reset,
    getSpansByType,
    getTotalCost,
    getTotalTokens,
    getDuration,
  };
}
