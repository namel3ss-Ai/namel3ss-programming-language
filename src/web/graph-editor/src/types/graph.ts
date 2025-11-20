// Core graph types
export type NodeType =
  | 'prompt'
  | 'pythonHook'
  | 'agent'
  | 'agentGraph'
  | 'ragDataset'
  | 'memoryStore'
  | 'safetyPolicy'
  | 'condition'
  | 'start'
  | 'end';

export interface GraphNode {
  id: string;
  type: NodeType;
  label: string;
  data: Record<string, any>;
  position?: { x: number; y: number };
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label?: string;
  conditionExpr?: string;
}

export interface ChainInfo {
  id: string;
  name: string;
}

export interface AgentInfo {
  id: string;
  name: string;
}

export interface GraphResponse {
  projectId: string;
  name: string;
  chains: ChainInfo[];
  agents: AgentInfo[];
  activeRootId: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  metadata: Record<string, any>;
}

export interface GraphUpdatePayload {
  nodes: GraphNode[];
  edges: GraphEdge[];
  metadata: Record<string, any>;
}

// Execution types
export interface ExecutionRequest {
  entry: string;
  input: Record<string, any>;
  options?: Record<string, any>;
}

export interface SpanAttribute {
  model?: string;
  temperature?: number;
  tokensPrompt?: number;
  tokensCompletion?: number;
  cost?: number;
  [key: string]: any;
}

export interface ExecutionSpan {
  spanId: string;
  parentSpanId: string | null;
  name: string;
  type: 'llm.call' | 'tool.call' | 'rag.retrieve' | 'agent.step' | 'chain.step';
  startTime: string;
  endTime: string;
  durationMs: number;
  status: 'ok' | 'error';
  attributes: SpanAttribute;
  input?: any;
  output?: any;
}

export interface ExecutionResult {
  result: any;
  trace: ExecutionSpan[];
}

// Share types
export interface ShareLink {
  id: string;
  projectId: string;
  token: string;
  url: string;
  role: 'viewer' | 'editor';
  createdAt: string;
  expiresAt: string | null;
  createdByUserId?: string;
}

export interface CreateShareRequest {
  role: 'viewer' | 'editor';
  expiresInHours?: number | null;
}

export interface ShareTokenValidation {
  projectId: string;
  role: 'viewer' | 'editor';
}

// Collaboration types
export interface UserPresence {
  userId: string;
  displayName: string;
  color: string;
  selectedNodeId?: string;
  cursor?: { x: number; y: number };
}

// Tool types
export interface ToolMetadata {
  name: string;
  description: string;
  inputSchema: Record<string, any>;
  outputSchema?: Record<string, any>;
  tags?: string[];
  source: 'function' | 'openapi' | 'langchain' | 'class';
}

export interface ToolExecutionRequest {
  name: string;
  args: Record<string, any>;
  async?: boolean;
}

export interface ToolExecutionResult {
  success: boolean;
  result?: any;
  error?: string;
  durationMs: number;
}

// Policy types
export interface PolicyMetadata {
  id: string;
  agentId: string;
  version: string;
  createdAt: string;
  trainedOn?: {
    feedbackCount: number;
    rewardMean: number;
    rewardStd: number;
  };
}

export interface FeedbackSubmission {
  prompt: string;
  response: string;
  score: number;
  notes?: string | null;
  runId: string;
}

export interface PolicyTrainingRequest {
  dryRun?: boolean;
  maxSteps?: number;
  learningRate?: number;
}
