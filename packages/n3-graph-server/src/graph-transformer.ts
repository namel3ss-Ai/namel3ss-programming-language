/**
 * Graph Transformer
 * 
 * Converts N3 AST to a graph structure with nodes and edges for visualization.
 * Each N3 component (prompt, chain, agent, RAG, etc.) becomes a node with appropriate metadata.
 */

import type {
  N3Module,
  N3App,
  N3Prompt,
  N3Chain,
  N3ChainStep,
  N3Agent,
  N3RagPipeline,
  N3Index,
  N3LLM,
  N3Tool,
  N3Memory,
  N3Dataset,
} from './parser.js';

export interface GraphNode {
  id: string;
  type: NodeType;
  data: NodeData;
  position: { x: number; y: number };
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: EdgeType;
  label?: string;
  data?: EdgeData;
}

export type NodeType =
  | 'app'
  | 'prompt'
  | 'chain'
  | 'chainStep'
  | 'agent'
  | 'ragPipeline'
  | 'index'
  | 'llm'
  | 'tool'
  | 'memory'
  | 'dataset'
  | 'page';

export type EdgeType = 'default' | 'conditional' | 'step' | 'reference';

export interface NodeData {
  label: string;
  description?: string;
  metadata?: Record<string, any>;
  // Node-specific data
  [key: string]: any;
}

export interface EdgeData {
  condition?: string;
  stepIndex?: number;
  [key: string]: any;
}

export interface Graph {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

let nodeCounter = 0;

function generateNodeId(prefix: string): string {
  return `${prefix}_${nodeCounter++}`;
}

function resetNodeCounter(): void {
  nodeCounter = 0;
}

/**
 * Transform N3 Module AST to graph structure
 */
export function transformToGraph(module: N3Module): Graph {
  resetNodeCounter();
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  
  // Process each body item (should contain App)
  for (const item of module.body) {
    if (item.type === 'App') {
      processApp(item as N3App, nodes, edges);
    }
  }
  
  // Auto-layout nodes if no positions set
  if (nodes.length > 0) {
    applyInitialLayout(nodes);
  }
  
  return { nodes, edges };
}

/**
 * Process an App node and its children
 */
function processApp(app: N3App, nodes: GraphNode[], edges: GraphEdge[]): void {
  const appId = generateNodeId('app');
  
  nodes.push({
    id: appId,
    type: 'app',
    data: {
      label: app.name,
      description: `Application: ${app.name}`,
      metadata: {
        pages: app.pages?.length || 0,
        datasets: app.datasets?.length || 0,
        chains: app.chains?.length || 0,
        agents: app.agents?.length || 0,
      },
    },
    position: { x: 0, y: 0 },
  });
  
  // Process LLMs
  if (app.llms) {
    for (const llm of app.llms) {
      const llmId = processLLM(llm, nodes);
      edges.push({
        id: `${appId}-${llmId}`,
        source: appId,
        target: llmId,
        type: 'reference',
        label: 'defines',
      });
    }
  }
  
  // Process Tools
  if (app.tools) {
    for (const tool of app.tools) {
      const toolId = processTool(tool, nodes);
      edges.push({
        id: `${appId}-${toolId}`,
        source: appId,
        target: toolId,
        type: 'reference',
        label: 'defines',
      });
    }
  }
  
  // Process Datasets
  if (app.datasets) {
    for (const dataset of app.datasets) {
      const datasetId = processDataset(dataset, nodes);
      edges.push({
        id: `${appId}-${datasetId}`,
        source: appId,
        target: datasetId,
        type: 'reference',
        label: 'defines',
      });
    }
  }
  
  // Process Indices
  if (app.indices) {
    for (const index of app.indices) {
      const indexId = processIndex(index, nodes, edges);
      edges.push({
        id: `${appId}-${indexId}`,
        source: appId,
        target: indexId,
        type: 'reference',
        label: 'defines',
      });
    }
  }
  
  // Process Prompts
  if (app.prompts) {
    for (const prompt of app.prompts) {
      const promptId = processPrompt(prompt, nodes);
      edges.push({
        id: `${appId}-${promptId}`,
        source: appId,
        target: promptId,
        type: 'reference',
        label: 'defines',
      });
    }
  }
  
  // Process RAG Pipelines
  if (app.rag_pipelines) {
    for (const ragPipeline of app.rag_pipelines) {
      const ragId = processRagPipeline(ragPipeline, nodes, edges);
      edges.push({
        id: `${appId}-${ragId}`,
        source: appId,
        target: ragId,
        type: 'reference',
        label: 'defines',
      });
    }
  }
  
  // Process Memories
  if (app.memories) {
    for (const memory of app.memories) {
      const memoryId = processMemory(memory, nodes);
      edges.push({
        id: `${appId}-${memoryId}`,
        source: appId,
        target: memoryId,
        type: 'reference',
        label: 'defines',
      });
    }
  }
  
  // Process Chains
  if (app.chains) {
    for (const chain of app.chains) {
      const chainId = processChain(chain, nodes, edges);
      edges.push({
        id: `${appId}-${chainId}`,
        source: appId,
        target: chainId,
        type: 'reference',
        label: 'defines',
      });
    }
  }
  
  // Process Agents
  if (app.agents) {
    for (const agent of app.agents) {
      const agentId = processAgent(agent, nodes, edges);
      edges.push({
        id: `${appId}-${agentId}`,
        source: appId,
        target: agentId,
        type: 'reference',
        label: 'defines',
      });
    }
  }
}

function processLLM(llm: N3LLM, nodes: GraphNode[]): string {
  const id = generateNodeId('llm');
  nodes.push({
    id,
    type: 'llm',
    data: {
      label: llm.name,
      description: `${llm.provider}/${llm.model}`,
      provider: llm.provider,
      model: llm.model,
      temperature: llm.temperature,
      maxTokens: llm.max_tokens,
      metadata: llm.config,
    },
    position: { x: 0, y: 0 },
  });
  return id;
}

function processTool(tool: N3Tool, nodes: GraphNode[]): string {
  const id = generateNodeId('tool');
  nodes.push({
    id,
    type: 'tool',
    data: {
      label: tool.name,
      description: `Tool: ${tool.type}`,
      toolType: tool.type,
      endpoint: tool.endpoint,
      metadata: tool.config,
    },
    position: { x: 0, y: 0 },
  });
  return id;
}

function processDataset(dataset: N3Dataset, nodes: GraphNode[]): string {
  const id = generateNodeId('dataset');
  nodes.push({
    id,
    type: 'dataset',
    data: {
      label: dataset.name,
      description: `Dataset: ${dataset.source}`,
      source: dataset.source,
      metadata: { filter: dataset.filter },
    },
    position: { x: 0, y: 0 },
  });
  return id;
}

function processIndex(index: N3Index, nodes: GraphNode[], edges: GraphEdge[]): string {
  const id = generateNodeId('index');
  nodes.push({
    id,
    type: 'index',
    data: {
      label: index.name,
      description: `Index: ${index.backend}`,
      sourceDataset: index.source_dataset,
      embeddingModel: index.embedding_model,
      chunkSize: index.chunk_size,
      overlap: index.overlap,
      backend: index.backend,
      namespace: index.namespace,
      metadataFields: index.metadata_fields,
    },
    position: { x: 0, y: 0 },
  });
  
  // Create edge to source dataset if it exists in nodes
  const datasetNode = nodes.find(
    (n) => n.type === 'dataset' && n.data.label === index.source_dataset
  );
  if (datasetNode) {
    edges.push({
      id: `${datasetNode.id}-${id}`,
      source: datasetNode.id,
      target: id,
      type: 'reference',
      label: 'indexes',
    });
  }
  
  return id;
}

function processPrompt(prompt: N3Prompt, nodes: GraphNode[]): string {
  const id = generateNodeId('prompt');
  nodes.push({
    id,
    type: 'prompt',
    data: {
      label: prompt.name,
      description: 'Prompt Template',
      inputArgs: prompt.input_args,
      template: prompt.template,
      outputSchema: prompt.output_schema,
    },
    position: { x: 0, y: 0 },
  });
  return id;
}

function processRagPipeline(
  ragPipeline: N3RagPipeline,
  nodes: GraphNode[],
  edges: GraphEdge[]
): string {
  const id = generateNodeId('rag');
  nodes.push({
    id,
    type: 'ragPipeline',
    data: {
      label: ragPipeline.name,
      description: `RAG Pipeline (top_k: ${ragPipeline.top_k})`,
      queryEncoder: ragPipeline.query_encoder,
      index: ragPipeline.index,
      topK: ragPipeline.top_k,
      reranker: ragPipeline.reranker,
      distanceMetric: ragPipeline.distance_metric,
      filters: ragPipeline.filters,
    },
    position: { x: 0, y: 0 },
  });
  
  // Create edge to index
  const indexNode = nodes.find((n) => n.type === 'index' && n.data.label === ragPipeline.index);
  if (indexNode) {
    edges.push({
      id: `${id}-${indexNode.id}`,
      source: id,
      target: indexNode.id,
      type: 'reference',
      label: 'uses',
    });
  }
  
  return id;
}

function processMemory(memory: N3Memory, nodes: GraphNode[]): string {
  const id = generateNodeId('memory');
  nodes.push({
    id,
    type: 'memory',
    data: {
      label: memory.name,
      description: `Memory: ${memory.scope}`,
      scope: memory.scope,
      kind: memory.kind,
      maxItems: memory.max_items,
    },
    position: { x: 0, y: 0 },
  });
  return id;
}

function processChain(chain: N3Chain, nodes: GraphNode[], edges: GraphEdge[]): string {
  const id = generateNodeId('chain');
  nodes.push({
    id,
    type: 'chain',
    data: {
      label: chain.name,
      description: `Chain (${chain.steps?.length || 0} steps)`,
      inputKey: chain.input_key,
      steps: chain.steps,
      metadata: chain.metadata,
    },
    position: { x: 0, y: 0 },
  });
  
  // Process chain steps
  if (chain.steps && chain.steps.length > 0) {
    let prevStepId = id;
    
    for (let i = 0; i < chain.steps.length; i++) {
      const step = chain.steps[i];
      const stepId = processChainStep(step, i, nodes, edges);
      
      // Connect steps sequentially
      edges.push({
        id: `${prevStepId}-${stepId}`,
        source: prevStepId,
        target: stepId,
        type: 'step',
        label: `step ${i + 1}`,
        data: { stepIndex: i },
      });
      
      prevStepId = stepId;
    }
  }
  
  return id;
}

function processChainStep(
  step: N3ChainStep,
  index: number,
  nodes: GraphNode[],
  edges: GraphEdge[]
): string {
  const id = generateNodeId('chainStep');
  nodes.push({
    id,
    type: 'chainStep',
    data: {
      label: step.name || `Step ${index + 1}`,
      description: `${step.kind}: ${step.target}`,
      kind: step.kind,
      target: step.target,
      options: step.options,
      stopOnError: step.stop_on_error,
    },
    position: { x: 0, y: 0 },
  });
  
  // Try to connect to target node if it exists
  const targetNode = nodes.find((n) => n.data.label === step.target);
  if (targetNode) {
    edges.push({
      id: `${id}-${targetNode.id}`,
      source: id,
      target: targetNode.id,
      type: 'reference',
      label: 'invokes',
    });
  }
  
  return id;
}

function processAgent(agent: N3Agent, nodes: GraphNode[], edges: GraphEdge[]): string {
  const id = generateNodeId('agent');
  nodes.push({
    id,
    type: 'agent',
    data: {
      label: agent.name,
      description: `Agent: ${agent.goal}`,
      llmName: agent.llm_name,
      toolNames: agent.tool_names,
      memoryConfig: agent.memory_config,
      goal: agent.goal,
      systemPrompt: agent.system_prompt,
      maxTurns: agent.max_turns,
      metadata: agent.config,
    },
    position: { x: 0, y: 0 },
  });
  
  // Connect to LLM
  const llmNode = nodes.find((n) => n.type === 'llm' && n.data.label === agent.llm_name);
  if (llmNode) {
    edges.push({
      id: `${id}-${llmNode.id}`,
      source: id,
      target: llmNode.id,
      type: 'reference',
      label: 'uses',
    });
  }
  
  // Connect to tools
  if (agent.tool_names) {
    for (const toolName of agent.tool_names) {
      const toolNode = nodes.find((n) => n.type === 'tool' && n.data.label === toolName);
      if (toolNode) {
        edges.push({
          id: `${id}-${toolNode.id}`,
          source: id,
          target: toolNode.id,
          type: 'reference',
          label: 'uses',
        });
      }
    }
  }
  
  return id;
}

/**
 * Apply a simple initial layout to nodes
 * Groups nodes by type and arranges them in a grid
 */
function applyInitialLayout(nodes: GraphNode[]): void {
  const typeGroups: Record<string, GraphNode[]> = {};
  
  // Group nodes by type
  for (const node of nodes) {
    if (!typeGroups[node.type]) {
      typeGroups[node.type] = [];
    }
    typeGroups[node.type].push(node);
  }
  
  const horizontalSpacing = 300;
  const verticalSpacing = 150;
  let currentX = 50;
  
  // Layout each type group
  const typeOrder = [
    'app',
    'dataset',
    'index',
    'llm',
    'tool',
    'memory',
    'prompt',
    'ragPipeline',
    'agent',
    'chain',
    'chainStep',
  ];
  
  for (const type of typeOrder) {
    const group = typeGroups[type];
    if (!group) continue;
    
    let currentY = 50;
    for (const node of group) {
      node.position = { x: currentX, y: currentY };
      currentY += verticalSpacing;
    }
    
    currentX += horizontalSpacing;
  }
}

export { transformToGraph as default };
