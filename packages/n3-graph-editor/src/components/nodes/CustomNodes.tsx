/**
 * Custom Node Components for React Flow
 * 
 * Defines specialized node components for each N3 entity type
 * with accessible ARIA labels and semantic markup.
 */

import React, { memo } from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';

// Base node component with accessibility support
interface BaseNodeProps extends NodeProps {
  type: string;
  ariaLabel?: string;
}

const BaseNode: React.FC<BaseNodeProps> = memo(({ data, type, ariaLabel, selected }) => {
  const nodeClass = `n3-node n3-node-${type} ${selected ? 'selected' : ''}`;
  
  return (
    <div
      className={nodeClass}
      role="region"
      aria-label={ariaLabel || data.label}
      tabIndex={0}
    >
      <Handle type="target" position={Position.Top} />
      
      <div className="n3-node-header">{data.label}</div>
      
      {data.description && (
        <div className="n3-node-description">{data.description}</div>
      )}
      
      {data.metadata && Object.keys(data.metadata).length > 0 && (
        <div className="n3-node-badge" style={{ backgroundColor: '#e5e7eb', color: '#374151' }}>
          {Object.entries(data.metadata)
            .slice(0, 1)
            .map(([key, value]) => `${key}: ${value}`)
            .join(', ')}
        </div>
      )}
      
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
});

BaseNode.displayName = 'BaseNode';

// App Node
export const AppNode: React.FC<NodeProps> = memo((props) => (
  <BaseNode {...props} type="app" ariaLabel={`Application: ${props.data.label}`} />
));

AppNode.displayName = 'AppNode';

// Prompt Node
export const PromptNode: React.FC<NodeProps> = memo((props) => {
  return (
    <div
      className={`n3-node n3-node-prompt ${props.selected ? 'selected' : ''}`}
      role="region"
      aria-label={`Prompt: ${props.data.label}`}
      tabIndex={0}
    >
      <Handle type="target" position={Position.Top} />
      
      <div className="n3-node-header">📝 {props.data.label}</div>
      <div className="n3-node-description">{props.data.description}</div>
      
      {props.data.inputArgs && props.data.inputArgs.length > 0 && (
        <div className="n3-node-badge" style={{ backgroundColor: '#dbeafe', color: '#1e40af' }}>
          {props.data.inputArgs.length} input{props.data.inputArgs.length > 1 ? 's' : ''}
        </div>
      )}
      
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
});

PromptNode.displayName = 'PromptNode';

// Chain Node
export const ChainNode: React.FC<NodeProps> = memo((props) => {
  return (
    <div
      className={`n3-node n3-node-chain ${props.selected ? 'selected' : ''}`}
      role="region"
      aria-label={`Chain: ${props.data.label}`}
      tabIndex={0}
    >
      <Handle type="target" position={Position.Top} />
      
      <div className="n3-node-header">⛓️ {props.data.label}</div>
      <div className="n3-node-description">{props.data.description}</div>
      
      {props.data.steps && (
        <div className="n3-node-badge" style={{ backgroundColor: '#d1fae5', color: '#065f46' }}>
          {props.data.steps.length} step{props.data.steps.length > 1 ? 's' : ''}
        </div>
      )}
      
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
});

ChainNode.displayName = 'ChainNode';

// Chain Step Node
export const ChainStepNode: React.FC<NodeProps> = memo((props) => {
  return (
    <div
      className={`n3-node n3-node-chainStep ${props.selected ? 'selected' : ''}`}
      role="region"
      aria-label={`Chain Step: ${props.data.label}`}
      tabIndex={0}
      style={{ minWidth: '150px' }}
    >
      <Handle type="target" position={Position.Top} />
      
      <div className="n3-node-header" style={{ fontSize: '12px' }}>
        {props.data.label}
      </div>
      <div className="n3-node-description" style={{ fontSize: '11px' }}>
        {props.data.description}
      </div>
      
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
});

ChainStepNode.displayName = 'ChainStepNode';

// Agent Node
export const AgentNode: React.FC<NodeProps> = memo((props) => {
  return (
    <div
      className={`n3-node n3-node-agent ${props.selected ? 'selected' : ''}`}
      role="region"
      aria-label={`Agent: ${props.data.label}`}
      tabIndex={0}
    >
      <Handle type="target" position={Position.Top} />
      
      <div className="n3-node-header">🤖 {props.data.label}</div>
      <div className="n3-node-description">{props.data.description}</div>
      
      {props.data.toolNames && props.data.toolNames.length > 0 && (
        <div className="n3-node-badge" style={{ backgroundColor: '#fed7aa', color: '#92400e' }}>
          {props.data.toolNames.length} tool{props.data.toolNames.length > 1 ? 's' : ''}
        </div>
      )}
      
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
});

AgentNode.displayName = 'AgentNode';

// RAG Pipeline Node
export const RagPipelineNode: React.FC<NodeProps> = memo((props) => {
  return (
    <div
      className={`n3-node n3-node-ragPipeline ${props.selected ? 'selected' : ''}`}
      role="region"
      aria-label={`RAG Pipeline: ${props.data.label}`}
      tabIndex={0}
    >
      <Handle type="target" position={Position.Top} />
      
      <div className="n3-node-header">🔍 {props.data.label}</div>
      <div className="n3-node-description">{props.data.description}</div>
      
      {props.data.topK && (
        <div className="n3-node-badge" style={{ backgroundColor: '#fce7f3', color: '#831843' }}>
          top_k: {props.data.topK}
        </div>
      )}
      
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
});

RagPipelineNode.displayName = 'RagPipelineNode';

// Index Node
export const IndexNode: React.FC<NodeProps> = memo((props) => {
  return (
    <div
      className={`n3-node n3-node-index ${props.selected ? 'selected' : ''}`}
      role="region"
      aria-label={`Index: ${props.data.label}`}
      tabIndex={0}
    >
      <Handle type="target" position={Position.Top} />
      
      <div className="n3-node-header">📚 {props.data.label}</div>
      <div className="n3-node-description">{props.data.description}</div>
      
      {props.data.backend && (
        <div className="n3-node-badge" style={{ backgroundColor: '#ffedd5', color: '#9a3412' }}>
          {props.data.backend}
        </div>
      )}
      
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
});

IndexNode.displayName = 'IndexNode';

// LLM Node
export const LLMNode: React.FC<NodeProps> = memo((props) => {
  return (
    <div
      className={`n3-node n3-node-llm ${props.selected ? 'selected' : ''}`}
      role="region"
      aria-label={`LLM: ${props.data.label}`}
      tabIndex={0}
    >
      <Handle type="target" position={Position.Top} />
      
      <div className="n3-node-header">🧠 {props.data.label}</div>
      <div className="n3-node-description">{props.data.description}</div>
      
      {props.data.temperature !== undefined && (
        <div className="n3-node-badge" style={{ backgroundColor: '#e0e7ff', color: '#3730a3' }}>
          temp: {props.data.temperature}
        </div>
      )}
      
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
});

LLMNode.displayName = 'LLMNode';

// Tool Node
export const ToolNode: React.FC<NodeProps> = memo((props) => (
  <div
    className={`n3-node n3-node-tool ${props.selected ? 'selected' : ''}`}
    role="region"
    aria-label={`Tool: ${props.data.label}`}
    tabIndex={0}
  >
    <Handle type="target" position={Position.Top} />
    
    <div className="n3-node-header">🔧 {props.data.label}</div>
    <div className="n3-node-description">{props.data.description}</div>
    
    {props.data.toolType && (
      <div className="n3-node-badge" style={{ backgroundColor: '#ccfbf1', color: '#134e4a' }}>
        {props.data.toolType}
      </div>
    )}
    
    <Handle type="source" position={Position.Bottom} />
  </div>
));

ToolNode.displayName = 'ToolNode';

// Memory Node
export const MemoryNode: React.FC<NodeProps> = memo((props) => (
  <div
    className={`n3-node n3-node-memory ${props.selected ? 'selected' : ''}`}
    role="region"
    aria-label={`Memory: ${props.data.label}`}
    tabIndex={0}
  >
    <Handle type="target" position={Position.Top} />
    
    <div className="n3-node-header">💾 {props.data.label}</div>
    <div className="n3-node-description">{props.data.description}</div>
    
    {props.data.scope && (
      <div className="n3-node-badge" style={{ backgroundColor: '#f3e8ff', color: '#581c87' }}>
        {props.data.scope}
      </div>
    )}
    
    <Handle type="source" position={Position.Bottom} />
  </div>
));

MemoryNode.displayName = 'MemoryNode';

// Dataset Node
export const DatasetNode: React.FC<NodeProps> = memo((props) => (
  <div
    className={`n3-node n3-node-dataset ${props.selected ? 'selected' : ''}`}
    role="region"
    aria-label={`Dataset: ${props.data.label}`}
    tabIndex={0}
  >
    <Handle type="target" position={Position.Top} />
    
    <div className="n3-node-header">📊 {props.data.label}</div>
    <div className="n3-node-description">{props.data.description}</div>
    
    {props.data.source && (
      <div className="n3-node-badge" style={{ backgroundColor: '#ecfccb', color: '#3f6212' }}>
        {props.data.source}
      </div>
    )}
    
    <Handle type="source" position={Position.Bottom} />
  </div>
));

DatasetNode.displayName = 'DatasetNode';

// Export all node types
export const nodeTypes = {
  app: AppNode,
  prompt: PromptNode,
  chain: ChainNode,
  chainStep: ChainStepNode,
  agent: AgentNode,
  ragPipeline: RagPipelineNode,
  index: IndexNode,
  llm: LLMNode,
  tool: ToolNode,
  memory: MemoryNode,
  dataset: DatasetNode,
};
