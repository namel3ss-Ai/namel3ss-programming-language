import { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

export default memo(function RAGNode({ data }: NodeProps) {
  return (
    <div className="rounded-md border border-green-500 bg-background p-4 shadow-md">
      <Handle type="target" position={Position.Left} />
      <div className="font-semibold text-green-600">{data.label || 'RAG'}</div>
      <div className="mt-2 text-xs text-muted-foreground">
        {data.description || 'RAG dataset'}
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
});
