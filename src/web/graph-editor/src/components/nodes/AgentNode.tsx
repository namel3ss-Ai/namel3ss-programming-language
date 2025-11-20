import { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

export default memo(function AgentNode({ data }: NodeProps) {
  return (
    <div className="rounded-md border border-purple-500 bg-background p-4 shadow-md">
      <Handle type="target" position={Position.Left} />
      <div className="font-semibold text-purple-600">{data.label || 'Agent'}</div>
      <div className="mt-2 text-xs text-muted-foreground">
        {data.description || 'Agent node'}
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
});
