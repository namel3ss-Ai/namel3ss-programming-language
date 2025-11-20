import { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

export default memo(function PromptNode({ data }: NodeProps) {
  return (
    <div className="rounded-md border border-blue-500 bg-background p-4 shadow-md">
      <Handle type="target" position={Position.Left} />
      <div className="font-semibold text-blue-600">{data.label || 'Prompt'}</div>
      <div className="mt-2 text-xs text-muted-foreground">
        {data.text?.substring(0, 50) || 'Empty prompt'}
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
});
