import { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

export default memo(function ConditionNode({ data }: NodeProps) {
  return (
    <div className="rounded-md border border-yellow-500 bg-background p-4 shadow-md">
      <Handle type="target" position={Position.Left} />
      <div className="font-semibold text-yellow-600">Condition</div>
      <div className="mt-2 text-xs text-muted-foreground">
        {data.expression || 'if condition'}
      </div>
      <Handle type="source" position={Position.Right} id="true" />
      <Handle type="source" position={Position.Bottom} id="false" />
    </div>
  );
});
