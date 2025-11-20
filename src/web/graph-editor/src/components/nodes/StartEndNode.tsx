import { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

export default memo(function StartEndNode({ data, type }: NodeProps) {
  const isStart = type === 'start';
  
  return (
    <div className="rounded-full border border-gray-500 bg-background px-6 py-3 shadow-md">
      {!isStart && <Handle type="target" position={Position.Left} />}
      <div className="font-semibold text-gray-600">
        {isStart ? 'START' : 'END'}
      </div>
      {isStart && <Handle type="source" position={Position.Right} />}
    </div>
  );
});
