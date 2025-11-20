import { useCallback, useEffect } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  Node,
  Edge,
  Connection,
  addEdge,
  useNodesState,
  useEdgesState,
  NodeTypes,
  Panel,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { GraphResponse } from '@/types/graph';
import { useYjsGraph } from '@/hooks/useYjsGraph';
import PromptNode from './nodes/PromptNode';
import AgentNode from './nodes/AgentNode';
import RAGNode from './nodes/RAGNode';
import ConditionNode from './nodes/ConditionNode';
import StartEndNode from './nodes/StartEndNode';
import CollaborationPanel from './CollaborationPanel';

interface GraphCanvasProps {
  projectId: string;
  initialGraph: GraphResponse;
}

const nodeTypes: NodeTypes = {
  prompt: PromptNode,
  agent: AgentNode,
  agentGraph: AgentNode,
  pythonHook: AgentNode,
  ragDataset: RAGNode,
  memoryStore: RAGNode,
  safetyPolicy: RAGNode,
  condition: ConditionNode,
  start: StartEndNode,
  end: StartEndNode,
};

export default function GraphCanvas({ projectId, initialGraph }: GraphCanvasProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState(
    initialGraph.nodes.map((n) => ({
      id: n.id,
      type: n.type,
      position: n.position || { x: 0, y: 0 },
      data: { ...n.data, label: n.label },
    }))
  );

  const [edges, setEdges, onEdgesChange] = useEdgesState(
    initialGraph.edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      label: e.label,
      data: { conditionExpr: e.conditionExpr },
    }))
  );

  const { users, sync, isConnected } = useYjsGraph({
    projectId,
    nodes,
    edges,
    onRemoteUpdate: (remoteNodes, remoteEdges) => {
      setNodes(remoteNodes);
      setEdges(remoteEdges);
    },
  });

  useEffect(() => {
    sync(nodes, edges);
  }, [nodes, edges, sync]);

  const onConnect = useCallback(
    (connection: Connection) => {
      setEdges((eds) => addEdge(connection, eds));
    },
    [setEdges]
  );

  return (
    <div className="relative flex-1">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
        fitView
      >
        <Background />
        <Controls />
        <MiniMap />
        <Panel position="top-right">
          <CollaborationPanel users={users} isConnected={isConnected} />
        </Panel>
      </ReactFlow>
    </div>
  );
}
