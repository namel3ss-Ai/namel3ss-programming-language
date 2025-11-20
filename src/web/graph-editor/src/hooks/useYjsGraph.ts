import { useState, useEffect, useCallback } from 'react';
import * as Y from 'yjs';
import { WebsocketProvider } from 'y-websocket';
import { Node, Edge } from 'reactflow';
import { UserPresence } from '@/types/graph';

interface UseYjsGraphProps {
  projectId: string;
  nodes: Node[];
  edges: Edge[];
  onRemoteUpdate: (nodes: Node[], edges: Edge[]) => void;
}

export function useYjsGraph({
  projectId,
  nodes,
  edges,
  onRemoteUpdate,
}: UseYjsGraphProps) {
  const [ydoc] = useState(() => new Y.Doc());
  const [provider, setProvider] = useState<WebsocketProvider | null>(null);
  const [users, setUsers] = useState<UserPresence[]>([]);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const wsUrl = import.meta.env.VITE_YJS_URL || 'ws://localhost:1234';
    const newProvider = new WebsocketProvider(wsUrl, projectId, ydoc);

    newProvider.on('status', ({ status }: { status: string }) => {
      setIsConnected(status === 'connected');
    });

    // Awareness for user presence
    const awareness = newProvider.awareness;
    awareness.setLocalStateField('user', {
      userId: `user-${Math.random().toString(36).substr(2, 9)}`,
      displayName: 'User',
      color: `#${Math.floor(Math.random() * 16777215).toString(16)}`,
    });

    awareness.on('change', () => {
      const states = Array.from(awareness.getStates().values());
      setUsers(
        states
          .filter((state) => state.user)
          .map((state) => state.user as UserPresence)
      );
    });

    setProvider(newProvider);

    return () => {
      newProvider.destroy();
    };
  }, [projectId, ydoc]);

  useEffect(() => {
    if (!provider) return;

    const yNodes = ydoc.getArray('nodes');
    const yEdges = ydoc.getArray('edges');

    const observer = () => {
      const remoteNodes = yNodes.toArray() as Node[];
      const remoteEdges = yEdges.toArray() as Edge[];
      onRemoteUpdate(remoteNodes, remoteEdges);
    };

    yNodes.observe(observer);
    yEdges.observe(observer);

    return () => {
      yNodes.unobserve(observer);
      yEdges.unobserve(observer);
    };
  }, [provider, ydoc, onRemoteUpdate]);

  const sync = useCallback(
    (currentNodes: Node[], currentEdges: Edge[]) => {
      const yNodes = ydoc.getArray('nodes');
      const yEdges = ydoc.getArray('edges');

      ydoc.transact(() => {
        yNodes.delete(0, yNodes.length);
        yNodes.push(currentNodes);
        yEdges.delete(0, yEdges.length);
        yEdges.push(currentEdges);
      });
    },
    [ydoc]
  );

  return { users, sync, isConnected };
}
