/**
 * Graph Store with Yjs Integration
 * 
 * Zustand store that manages graph state (nodes, edges) and integrates with Yjs
 * for real-time collaborative editing across multiple clients.
 */

import { create } from 'zustand';
import * as Y from 'yjs';
import { WebsocketProvider } from 'y-websocket';
import type { Node, Edge, NodeChange, EdgeChange } from 'reactflow';
import { applyNodeChanges, applyEdgeChanges } from 'reactflow';

export interface GraphState {
  nodes: Node[];
  edges: Edge[];
  selectedNodeIds: Set<string>;
  
  // Yjs collaboration
  ydoc: Y.Doc | null;
  provider: WebsocketProvider | null;
  awareness: any | null;
  
  // Actions
  setNodes: (nodes: Node[]) => void;
  setEdges: (edges: Edge[]) => void;
  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  addNode: (node: Node) => void;
  updateNode: (id: string, data: Partial<Node['data']>) => void;
  deleteNode: (id: string) => void;
  selectNode: (id: string) => void;
  deselectNode: (id: string) => void;
  clearSelection: () => void;
  
  // Yjs setup
  initYjs: (docName: string, wsUrl: string) => void;
  destroyYjs: () => void;
  
  // Layout
  autoLayout: () => void;
  
  // File management
  currentFile: string | null;
  setCurrentFile: (file: string | null) => void;
}

export const useGraphStore = create<GraphState>((set, get) => ({
  nodes: [],
  edges: [],
  selectedNodeIds: new Set(),
  
  ydoc: null,
  provider: null,
  awareness: null,
  
  currentFile: null,
  
  setNodes: (nodes) => {
    set({ nodes });
    
    // Sync to Yjs
    const { ydoc } = get();
    if (ydoc) {
      const yNodes = ydoc.getArray('nodes');
      ydoc.transact(() => {
        yNodes.delete(0, yNodes.length);
        nodes.forEach((node) => yNodes.push([node]));
      });
    }
  },
  
  setEdges: (edges) => {
    set({ edges });
    
    // Sync to Yjs
    const { ydoc } = get();
    if (ydoc) {
      const yEdges = ydoc.getArray('edges');
      ydoc.transact(() => {
        yEdges.delete(0, yEdges.length);
        edges.forEach((edge) => yEdges.push([edge]));
      });
    }
  },
  
  onNodesChange: (changes) => {
    set((state) => ({
      nodes: applyNodeChanges(changes, state.nodes),
    }));
    
    // Sync changes to Yjs
    const { nodes, ydoc } = get();
    if (ydoc) {
      const yNodes = ydoc.getArray('nodes');
      ydoc.transact(() => {
        yNodes.delete(0, yNodes.length);
        nodes.forEach((node) => yNodes.push([node]));
      });
    }
  },
  
  onEdgesChange: (changes) => {
    set((state) => ({
      edges: applyEdgeChanges(changes, state.edges),
    }));
    
    // Sync changes to Yjs
    const { edges, ydoc } = get();
    if (ydoc) {
      const yEdges = ydoc.getArray('edges');
      ydoc.transact(() => {
        yEdges.delete(0, yEdges.length);
        edges.forEach((edge) => yEdges.push([edge]));
      });
    }
  },
  
  addNode: (node) => {
    set((state) => ({
      nodes: [...state.nodes, node],
    }));
    
    const { ydoc } = get();
    if (ydoc) {
      const yNodes = ydoc.getArray('nodes');
      yNodes.push([node]);
    }
  },
  
  updateNode: (id, data) => {
    set((state) => ({
      nodes: state.nodes.map((node) =>
        node.id === id ? { ...node, data: { ...node.data, ...data } } : node
      ),
    }));
    
    const { nodes, ydoc } = get();
    if (ydoc) {
      const yNodes = ydoc.getArray('nodes');
      ydoc.transact(() => {
        yNodes.delete(0, yNodes.length);
        nodes.forEach((node) => yNodes.push([node]));
      });
    }
  },
  
  deleteNode: (id) => {
    set((state) => ({
      nodes: state.nodes.filter((node) => node.id !== id),
      edges: state.edges.filter((edge) => edge.source !== id && edge.target !== id),
    }));
    
    const { nodes, edges, ydoc } = get();
    if (ydoc) {
      ydoc.transact(() => {
        const yNodes = ydoc.getArray('nodes');
        const yEdges = ydoc.getArray('edges');
        yNodes.delete(0, yNodes.length);
        yEdges.delete(0, yEdges.length);
        nodes.forEach((node) => yNodes.push([node]));
        edges.forEach((edge) => yEdges.push([edge]));
      });
    }
  },
  
  selectNode: (id) => {
    set((state) => {
      const newSet = new Set(state.selectedNodeIds);
      newSet.add(id);
      return { selectedNodeIds: newSet };
    });
  },
  
  deselectNode: (id) => {
    set((state) => {
      const newSet = new Set(state.selectedNodeIds);
      newSet.delete(id);
      return { selectedNodeIds: newSet };
    });
  },
  
  clearSelection: () => {
    set({ selectedNodeIds: new Set() });
  },
  
  initYjs: (docName, wsUrl) => {
    const ydoc = new Y.Doc();
    const provider = new WebsocketProvider(wsUrl, docName, ydoc);
    const awareness = provider.awareness;
    
    // Initialize Yjs arrays
    const yNodes = ydoc.getArray('nodes');
    const yEdges = ydoc.getArray('edges');
    
    // Listen for remote updates
    yNodes.observe(() => {
      const nodes = yNodes.toArray() as Node[];
      set({ nodes });
    });
    
    yEdges.observe(() => {
      const edges = yEdges.toArray() as Edge[];
      set({ edges });
    });
    
    // Set local state
    set({ ydoc, provider, awareness });
    
    // Set user info in awareness
    awareness.setLocalStateField('user', {
      name: `User-${Math.floor(Math.random() * 1000)}`,
      color: `#${Math.floor(Math.random() * 16777215).toString(16)}`,
    });
  },
  
  destroyYjs: () => {
    const { provider, ydoc } = get();
    
    if (provider) {
      provider.destroy();
    }
    
    if (ydoc) {
      ydoc.destroy();
    }
    
    set({ ydoc: null, provider: null, awareness: null });
  },
  
  autoLayout: () => {
    // Import dagre dynamically for auto-layout
    import('dagre').then(({ default: dagre }) => {
      const { nodes, edges } = get();
      
      const dagreGraph = new dagre.graphlib.Graph();
      dagreGraph.setDefaultEdgeLabel(() => ({}));
      dagreGraph.setGraph({ rankdir: 'TB', nodesep: 100, ranksep: 150 });
      
      // Add nodes and edges to dagre
      nodes.forEach((node) => {
        dagreGraph.setNode(node.id, { width: 180, height: 80 });
      });
      
      edges.forEach((edge) => {
        dagreGraph.setEdge(edge.source, edge.target);
      });
      
      // Calculate layout
      dagre.layout(dagreGraph);
      
      // Apply new positions
      const layoutedNodes = nodes.map((node) => {
        const nodeWithPosition = dagreGraph.node(node.id);
        return {
          ...node,
          position: {
            x: nodeWithPosition.x - 90,
            y: nodeWithPosition.y - 40,
          },
        };
      });
      
      set({ nodes: layoutedNodes });
    });
  },
  
  setCurrentFile: (file) => {
    set({ currentFile: file });
  },
}));
