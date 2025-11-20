/**
 * GraphEditor Component
 * 
 * Main React Flow graph editor with:
 * - Custom node types for N3 entities
 * - Real-time collaboration via Yjs
 * - Keyboard navigation and accessibility
 * - Auto-layout support
 */

import React, { useCallback, useEffect, useRef } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  ReactFlowProvider,
  useReactFlow,
} from 'reactflow';
import 'reactflow/dist/style.css';

import { useGraphStore } from '../store/graphStore';
import { nodeTypes } from './nodes/CustomNodes';
import { Sidebar } from './Sidebar';
import { Toolbar } from './Toolbar';

const GraphEditorInner: React.FC = () => {
  const reactFlowInstance = useReactFlow();
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  
  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    selectedNodeIds,
    selectNode,
    clearSelection,
    initYjs,
    destroyYjs,
    currentFile,
  } = useGraphStore();
  
  // Initialize Yjs collaboration
  useEffect(() => {
    const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:3001';
    const docName = currentFile || 'default-graph';
    
    initYjs(docName, wsUrl);
    
    return () => {
      destroyYjs();
    };
  }, [currentFile, initYjs, destroyYjs]);
  
  // Handle node selection
  const onNodeClick = useCallback(
    (_event: React.MouseEvent, node: any) => {
      selectNode(node.id);
    },
    [selectNode]
  );
  
  // Handle pane click (deselect all)
  const onPaneClick = useCallback(() => {
    clearSelection();
  }, [clearSelection]);
  
  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Tab: Focus next node
      if (event.key === 'Tab') {
        event.preventDefault();
        // TODO: Implement tab navigation
      }
      
      // Arrow keys: Move selected nodes
      if (event.key.startsWith('Arrow')) {
        event.preventDefault();
        // TODO: Implement arrow key movement
      }
      
      // Delete: Remove selected nodes
      if (event.key === 'Delete' || event.key === 'Backspace') {
        event.preventDefault();
        // TODO: Implement node deletion
      }
    };
    
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);
  
  return (
    <div className="flex h-full w-full" ref={reactFlowWrapper}>
      <Sidebar />
      
      <div className="flex-1 relative">
        <Toolbar />
        
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={onNodeClick}
          onPaneClick={onPaneClick}
          nodeTypes={nodeTypes}
          fitView
          onlyRenderVisibleElements
          attributionPosition="bottom-right"
          // Accessibility
          nodesFocusable
          edgesFocusable
          elementsSelectable
        >
          <Background />
          <Controls />
          <MiniMap
            nodeColor={(node) => {
              const colors: Record<string, string> = {
                app: '#8b5cf6',
                prompt: '#3b82f6',
                chain: '#10b981',
                agent: '#f59e0b',
                ragPipeline: '#ec4899',
                llm: '#6366f1',
                tool: '#14b8a6',
                memory: '#a855f7',
                dataset: '#84cc16',
                index: '#f97316',
              };
              return colors[node.type || 'default'] || '#9ca3af';
            }}
          />
        </ReactFlow>
        
        {/* Accessibility: Live region for announcements */}
        <div
          role="status"
          aria-live="polite"
          aria-atomic="true"
          className="visually-hidden"
        >
          {selectedNodeIds.size > 0
            ? `${selectedNodeIds.size} node${selectedNodeIds.size > 1 ? 's' : ''} selected`
            : 'No nodes selected'}
        </div>
      </div>
    </div>
  );
};

export const GraphEditor: React.FC = () => {
  return (
    <ReactFlowProvider>
      <GraphEditorInner />
    </ReactFlowProvider>
  );
};
