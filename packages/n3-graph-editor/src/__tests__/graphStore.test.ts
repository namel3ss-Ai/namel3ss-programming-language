/**
 * Graph Store Tests
 */

import { renderHook, act } from '@testing-library/react';
import { useGraphStore } from '../store/graphStore';

describe('GraphStore', () => {
  beforeEach(() => {
    const { result } = renderHook(() => useGraphStore());
    act(() => {
      result.current.setNodes([]);
      result.current.setEdges([]);
    });
  });

  it('should initialize with empty state', () => {
    const { result } = renderHook(() => useGraphStore());
    expect(result.current.nodes).toEqual([]);
    expect(result.current.edges).toEqual([]);
    expect(result.current.selectedNodeIds.size).toBe(0);
  });

  it('should add nodes', () => {
    const { result } = renderHook(() => useGraphStore());
    
    const testNode = {
      id: 'test-1',
      type: 'prompt',
      data: { label: 'Test Prompt' },
      position: { x: 0, y: 0 },
    };

    act(() => {
      result.current.addNode(testNode);
    });

    expect(result.current.nodes).toHaveLength(1);
    expect(result.current.nodes[0].id).toBe('test-1');
  });

  it('should select and deselect nodes', () => {
    const { result } = renderHook(() => useGraphStore());

    act(() => {
      result.current.selectNode('node-1');
    });

    expect(result.current.selectedNodeIds.has('node-1')).toBe(true);

    act(() => {
      result.current.deselectNode('node-1');
    });

    expect(result.current.selectedNodeIds.has('node-1')).toBe(false);
  });

  it('should clear selection', () => {
    const { result } = renderHook(() => useGraphStore());

    act(() => {
      result.current.selectNode('node-1');
      result.current.selectNode('node-2');
    });

    expect(result.current.selectedNodeIds.size).toBe(2);

    act(() => {
      result.current.clearSelection();
    });

    expect(result.current.selectedNodeIds.size).toBe(0);
  });

  it('should update node data', () => {
    const { result } = renderHook(() => useGraphStore());
    
    const testNode = {
      id: 'test-1',
      type: 'prompt',
      data: { label: 'Original' },
      position: { x: 0, y: 0 },
    };

    act(() => {
      result.current.addNode(testNode);
      result.current.updateNode('test-1', { label: 'Updated' });
    });

    const updatedNode = result.current.nodes.find((n) => n.id === 'test-1');
    expect(updatedNode?.data.label).toBe('Updated');
  });

  it('should delete nodes and related edges', () => {
    const { result } = renderHook(() => useGraphStore());
    
    const node1 = {
      id: 'node-1',
      type: 'prompt',
      data: { label: 'Node 1' },
      position: { x: 0, y: 0 },
    };

    const node2 = {
      id: 'node-2',
      type: 'chain',
      data: { label: 'Node 2' },
      position: { x: 100, y: 0 },
    };

    const edge = {
      id: 'edge-1',
      source: 'node-1',
      target: 'node-2',
      type: 'reference',
    };

    act(() => {
      result.current.setNodes([node1, node2]);
      result.current.setEdges([edge]);
      result.current.deleteNode('node-1');
    });

    expect(result.current.nodes).toHaveLength(1);
    expect(result.current.edges).toHaveLength(0);
  });
});
