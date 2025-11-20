/**
 * Sidebar Component
 * 
 * Displays:
 * - List of available .n3 files
 * - Selected node properties (editable)
 * - Graph structure overview
 */

import React, { useState, useEffect } from 'react';
import { useGraphStore } from '../store/graphStore';
import axios from 'axios';

export const Sidebar: React.FC = () => {
  const [files, setFiles] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedNode, setSelectedNode] = useState<any>(null);
  
  const { nodes, selectedNodeIds, currentFile, setCurrentFile, setNodes, setEdges } =
    useGraphStore();
  
  // Fetch available .n3 files
  useEffect(() => {
    fetchFiles();
  }, []);
  
  const fetchFiles = async () => {
    try {
      const response = await axios.get('/api/files');
      setFiles(response.data.files || []);
    } catch (error) {
      console.error('Failed to fetch files:', error);
    }
  };
  
  // Load a file
  const loadFile = async (file: string) => {
    setLoading(true);
    try {
      const response = await axios.get('/api/parse', { params: { file } });
      const { graph } = response.data;
      
      setNodes(graph.nodes);
      setEdges(graph.edges);
      setCurrentFile(file);
    } catch (error) {
      console.error('Failed to load file:', error);
    } finally {
      setLoading(false);
    }
  };
  
  // Update selected node when selection changes
  useEffect(() => {
    if (selectedNodeIds.size === 1) {
      const nodeId = Array.from(selectedNodeIds)[0];
      const node = nodes.find((n) => n.id === nodeId);
      setSelectedNode(node || null);
    } else {
      setSelectedNode(null);
    }
  }, [selectedNodeIds, nodes]);
  
  return (
    <div className="w-80 bg-white border-r border-gray-200 flex flex-col overflow-hidden">
      {/* Files Section */}
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-lg font-semibold mb-2">N3 Files</h2>
        
        {loading && <div className="text-sm text-gray-500">Loading...</div>}
        
        <div className="space-y-1 max-h-48 overflow-y-auto">
          {files.map((file) => (
            <button
              key={file}
              onClick={() => loadFile(file)}
              className={`w-full text-left px-3 py-2 rounded text-sm hover:bg-gray-100 ${
                currentFile === file ? 'bg-blue-50 text-blue-600 font-medium' : 'text-gray-700'
              }`}
            >
              {file}
            </button>
          ))}
        </div>
        
        {files.length === 0 && !loading && (
          <div className="text-sm text-gray-500">No .n3 files found</div>
        )}
      </div>
      
      {/* Node Properties Section */}
      <div className="flex-1 p-4 overflow-y-auto">
        <h2 className="text-lg font-semibold mb-2">Properties</h2>
        
        {selectedNode ? (
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
              <input
                type="text"
                value={selectedNode.data.label}
                readOnly
                className="w-full px-3 py-2 border border-gray-300 rounded text-sm bg-gray-50"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Type</label>
              <input
                type="text"
                value={selectedNode.type}
                readOnly
                className="w-full px-3 py-2 border border-gray-300 rounded text-sm bg-gray-50"
              />
            </div>
            
            {selectedNode.data.description && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <textarea
                  value={selectedNode.data.description}
                  readOnly
                  className="w-full px-3 py-2 border border-gray-300 rounded text-sm bg-gray-50"
                  rows={3}
                />
              </div>
            )}
            
            {/* Display node-specific properties */}
            {Object.entries(selectedNode.data)
              .filter(([key]) => !['label', 'description', 'metadata'].includes(key))
              .map(([key, value]) => (
                <div key={key}>
                  <label className="block text-sm font-medium text-gray-700 mb-1 capitalize">
                    {key.replace(/([A-Z])/g, ' $1').trim()}
                  </label>
                  <input
                    type="text"
                    value={typeof value === 'object' ? JSON.stringify(value) : String(value)}
                    readOnly
                    className="w-full px-3 py-2 border border-gray-300 rounded text-sm bg-gray-50"
                  />
                </div>
              ))}
          </div>
        ) : (
          <div className="text-sm text-gray-500">Select a node to view properties</div>
        )}
      </div>
      
      {/* Graph Stats */}
      <div className="p-4 border-t border-gray-200 bg-gray-50">
        <h3 className="text-sm font-semibold text-gray-700 mb-2">Graph Stats</h3>
        <div className="space-y-1 text-sm text-gray-600">
          <div>Nodes: {nodes.length}</div>
          <div>Selected: {selectedNodeIds.size}</div>
        </div>
      </div>
    </div>
  );
};
