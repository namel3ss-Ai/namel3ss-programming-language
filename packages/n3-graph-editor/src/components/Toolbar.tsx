/**
 * Toolbar Component
 * 
 * Provides controls for:
 * - Auto-layout
 * - Zoom/Fit
 * - Save
 * - Search/Filter
 */

import React from 'react';
import { useReactFlow } from 'reactflow';
import { useGraphStore } from '../store/graphStore';

export const Toolbar: React.FC = () => {
  const { fitView, zoomIn, zoomOut } = useReactFlow();
  const { autoLayout, currentFile } = useGraphStore();
  
  const handleAutoLayout = () => {
    autoLayout();
  };
  
  const handleFitView = () => {
    fitView({ padding: 0.2 });
  };
  
  const handleZoomIn = () => {
    zoomIn();
  };
  
  const handleZoomOut = () => {
    zoomOut();
  };
  
  const handleSave = async () => {
    if (!currentFile) {
      alert('No file selected');
      return;
    }
    
    // TODO: Implement save functionality
    alert('Save functionality not yet implemented');
  };
  
  return (
    <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-10">
      <div className="bg-white rounded-lg shadow-lg border border-gray-200 px-4 py-2 flex items-center gap-2">
        {/* File Name */}
        {currentFile && (
          <span className="text-sm font-medium text-gray-700 mr-2">{currentFile}</span>
        )}
        
        {/* Divider */}
        {currentFile && <div className="h-6 w-px bg-gray-300"></div>}
        
        {/* Auto Layout */}
        <button
          onClick={handleAutoLayout}
          className="px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-100 rounded transition"
          title="Auto Layout (Ctrl+L)"
        >
          🔄 Layout
        </button>
        
        {/* Fit View */}
        <button
          onClick={handleFitView}
          className="px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-100 rounded transition"
          title="Fit View (Ctrl+F)"
        >
          🔍 Fit
        </button>
        
        {/* Zoom In */}
        <button
          onClick={handleZoomIn}
          className="px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-100 rounded transition"
          title="Zoom In (+)"
        >
          ➕
        </button>
        
        {/* Zoom Out */}
        <button
          onClick={handleZoomOut}
          className="px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-100 rounded transition"
          title="Zoom Out (-)"
        >
          ➖
        </button>
        
        {/* Divider */}
        <div className="h-6 w-px bg-gray-300"></div>
        
        {/* Save */}
        <button
          onClick={handleSave}
          className="px-3 py-1.5 text-sm font-medium text-white bg-blue-500 hover:bg-blue-600 rounded transition"
          title="Save (Ctrl+S)"
        >
          💾 Save
        </button>
      </div>
    </div>
  );
};
