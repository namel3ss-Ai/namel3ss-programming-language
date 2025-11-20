import { Routes, Route, Navigate } from 'react-router-dom';
import GraphEditorPage from './pages/GraphEditorPage';
import ShareOpenPage from './pages/ShareOpenPage';
import NotFoundPage from './pages/NotFoundPage';

function App() {
  return (
    <div className="h-screen w-screen overflow-hidden bg-background text-foreground">
      <Routes>
        <Route path="/" element={<Navigate to="/project/demo" replace />} />
        <Route path="/project/:projectId" element={<GraphEditorPage />} />
        <Route path="/open/:token" element={<ShareOpenPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </div>
  );
}

export default App;
