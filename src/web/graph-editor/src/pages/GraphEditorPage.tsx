import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { graphApi } from '@/lib/api';
import GraphCanvas from '@/components/GraphCanvas';
import Toolbar from '@/components/Toolbar';
import ExecutionPanel from '@/components/ExecutionPanel';
import SharePanel from '@/components/SharePanel';

export default function GraphEditorPage() {
  const { projectId } = useParams<{ projectId: string }>();

  const { data: graphData, isLoading, error } = useQuery({
    queryKey: ['graph', projectId],
    queryFn: () => graphApi.getGraph(projectId!),
    enabled: !!projectId,
  });

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-lg text-muted-foreground">Loading graph...</div>
      </div>
    );
  }

  if (error || !graphData) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-lg text-destructive">
          Failed to load graph: {error?.message || 'Unknown error'}
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <Toolbar projectId={projectId!} projectName={graphData.name} />
      <div className="flex flex-1 overflow-hidden">
        <GraphCanvas projectId={projectId!} initialGraph={graphData} />
        <div className="w-96 border-l border-border flex flex-col">
          <ExecutionPanel projectId={projectId!} />
          <SharePanel projectId={projectId!} />
        </div>
      </div>
    </div>
  );
}
