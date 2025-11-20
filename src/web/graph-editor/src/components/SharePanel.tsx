interface SharePanelProps {
  projectId: string;
}

export default function SharePanel({ projectId }: SharePanelProps) {
  return (
    <div className="flex-1 overflow-auto p-4">
      <h2 className="mb-4 text-lg font-semibold">Share</h2>
      <div className="text-sm text-muted-foreground">
        Share panel for {projectId}
      </div>
    </div>
  );
}
