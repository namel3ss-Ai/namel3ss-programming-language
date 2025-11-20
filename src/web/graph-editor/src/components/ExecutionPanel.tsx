interface ExecutionPanelProps {
  projectId: string;
}

export default function ExecutionPanel({ projectId }: ExecutionPanelProps) {
  return (
    <div className="flex-1 overflow-auto border-b border-border p-4">
      <h2 className="mb-4 text-lg font-semibold">Execution</h2>
      <div className="text-sm text-muted-foreground">
        Execution panel for {projectId}
      </div>
    </div>
  );
}
