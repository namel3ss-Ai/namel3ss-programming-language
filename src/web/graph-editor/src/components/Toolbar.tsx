interface ToolbarProps {
  projectId: string;
  projectName: string;
}

export default function Toolbar({ projectId, projectName }: ToolbarProps) {
  return (
    <div className="flex h-14 items-center justify-between border-b border-border bg-background px-4">
      <div className="flex items-center gap-4">
        <h1 className="text-xl font-bold">{projectName}</h1>
        <div className="text-xs text-muted-foreground">ID: {projectId}</div>
      </div>
      <div className="flex items-center gap-2">
        <button className="rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground hover:bg-primary/90">
          Save
        </button>
      </div>
    </div>
  );
}
