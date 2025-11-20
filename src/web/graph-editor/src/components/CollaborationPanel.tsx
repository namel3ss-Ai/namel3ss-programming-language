import { UserPresence } from '@/types/graph';

interface CollaborationPanelProps {
  users: UserPresence[];
  isConnected: boolean;
}

export default function CollaborationPanel({ users, isConnected }: CollaborationPanelProps) {
  return (
    <div className="rounded-md border border-border bg-background p-2 shadow-md">
      <div className="flex items-center gap-2">
        <div
          className={`h-2 w-2 rounded-full ${
            isConnected ? 'bg-green-500' : 'bg-red-500'
          }`}
        />
        <div className="text-xs font-medium">
          {users.length} {users.length === 1 ? 'user' : 'users'} online
        </div>
      </div>
      {users.length > 0 && (
        <div className="mt-2 flex gap-1">
          {users.map((user) => (
            <div
              key={user.userId}
              className="h-6 w-6 rounded-full border-2 border-background"
              style={{ backgroundColor: user.color }}
              title={user.displayName}
            />
          ))}
        </div>
      )}
    </div>
  );
}
