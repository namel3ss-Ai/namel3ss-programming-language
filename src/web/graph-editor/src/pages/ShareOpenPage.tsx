import { useParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { shareApi } from '@/lib/api';

export default function ShareOpenPage() {
  const { token } = useParams<{ token: string }>();
  const navigate = useNavigate();

  const { data, isLoading, error } = useQuery({
    queryKey: ['share-token', token],
    queryFn: () => shareApi.validateToken(token!),
    enabled: !!token,
  });

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-lg text-muted-foreground">Validating share link...</div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4">
        <div className="text-lg text-destructive">Invalid or expired share link</div>
        <button
          onClick={() => navigate('/')}
          className="rounded-md bg-primary px-4 py-2 text-sm text-primary-foreground hover:bg-primary/90"
        >
          Go to Home
        </button>
      </div>
    );
  }

  // Redirect to project with role stored in session
  sessionStorage.setItem(`project-${data.projectId}-role`, data.role);
  navigate(`/project/${data.projectId}`);

  return null;
}
