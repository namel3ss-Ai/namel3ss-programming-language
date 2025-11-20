import { useNavigate } from 'react-router-dom';

export default function NotFoundPage() {
  const navigate = useNavigate();

  return (
    <div className="flex h-full flex-col items-center justify-center gap-4">
      <div className="text-6xl font-bold text-muted-foreground">404</div>
      <div className="text-lg text-muted-foreground">Page not found</div>
      <button
        onClick={() => navigate('/')}
        className="rounded-md bg-primary px-4 py-2 text-sm text-primary-foreground hover:bg-primary/90"
      >
        Go to Home
      </button>
    </div>
  );
}
