import { cn } from '@/lib/utils';

export interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function LoadingSpinner({ size = 'md', className }: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'h-4 w-4 border-2',
    md: 'h-8 w-8 border-3',
    lg: 'h-12 w-12 border-4',
  };

  return (
    <div
      className={cn(
        'animate-spin rounded-full border-t-[var(--primary)] border-r-transparent border-b-transparent border-l-transparent',
        sizeClasses[size],
        className
      )}
    />
  );
}

export function LoadingScreen({ message }: { message?: string }) {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-4">
      <LoadingSpinner size="lg" />
      {message && (
        <p className="text-lg text-[var(--foreground-secondary)]">{message}</p>
      )}
    </div>
  );
}
