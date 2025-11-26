import { LucideIcon, FileQuestion } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

interface EmptyStateProps {
  icon?: LucideIcon;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  className?: string;
}

/**
 * Empty state component for displaying when no data is available
 * Provides consistent messaging and optional action button
 */
export function EmptyState({
  icon: Icon = FileQuestion,
  title,
  description,
  action,
  className,
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center p-12 text-center',
        className
      )}
    >
      <div className="p-6 bg-gray-50 rounded-full mb-6">
        <Icon className="w-12 h-12 text-gray-400" />
      </div>

      <h3 className="text-xl font-semibold text-gray-900 mb-2">{title}</h3>

      {description && (
        <p className="text-base text-gray-600 max-w-md mb-6">{description}</p>
      )}

      {action && (
        <button
          onClick={action.onClick}
          className="px-6 py-2.5 bg-red-700 text-white font-semibold rounded-lg hover:bg-red-800 transition-colors focus:outline-none focus:ring-2 focus:ring-red-700 focus:ring-offset-2"
        >
          {action.label}
        </button>
      )}
    </div>
  );
}
