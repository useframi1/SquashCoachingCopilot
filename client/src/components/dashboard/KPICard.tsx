import { LucideIcon } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

interface KPICardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: LucideIcon;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: string;
  className?: string;
}

/**
 * Reusable KPI card component
 * Displays a metric with optional icon, subtitle, and trend
 */
export function KPICard({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  trendValue,
  className,
}: KPICardProps) {
  return (
    <div
      className={cn(
        'bg-white p-6 rounded-lg border border-gray-200 hover:shadow-md transition-all duration-200',
        className
      )}
      role="article"
      aria-label={`${title}: ${value}`}
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-600">{title}</h3>
        {Icon && (
          <div className="p-2 bg-red-50 rounded-lg" aria-hidden="true">
            <Icon className="w-5 h-5 text-red-700" />
          </div>
        )}
      </div>

      <div className="space-y-1">
        <p className="text-3xl font-bold text-gray-900">{value}</p>

        {subtitle && <p className="text-sm text-gray-500">{subtitle}</p>}

        {trend && trendValue && (
          <div
            className={cn(
              'text-sm font-medium',
              trend === 'up' && 'text-green-600',
              trend === 'down' && 'text-red-600',
              trend === 'neutral' && 'text-gray-600'
            )}
            aria-label={`Trend: ${trendValue}`}
          >
            {trendValue}
          </div>
        )}
      </div>
    </div>
  );
}
