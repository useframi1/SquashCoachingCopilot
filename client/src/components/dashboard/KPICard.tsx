import { LucideIcon } from 'lucide-react';
import { cn } from '@/lib/utils/cn';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { InfoTooltip } from '@/components/ui/InfoTooltip';

interface KPICardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: LucideIcon;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: string;
  className?: string;
  infoTooltip?: string;
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
  infoTooltip,
}: KPICardProps) {
  return (
    <Card
      className={cn('transition-all duration-200 hover:shadow-md', className)}
      role="article"
      aria-label={`${title}: ${value}`}
    >
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
        <div className="flex items-center gap-2">
          <CardTitle className="text-base font-semibold">{title}</CardTitle>
          {infoTooltip && <InfoTooltip content={infoTooltip} />}
        </div>
        {Icon && (
          <div className="p-3 bg-accent rounded-lg" aria-hidden="true">
            <Icon className="w-6 h-6 text-accent-foreground" />
          </div>
        )}
      </CardHeader>
      <CardContent className="pb-6">
        <div className="text-3xl font-bold">{value}</div>
        {subtitle && (
          <CardDescription className="mt-2 text-sm">{subtitle}</CardDescription>
        )}
        {trend && trendValue && (
          <p
            className={cn(
              'text-sm font-medium mt-3',
              trend === 'up' && 'text-green-600',
              trend === 'down' && 'text-red-600',
              trend === 'neutral' && 'text-muted-foreground'
            )}
            aria-label={`Trend: ${trendValue}`}
          >
            {trendValue}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
