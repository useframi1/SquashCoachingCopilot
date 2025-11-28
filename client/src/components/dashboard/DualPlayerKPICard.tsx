import { LucideIcon } from 'lucide-react';
import { cn } from '@/lib/utils/cn';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';

interface DualPlayerKPICardProps {
  title: string;
  player1Value: string | number;
  player2Value: string | number;
  player1Name?: string;
  player2Name?: string;
  icon?: LucideIcon;
  subtitle?: string;
  className?: string;
}

/**
 * KPI card component that displays values for both players side-by-side
 */
export function DualPlayerKPICard({
  title,
  player1Value,
  player2Value,
  player1Name = 'Player 1',
  player2Name = 'Player 2',
  icon: Icon,
  subtitle,
  className,
}: DualPlayerKPICardProps) {
  return (
    <Card
      className={cn('transition-all duration-200 hover:shadow-md', className)}
      role="article"
      aria-label={`${title}: ${player1Name}: ${player1Value}, ${player2Name}: ${player2Value}`}
    >
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
        <CardTitle className="text-base font-semibold">{title}</CardTitle>
        {Icon && (
          <div className="p-3 bg-accent rounded-lg" aria-hidden="true">
            <Icon className="w-6 h-6 text-accent-foreground" />
          </div>
        )}
      </CardHeader>
      <CardContent className="pb-6">
        <div className="grid grid-cols-2 gap-6">
          <div className="text-center">
            <div className="text-sm text-muted-foreground mb-2">{player1Name}</div>
            <div className="text-3xl font-bold">{player1Value}</div>
          </div>
          <div className="text-center">
            <div className="text-sm text-muted-foreground mb-2">{player2Name}</div>
            <div className="text-3xl font-bold">{player2Value}</div>
          </div>
        </div>
        {subtitle && (
          <div className="text-sm text-muted-foreground mt-3 text-center">
            {subtitle}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
