import { LucideIcon } from 'lucide-react';
import { cn } from '@/lib/utils/cn';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';

interface ComparisonStatCardProps {
  title: string;
  player1Label: string;
  player2Label: string;
  player1Value: string | number;
  player2Value: string | number;
  player1Name?: string;
  player2Name?: string;
  icon?: LucideIcon;
  className?: string;
  highlightBetter?: boolean; // If true, highlight the better (higher) value
  highlightLower?: boolean; // If true, highlight the better (lower) value
}

/**
 * Compact comparison card showing a single stat for both players
 * with optional highlighting of the better performer
 */
export function ComparisonStatCard({
  title,
  player1Label,
  player2Label,
  player1Value,
  player2Value,
  player1Name = 'Player 1',
  player2Name = 'Player 2',
  icon: Icon,
  className,
  highlightBetter = false,
  highlightLower = false,
}: ComparisonStatCardProps) {
  // Determine which player has better value for highlighting
  const p1Num = typeof player1Value === 'number' ? player1Value : parseFloat(String(player1Value).replace(/[^0-9.-]/g, ''));
  const p2Num = typeof player2Value === 'number' ? player2Value : parseFloat(String(player2Value).replace(/[^0-9.-]/g, ''));

  let player1Better = false;
  let player2Better = false;

  if (!isNaN(p1Num) && !isNaN(p2Num)) {
    if (highlightBetter) {
      // Higher is better
      player1Better = p1Num > p2Num;
      player2Better = p2Num > p1Num;
    } else if (highlightLower) {
      // Lower is better
      player1Better = p1Num < p2Num;
      player2Better = p2Num < p1Num;
    }
  }

  return (
    <Card className={cn('transition-all duration-200 hover:shadow-md', className)}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {Icon && (
          <div className="p-2 bg-accent rounded-lg" aria-hidden="true">
            <Icon className="w-4 h-4 text-accent-foreground" />
          </div>
        )}
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {/* Player 1 */}
          <div className={cn(
            "flex items-center justify-between p-2 rounded-md transition-colors",
            player1Better && "bg-green-50 border border-green-200"
          )}>
            <div>
              <p className="text-xs text-muted-foreground">{player1Name}</p>
              <p className="text-xs text-gray-600">{player1Label}</p>
            </div>
            <div className={cn(
              "text-xl font-bold",
              player1Better && "text-green-700"
            )}>
              {player1Value}
            </div>
          </div>

          {/* Player 2 */}
          <div className={cn(
            "flex items-center justify-between p-2 rounded-md transition-colors",
            player2Better && "bg-green-50 border border-green-200"
          )}>
            <div>
              <p className="text-xs text-muted-foreground">{player2Name}</p>
              <p className="text-xs text-gray-600">{player2Label}</p>
            </div>
            <div className={cn(
              "text-xl font-bold",
              player2Better && "text-green-700"
            )}>
              {player2Value}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
