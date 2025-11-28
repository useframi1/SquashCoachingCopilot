'use client';

import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Trophy } from 'lucide-react';

interface ScoreCardProps {
  player1Name: string;
  player2Name: string;
  player1Score: number;
  player2Score: number;
  winner?: 1 | 2 | null;
  label?: string;
  subtitle?: string;
  variant?: 'default' | 'large';
}

/**
 * Score card displaying match or game scores in a visually appealing format
 */
export function ScoreCard({
  player1Name,
  player2Name,
  player1Score,
  player2Score,
  winner,
  label,
  subtitle,
  variant = 'default'
}: ScoreCardProps) {
  const isLarge = variant === 'large';

  return (
    <Card className={isLarge ? 'shadow-lg' : ''}>
      <CardContent className={isLarge ? 'pt-8 pb-8' : 'pt-6'}>
        <div className="space-y-4">
          {/* Label */}
          {label && (
            <div className="text-center">
              <Badge variant="secondary" className="text-sm">
                {label}
              </Badge>
            </div>
          )}

          {/* Score Display */}
          <div className="flex items-center justify-between gap-4">
            {/* Player 1 */}
            <div className={`flex-1 text-right ${isLarge ? 'space-y-2' : 'space-y-1'}`}>
              <div className={`font-semibold ${isLarge ? 'text-lg' : 'text-sm'} truncate ${winner === 1 ? 'text-primary' : 'text-foreground'}`}>
                {player1Name}
              </div>
              <div className={`font-bold ${isLarge ? 'text-4xl' : 'text-3xl'} ${winner === 1 ? 'text-primary' : 'text-muted-foreground'}`}>
                {player1Score}
              </div>
            </div>

            {/* Separator with Winner Icon */}
            <div className="flex flex-col items-center gap-2">
              {winner && isLarge && (
                <Trophy className={`h-5 w-5 ${winner === 1 ? '-ml-8' : 'ml-8'} text-primary`} />
              )}
              <div className={`${isLarge ? 'text-2xl' : 'text-xl'} text-muted-foreground font-light`}>
                -
              </div>
            </div>

            {/* Player 2 */}
            <div className={`flex-1 text-left ${isLarge ? 'space-y-2' : 'space-y-1'}`}>
              <div className={`font-semibold ${isLarge ? 'text-lg' : 'text-sm'} truncate ${winner === 2 ? 'text-primary' : 'text-foreground'}`}>
                {player2Name}
              </div>
              <div className={`font-bold ${isLarge ? 'text-4xl' : 'text-3xl'} ${winner === 2 ? 'text-primary' : 'text-muted-foreground'}`}>
                {player2Score}
              </div>
            </div>
          </div>

          {/* Subtitle */}
          {subtitle && (
            <div className="text-center">
              <p className="text-xs text-muted-foreground">{subtitle}</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
