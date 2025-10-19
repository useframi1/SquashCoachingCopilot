'use client';

import { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import { Button } from '../ui/Button';
import type { RallySummary } from '@/lib/types';
import { formatDuration, formatTimestamp } from '@/lib/utils';

interface RallyCardProps {
  rally: RallySummary;
  onPlayVideo?: () => void;
}

export function RallyCard({ rally, onPlayVideo }: RallyCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <Card className="bg-card-bg border-border hover:border-primary transition-colors">
      <CardHeader className="cursor-pointer" onClick={() => setExpanded(!expanded)}>
        <div className="flex justify-between items-start">
          <div>
            <CardTitle className="text-foreground text-lg">
              Rally #{rally.rally_number}
            </CardTitle>
            <p className="text-sm text-foreground-secondary mt-1">
              {formatTimestamp(rally.start_time)} - {formatTimestamp(rally.end_time)}
            </p>
          </div>
          <div className="text-right">
            <div className="text-xl font-bold text-primary">
              {formatDuration(rally.duration)}
            </div>
            <p className="text-xs text-foreground-secondary">
              {rally.total_strokes} strokes
            </p>
          </div>
        </div>
      </CardHeader>

      {expanded && (
        <CardContent className="pt-0 space-y-4">
          <div className="border-t border-border pt-4">
            {/* Player 1 Stats */}
            <div className="mb-3">
              <h4 className="text-sm font-semibold text-foreground mb-2">Player 1</h4>
              <div className="grid grid-cols-3 gap-2 text-sm">
                <div>
                  <span className="text-foreground-secondary">Total:</span>
                  <span className="ml-2 text-foreground font-medium">
                    {rally.player1_strokes.total}
                  </span>
                </div>
                <div>
                  <span className="text-foreground-secondary">Forehand:</span>
                  <span className="ml-2 text-foreground font-medium">
                    {rally.player1_strokes.forehand}
                  </span>
                </div>
                <div>
                  <span className="text-foreground-secondary">Backhand:</span>
                  <span className="ml-2 text-foreground font-medium">
                    {rally.player1_strokes.backhand}
                  </span>
                </div>
              </div>
            </div>

            {/* Player 2 Stats */}
            <div>
              <h4 className="text-sm font-semibold text-foreground mb-2">Player 2</h4>
              <div className="grid grid-cols-3 gap-2 text-sm">
                <div>
                  <span className="text-foreground-secondary">Total:</span>
                  <span className="ml-2 text-foreground font-medium">
                    {rally.player2_strokes.total}
                  </span>
                </div>
                <div>
                  <span className="text-foreground-secondary">Forehand:</span>
                  <span className="ml-2 text-foreground font-medium">
                    {rally.player2_strokes.forehand}
                  </span>
                </div>
                <div>
                  <span className="text-foreground-secondary">Backhand:</span>
                  <span className="ml-2 text-foreground font-medium">
                    {rally.player2_strokes.backhand}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {onPlayVideo && (
            <Button
              onClick={onPlayVideo}
              className="w-full bg-primary hover:bg-primary-hover"
            >
              Play Rally Video
            </Button>
          )}
        </CardContent>
      )}
    </Card>
  );
}
