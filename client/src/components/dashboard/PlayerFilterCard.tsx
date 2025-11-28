'use client';

import { useState } from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { usePlayerNames } from '@/lib/hooks/usePlayerNames';

interface PlayerFilterCardProps {
  videoId: string;
  title: string;
  children: (playerId: 1 | 2 | null) => React.ReactNode;
  defaultPlayer?: 1 | 2 | null;
}

/**
 * Card wrapper that provides player filtering at the card level
 * Renders a card with a title and player selector, passing the selected player to children
 */
export function PlayerFilterCard({
  videoId,
  title,
  children,
  defaultPlayer = null
}: PlayerFilterCardProps) {
  const [selectedPlayer, setSelectedPlayer] = useState<1 | 2 | null>(defaultPlayer);
  const { player1Name, player2Name } = usePlayerNames(videoId);

  return (
    <div className="bg-white p-6 rounded-lg border border-gray-200">
      {/* Card Header with Title and Player Filter */}
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-md font-semibold text-gray-900">{title}</h4>

        <div className="flex items-center gap-2">
          <label htmlFor="player-select" className="text-sm text-muted-foreground">
            Player:
          </label>
          <Select
            value={selectedPlayer?.toString() ?? 'all'}
            onValueChange={(value) =>
              setSelectedPlayer(value === 'all' ? null : (parseInt(value) as 1 | 2))
            }
          >
            <SelectTrigger className="w-[140px]">
              <SelectValue placeholder="All Players" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Players</SelectItem>
              <SelectItem value="1">{player1Name}</SelectItem>
              <SelectItem value="2">{player2Name}</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Card Content */}
      <div>
        {children(selectedPlayer)}
      </div>
    </div>
  );
}
