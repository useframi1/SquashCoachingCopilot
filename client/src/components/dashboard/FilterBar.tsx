'use client';

import { useEffect } from 'react';
import { Filter, X, Menu } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useFilterStore } from '@/lib/stores/filterStore';
import { useQuery } from '@tanstack/react-query';
import { getMatchSummary } from '@/lib/api/analytics';

interface FilterBarProps {
  videoId: string;
  onMenuClick?: () => void;
}

/**
 * Global filter bar for dashboard
 * Filters persist across all tabs and trigger data refetch
 */
export function FilterBar({ videoId, onMenuClick }: FilterBarProps) {
  const {
    gameNumber,
    playerId,
    setVideoId,
    setGameNumber,
    setPlayerId,
    clearFilters,
  } = useFilterStore();

  // Fetch match summary to get available games
  const { data: matchSummary } = useQuery({
    queryKey: ['match-summary', videoId],
    queryFn: () => getMatchSummary(videoId),
    staleTime: 5 * 60 * 1000,
  });

  // Set videoId in store when component mounts
  useEffect(() => {
    setVideoId(videoId);
  }, [videoId, setVideoId]);

  const hasActiveFilters = gameNumber !== null || playerId !== null;

  return (
    <div className="min-h-16 bg-white border-b border-gray-200 px-4 md:px-6 py-3">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        {/* Left: Mobile Menu + Filter Icon and Title */}
        <div className="flex items-center gap-3">
          {onMenuClick && (
            <Button
              variant="ghost"
              size="icon"
              onClick={onMenuClick}
              className="md:hidden"
              aria-label="Open menu"
            >
              <Menu className="h-5 w-5" />
            </Button>
          )}
          <Filter className="h-5 w-5 text-muted-foreground" />
          <span className="text-sm font-medium">Filters</span>
        </div>

        <div className="flex flex-col sm:flex-row sm:items-center gap-3 sm:gap-4">
          {/* Filter Controls */}
          <div className="flex items-center gap-3 sm:gap-4 flex-wrap">
            {/* Game Number Filter */}
            <div className="flex items-center gap-2">
              <label htmlFor="game-filter" className="text-sm text-muted-foreground">
                Game:
              </label>
              <Select
                value={gameNumber?.toString() ?? 'all'}
                onValueChange={(value) =>
                  setGameNumber(value === 'all' ? null : parseInt(value))
                }
              >
                <SelectTrigger className="w-[140px]">
                  <SelectValue placeholder="All Games" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Games</SelectItem>
                  {matchSummary?.games.map((game) => (
                    <SelectItem key={game.game_number} value={game.game_number.toString()}>
                      Game {game.game_number}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Player Filter */}
            <div className="flex items-center gap-2">
              <label htmlFor="player-filter" className="text-sm text-muted-foreground">
                Player:
              </label>
              <Select
                value={playerId?.toString() ?? 'all'}
                onValueChange={(value) =>
                  setPlayerId(value === 'all' ? null : (parseInt(value) as 1 | 2))
                }
              >
                <SelectTrigger className="w-[140px]">
                  <SelectValue placeholder="All Players" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Players</SelectItem>
                  <SelectItem value="1">Player 1</SelectItem>
                  <SelectItem value="2">Player 2</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Clear Filters Button */}
            {hasActiveFilters && (
              <Button
                variant="ghost"
                size="sm"
                onClick={clearFilters}
                className="text-destructive hover:text-destructive"
              >
                <X className="h-4 w-4 mr-1" />
                Clear
              </Button>
            )}
          </div>

          {/* Match Info */}
          <div className="text-sm text-muted-foreground">
            {matchSummary && (
              <span>
                {matchSummary.match.player_1_games_won}-
                {matchSummary.match.player_2_games_won}
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
