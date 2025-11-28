'use client';

import { useEffect, useState } from 'react';
import { Filter, X, Menu, UserCog } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { PlayerNamingModal } from '@/components/dashboard/PlayerNamingModal';
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
import { getVideoMetadata } from '@/lib/api/videos';

interface FilterBarProps {
  videoId: string;
  onMenuClick?: () => void;
}

/**
 * Global filter bar for dashboard
 * Filters persist across all tabs and trigger data refetch
 */
export function FilterBar({ videoId, onMenuClick }: FilterBarProps) {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const {
    gameNumber,
    setVideoId,
    setGameNumber,
    clearFilters,
  } = useFilterStore();

  // Fetch match summary to get available games
  const { data: matchSummary } = useQuery({
    queryKey: ['match-summary', videoId],
    queryFn: () => getMatchSummary(videoId),
    staleTime: 5 * 60 * 1000,
  });

  // Fetch video metadata for player names
  const { data: videoMetadata } = useQuery({
    queryKey: ['video', videoId],
    queryFn: () => getVideoMetadata(videoId),
    staleTime: 5 * 60 * 1000,
  });

  // Set videoId in store when component mounts
  useEffect(() => {
    setVideoId(videoId);
  }, [videoId, setVideoId]);

  const hasActiveFilters = gameNumber !== null;

  return (
    <>
      <div className="min-h-16 bg-card border-b border-border/20 shadow-sm px-4 md:px-6 py-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          {/* Left: Filter Icon, Title, and Filter Controls */}
          <div className="flex flex-col sm:flex-row sm:items-center gap-3 sm:gap-4">
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

            {/* Clear Filters Button */}
            {hasActiveFilters && (
              <Button
                variant="ghost"
                size="sm"
                onClick={clearFilters}
                className="text-destructive hover:text-destructive hover:bg-destructive/10"
              >
                <X className="h-4 w-4 mr-1" />
                Clear
              </Button>
            )}
          </div>
        </div>

        {/* Right: Edit Player Names Button */}
        <div className="flex items-center">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsModalOpen(true)}
            className="gap-2"
          >
            <UserCog className="h-4 w-4" />
            Edit Player Names
          </Button>
        </div>
      </div>
      </div>

      {/* Player Naming Modal */}
      <PlayerNamingModal
        videoId={videoId}
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        currentPlayer1Name={videoMetadata?.player_1_name}
        currentPlayer2Name={videoMetadata?.player_2_name}
      />
    </>
  );
}
