import { create } from 'zustand';

interface FilterState {
  // Current video being viewed
  videoId: string | null;

  // Filter values
  gameNumber: number | null;
  playerId: 1 | 2 | null;
  startTime: number | null;
  endTime: number | null;

  // Actions
  setVideoId: (id: string) => void;
  setGameNumber: (game: number | null) => void;
  setPlayerId: (player: 1 | 2 | null) => void;
  setTimeRange: (start: number | null, end: number | null) => void;
  clearFilters: () => void;
  resetAll: () => void;
}

/**
 * Global filter store using Zustand
 * Manages persistent filters across all dashboard tabs
 * Filter changes trigger React Query refetch
 */
export const useFilterStore = create<FilterState>((set) => ({
  // Initial state
  videoId: null,
  gameNumber: null,
  playerId: null,
  startTime: null,
  endTime: null,

  // Set current video
  setVideoId: (id) => set({ videoId: id }),

  // Set game filter (null = all games)
  setGameNumber: (game) => set({ gameNumber: game }),

  // Set player filter (null = all players)
  setPlayerId: (player) => set({ playerId: player }),

  // Set time range filter
  setTimeRange: (start, end) =>
    set({ startTime: start, endTime: end }),

  // Clear all filters except videoId
  clearFilters: () =>
    set({
      gameNumber: null,
      playerId: null,
      startTime: null,
      endTime: null,
    }),

  // Reset everything (used when navigating away)
  resetAll: () =>
    set({
      videoId: null,
      gameNumber: null,
      playerId: null,
      startTime: null,
      endTime: null,
    }),
}));
