import { useQuery } from '@tanstack/react-query';
import { useFilterStore } from '@/lib/stores/filterStore';
import type { AnalyticsFilters } from '@/types/api';
import {
  getMatchSummary,
  getMomentumTimeline,
  getStrokeDistribution,
  getShotTypeDistribution,
  getBallSpeed,
  getRhythmDisruption,
  getWallQuadrants,
  getWallHitsHeatmap,
  getRallyTimeline,
  getMovementMetrics,
  getTZoneOccupancy,
  getPlayerPositionHeatmap,
  getCourtQuadrants,
  getTimeToTTimeline,
  getShotEffectiveness,
  getWinningStats,
  getRallyIntensity,
  getLongestRally,
  getFastestShot,
  getLetStats,
  getBreakTime,
  getMovementMetricsPerRally,
  getTZoneOccupancyPerRally,
  getWinningEfficiencyPerRally,
} from '@/lib/api/analytics';

/**
 * Build filters object from store
 */
function useFilters(): AnalyticsFilters {
  const { gameNumber, playerId, startTime, endTime } = useFilterStore();

  return {
    game_number: gameNumber ?? undefined,
    player_id: playerId ?? undefined,
    start_time: startTime ?? undefined,
    end_time: endTime ?? undefined,
  };
}

/**
 * Build filters with optional player ID override (for card-level filtering)
 */
function useFiltersWithPlayer(playerIdOverride?: 1 | 2 | null): AnalyticsFilters {
  const { gameNumber, startTime, endTime } = useFilterStore();

  return {
    game_number: gameNumber ?? undefined,
    player_id: playerIdOverride ?? undefined,
    start_time: startTime ?? undefined,
    end_time: endTime ?? undefined,
  };
}

/**
 * Hook for match summary (no filters)
 */
export function useMatchSummary(videoId: string) {
  return useQuery({
    queryKey: ['match-summary', videoId],
    queryFn: () => getMatchSummary(videoId),
    enabled: !!videoId,
  });
}

/**
 * Hook for momentum timeline
 */
export function useMomentumTimeline(videoId: string) {
  const filters = useFilters();

  return useQuery({
    queryKey: ['momentum-timeline', videoId, filters],
    queryFn: () => getMomentumTimeline(videoId, filters),
    enabled: !!videoId,
  });
}

/**
 * Hook for stroke distribution
 */
export function useStrokeDistribution(videoId: string) {
  const filters = useFilters();

  return useQuery({
    queryKey: ['stroke-distribution', videoId, filters],
    queryFn: () => getStrokeDistribution(videoId, filters),
    enabled: !!videoId,
  });
}

/**
 * Hook for shot type distribution
 */
export function useShotTypeDistribution(videoId: string) {
  const filters = useFilters();

  return useQuery({
    queryKey: ['shot-type-distribution', videoId, filters],
    queryFn: () => getShotTypeDistribution(videoId, filters),
    enabled: !!videoId,
  });
}

/**
 * Hook for ball speed
 */
export function useBallSpeed(videoId: string) {
  const filters = useFilters();

  return useQuery({
    queryKey: ['ball-speed', videoId, filters],
    queryFn: () => getBallSpeed(videoId, filters),
    enabled: !!videoId,
  });
}

/**
 * Hook for rhythm disruption
 */
export function useRhythmDisruption(videoId: string) {
  const filters = useFilters();

  return useQuery({
    queryKey: ['rhythm-disruption', videoId, filters],
    queryFn: () => getRhythmDisruption(videoId, filters),
    enabled: !!videoId,
  });
}

/**
 * Hook for wall quadrants
 */
export function useWallQuadrants(videoId: string) {
  const filters = useFilters();

  return useQuery({
    queryKey: ['wall-quadrants', videoId, filters],
    queryFn: () => getWallQuadrants(videoId, filters),
    enabled: !!videoId,
  });
}

/**
 * Hook for rally timeline
 */
export function useRallyTimeline(videoId: string) {
  const filters = useFilters();

  return useQuery({
    queryKey: ['rally-timeline', videoId, filters],
    queryFn: () => getRallyTimeline(videoId, filters),
    enabled: !!videoId,
  });
}

/**
 * Hook for movement metrics
 */
export function useMovementMetrics(videoId: string) {
  const filters = useFilters();

  return useQuery({
    queryKey: ['movement-metrics', videoId, filters],
    queryFn: () => getMovementMetrics(videoId, filters),
    enabled: !!videoId,
  });
}

/**
 * Hook for T-zone occupancy
 */
export function useTZoneOccupancy(videoId: string) {
  const filters = useFilters();

  return useQuery({
    queryKey: ['t-zone-occupancy', videoId, filters],
    queryFn: () => getTZoneOccupancy(videoId, filters),
    enabled: !!videoId,
  });
}

/**
 * Hook for player position heatmap
 */
export function usePlayerPositionHeatmap(videoId: string) {
  const filters = useFilters();

  return useQuery({
    queryKey: ['player-position-heatmap', videoId, filters],
    queryFn: () => getPlayerPositionHeatmap(videoId, filters),
    enabled: !!videoId,
  });
}

/**
 * Hook for court quadrants
 */
export function useCourtQuadrants(videoId: string) {
  const filters = useFilters();

  return useQuery({
    queryKey: ['court-quadrants', videoId, filters],
    queryFn: () => getCourtQuadrants(videoId, filters),
    enabled: !!videoId,
  });
}

/**
 * Hook for time-to-T timeline
 */
export function useTimeToTTimeline(videoId: string) {
  const filters = useFilters();

  return useQuery({
    queryKey: ['time-to-t-timeline', videoId, filters],
    queryFn: () => getTimeToTTimeline(videoId, filters),
    enabled: !!videoId,
  });
}

/**
 * Hook for shot effectiveness (requires player ID)
 */
export function useShotEffectiveness(videoId: string, playerId: 1 | 2) {
  const filters = useFilters();

  return useQuery({
    queryKey: ['shot-effectiveness', videoId, playerId, filters],
    queryFn: () => getShotEffectiveness(videoId, playerId, filters),
    enabled: !!videoId && !!playerId,
  });
}

/**
 * Hook for winning stats (requires player ID)
 */
export function useWinningStats(videoId: string, playerId: 1 | 2) {
  const filters = useFilters();

  return useQuery({
    queryKey: ['winning-stats', videoId, playerId, filters],
    queryFn: () => getWinningStats(videoId, playerId, filters),
    enabled: !!videoId && !!playerId,
  });
}

/**
 * Hook for rally intensity
 */
export function useRallyIntensity(videoId: string) {
  const filters = useFilters();

  return useQuery({
    queryKey: ['rally-intensity', videoId, filters],
    queryFn: () => getRallyIntensity(videoId, filters),
    enabled: !!videoId,
  });
}

// ============================================================================
// MATCH HIGHLIGHTS HOOKS
// ============================================================================

/**
 * Hook for longest rally
 */
export function useLongestRally(videoId: string) {
  const filters = useFilters();

  return useQuery({
    queryKey: ['longest-rally', videoId, filters],
    queryFn: () => getLongestRally(videoId, filters),
    enabled: !!videoId,
  });
}

/**
 * Hook for fastest shot
 */
export function useFastestShot(videoId: string) {
  const filters = useFilters();

  return useQuery({
    queryKey: ['fastest-shot', videoId, filters],
    queryFn: () => getFastestShot(videoId, filters),
    enabled: !!videoId,
  });
}

/**
 * Hook for let statistics
 */
export function useLetStats(videoId: string) {
  const filters = useFilters();

  return useQuery({
    queryKey: ['let-stats', videoId, filters],
    queryFn: () => getLetStats(videoId, filters),
    enabled: !!videoId,
  });
}

/**
 * Hook for break time statistics
 */
export function useBreakTime(videoId: string) {
  const filters = useFilters();

  return useQuery({
    queryKey: ['break-time', videoId, filters],
    queryFn: () => getBreakTime(videoId, filters),
    enabled: !!videoId,
  });
}

/**
 * Hook for wall hits heatmap
 */
export function useWallHitsHeatmap(videoId: string) {
  const filters = useFilters();

  return useQuery({
    queryKey: ['wall-hits-heatmap', videoId, filters],
    queryFn: () => getWallHitsHeatmap(videoId, filters),
    enabled: !!videoId,
  });
}

/**
 * Hook for movement metrics per rally
 * Excludes player_id filter since charts show both players
 */
export function useMovementMetricsPerRally(videoId: string) {
  const filters = useFilters();

  // Exclude player_id since charts show both players
  const { player_id, ...chartFilters } = filters;

  return useQuery({
    queryKey: ['movement-metrics-per-rally', videoId, chartFilters],
    queryFn: () => getMovementMetricsPerRally(videoId, chartFilters),
    enabled: !!videoId,
  });
}

/**
 * Hook for T-zone occupancy per rally
 * Excludes player_id filter since charts show both players
 */
export function useTZoneOccupancyPerRally(videoId: string) {
  const filters = useFilters();

  // Exclude player_id since charts show both players
  const { player_id, ...chartFilters } = filters;

  return useQuery({
    queryKey: ['t-zone-occupancy-per-rally', videoId, chartFilters],
    queryFn: () => getTZoneOccupancyPerRally(videoId, chartFilters),
    enabled: !!videoId,
  });
}

/**
 * Hook for winning efficiency per rally
 * Excludes player_id filter since charts show both players
 */
export function useWinningEfficiencyPerRally(videoId: string) {
  const filters = useFilters();

  // Exclude player_id since charts show both players
  const { player_id, ...chartFilters } = filters;

  return useQuery({
    queryKey: ['winning-efficiency-per-rally', videoId, chartFilters],
    queryFn: () => getWinningEfficiencyPerRally(videoId, chartFilters),
    enabled: !!videoId,
  });
}

/**
 * Hook for movement metrics for a specific player (ignores global filter)
 */
export function useMovementMetricsForPlayer(videoId: string, playerId: 1 | 2) {
  const filters = useFilters();

  // Override player_id filter for this specific call
  const playerFilters = { ...filters, player_id: playerId };

  return useQuery({
    queryKey: ['movement-metrics', videoId, playerFilters],
    queryFn: () => getMovementMetrics(videoId, playerFilters),
    enabled: !!videoId,
  });
}

/**
 * Hook for T-zone occupancy for a specific player (ignores global filter)
 */
export function useTZoneOccupancyForPlayer(videoId: string, playerId: 1 | 2) {
  const filters = useFilters();

  // Override player_id filter for this specific call
  const playerFilters = { ...filters, player_id: playerId };

  return useQuery({
    queryKey: ['t-zone-occupancy', videoId, playerFilters],
    queryFn: () => getTZoneOccupancy(videoId, playerFilters),
    enabled: !!videoId,
  });
}

/**
 * Hook for player position heatmap for a specific player (ignores global filter)
 */
export function usePlayerPositionHeatmapForPlayer(videoId: string, playerId: 1 | 2) {
  const filters = useFilters();

  // Override player_id filter for this specific call
  const playerFilters = { ...filters, player_id: playerId };

  return useQuery({
    queryKey: ['player-position-heatmap', videoId, playerFilters],
    queryFn: () => getPlayerPositionHeatmap(videoId, playerFilters),
    enabled: !!videoId,
  });
}

/**
 * Hook for court quadrants for a specific player (ignores global filter)
 */
export function useCourtQuadrantsForPlayer(videoId: string, playerId: 1 | 2) {
  const filters = useFilters();

  // Override player_id filter for this specific call
  const playerFilters = { ...filters, player_id: playerId };

  return useQuery({
    queryKey: ['court-quadrants', videoId, playerFilters],
    queryFn: () => getCourtQuadrants(videoId, playerFilters),
    enabled: !!videoId,
  });
}

// ============================================================================
// CARD-LEVEL PLAYER FILTERING HOOKS
// These hooks accept an optional playerId parameter for card-level filtering
// ============================================================================

/**
 * Hook for wall hits heatmap with optional player filter override
 */
export function useWallHitsHeatmapWithPlayer(videoId: string, playerId?: 1 | 2 | null) {
  const filters = useFiltersWithPlayer(playerId);

  return useQuery({
    queryKey: ['wall-hits-heatmap', videoId, filters],
    queryFn: () => getWallHitsHeatmap(videoId, filters),
    enabled: !!videoId,
  });
}

/**
 * Hook for stroke distribution with optional player filter override
 */
export function useStrokeDistributionWithPlayer(videoId: string, playerId?: 1 | 2 | null) {
  const filters = useFiltersWithPlayer(playerId);

  return useQuery({
    queryKey: ['stroke-distribution', videoId, filters],
    queryFn: () => getStrokeDistribution(videoId, filters),
    enabled: !!videoId,
  });
}

/**
 * Hook for shot type distribution with optional player filter override
 */
export function useShotTypeDistributionWithPlayer(videoId: string, playerId?: 1 | 2 | null) {
  const filters = useFiltersWithPlayer(playerId);

  return useQuery({
    queryKey: ['shot-type-distribution', videoId, filters],
    queryFn: () => getShotTypeDistribution(videoId, filters),
    enabled: !!videoId,
  });
}

/**
 * Hook for wall quadrants with optional player filter override
 */
export function useWallQuadrantsWithPlayer(videoId: string, playerId?: 1 | 2 | null) {
  const filters = useFiltersWithPlayer(playerId);

  return useQuery({
    queryKey: ['wall-quadrants', videoId, filters],
    queryFn: () => getWallQuadrants(videoId, filters),
    enabled: !!videoId,
  });
}
