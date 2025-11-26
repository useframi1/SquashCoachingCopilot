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
  getRallyTimeline,
  getMovementMetrics,
  getTZoneOccupancy,
  getPlayerPositionHeatmap,
  getCourtQuadrants,
  getTimeToTTimeline,
  getShotEffectiveness,
  getWinningStats,
  getRallyIntensity,
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
