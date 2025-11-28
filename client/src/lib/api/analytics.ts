import { apiClient } from './client';
import type {
  AnalyticsFilters,
  MatchSummaryResponse,
  StrokeDistributionResponse,
  ShotTypeDistributionResponse,
  BallSpeedResponse,
  BallSpeedPerRallyResponse,
  RhythmDisruptionResponse,
  PlayerPositionHeatmapResponse,
  WallHitHeatmapResponse,
  CourtQuadrantResponse,
  WallQuadrantResponse,
  MovementMetricsResponse,
  TZoneOccupancyResponse,
  TZoneOccupancyPerRallyResponse,
  ShotEffectivenessResponse,
  ShotEffectivenessPerRallyResponse,
  ShotPlacementResponse,
  WinningStatsResponse,
  RallyIntensityResponse,
  RallyTimelineResponse,
  MomentumTimelineResponse,
  TimeToTTimelineResponse,
  LongestRallyResponse,
  FastestShotResponse,
  LetStatsResponse,
  BreakTimeResponse,
} from '@/types/api';

/**
 * Build query parameters from filters
 */
const buildQueryParams = (filters: AnalyticsFilters): URLSearchParams => {
  const params = new URLSearchParams();

  if (filters.game_number !== undefined) {
    params.append('game_number', filters.game_number.toString());
  }
  if (filters.player_id !== undefined) {
    params.append('player_id', filters.player_id.toString());
  }
  if (filters.start_time !== undefined) {
    params.append('start_time', filters.start_time.toString());
  }
  if (filters.end_time !== undefined) {
    params.append('end_time', filters.end_time.toString());
  }

  return params;
};

// ============================================================================
// MATCH SUMMARY
// ============================================================================

/**
 * Get complete match summary including game results and winner
 */
export const getMatchSummary = async (
  videoId: string
): Promise<MatchSummaryResponse> => {
  const { data } = await apiClient.get<MatchSummaryResponse>(
    `/api/analysis/${videoId}/match-summary`
  );
  return data;
};

// ============================================================================
// DISTRIBUTION ANALYTICS
// ============================================================================

/**
 * Get stroke distribution (forehand vs backhand)
 */
export const getStrokeDistribution = async (
  videoId: string,
  filters: AnalyticsFilters = {}
): Promise<StrokeDistributionResponse> => {
  const params = buildQueryParams(filters);
  const { data } = await apiClient.get<StrokeDistributionResponse>(
    `/api/analysis/${videoId}/analytics/stroke-distribution?${params}`
  );
  return data;
};

/**
 * Get shot types distribution
 */
export const getShotTypeDistribution = async (
  videoId: string,
  filters: AnalyticsFilters = {}
): Promise<ShotTypeDistributionResponse> => {
  const params = buildQueryParams(filters);
  const { data } = await apiClient.get<ShotTypeDistributionResponse>(
    `/api/analysis/${videoId}/analytics/shot-types-distribution?${params}`
  );
  return data;
};

/**
 * Get court quadrants distribution
 */
export const getCourtQuadrants = async (
  videoId: string,
  filters: AnalyticsFilters = {}
): Promise<CourtQuadrantResponse> => {
  const params = buildQueryParams(filters);
  const { data } = await apiClient.get<CourtQuadrantResponse>(
    `/api/analysis/${videoId}/analytics/court-quadrants?${params}`
  );
  return data;
};

/**
 * Get wall quadrants distribution
 */
export const getWallQuadrants = async (
  videoId: string,
  filters: AnalyticsFilters = {}
): Promise<WallQuadrantResponse> => {
  const params = buildQueryParams(filters);
  const { data } = await apiClient.get<WallQuadrantResponse>(
    `/api/analysis/${videoId}/analytics/wall-quadrants?${params}`
  );
  return data;
};

// ============================================================================
// AGGREGATE ANALYTICS
// ============================================================================

/**
 * Get ball speed statistics
 */
export const getBallSpeed = async (
  videoId: string,
  filters: AnalyticsFilters = {}
): Promise<BallSpeedResponse> => {
  const params = buildQueryParams(filters);
  const { data } = await apiClient.get<BallSpeedResponse>(
    `/api/analysis/${videoId}/analytics/ball-speed?${params}`
  );
  return data;
};

/**
 * Get rhythm disruption metrics
 */
export const getRhythmDisruption = async (
  videoId: string,
  filters: AnalyticsFilters = {}
): Promise<RhythmDisruptionResponse> => {
  const params = buildQueryParams(filters);
  const { data } = await apiClient.get<RhythmDisruptionResponse>(
    `/api/analysis/${videoId}/analytics/rhythm-disruption?${params}`
  );
  return data;
};

/**
 * Get shot placement effectiveness for a specific player
 */
export const getShotPlacement = async (
  videoId: string,
  playerId: 1 | 2,
  filters: AnalyticsFilters = {}
): Promise<ShotPlacementResponse> => {
  const params = buildQueryParams(filters);
  const { data } = await apiClient.get<ShotPlacementResponse>(
    `/api/analysis/${videoId}/analytics/shot-placement/${playerId}?${params}`
  );
  return data;
};

/**
 * Get winning statistics for a specific player
 */
export const getWinningStats = async (
  videoId: string,
  playerId: 1 | 2,
  filters: AnalyticsFilters = {}
): Promise<WinningStatsResponse> => {
  const params = buildQueryParams(filters);
  const { data } = await apiClient.get<WinningStatsResponse>(
    `/api/analysis/${videoId}/analytics/winning-stats/${playerId}?${params}`
  );
  return data;
};

/**
 * Get rally intensity metrics
 */
export const getRallyIntensity = async (
  videoId: string,
  filters: AnalyticsFilters = {}
): Promise<RallyIntensityResponse> => {
  const params = buildQueryParams(filters);
  const { data } = await apiClient.get<RallyIntensityResponse>(
    `/api/analysis/${videoId}/analytics/rally-intensity?${params}`
  );
  return data;
};

// ============================================================================
// SPATIAL ANALYTICS (HEATMAPS)
// ============================================================================

/**
 * Get player position heatmap
 */
export const getPlayerPositionHeatmap = async (
  videoId: string,
  filters: AnalyticsFilters = {}
): Promise<PlayerPositionHeatmapResponse> => {
  const params = buildQueryParams(filters);
  const { data } = await apiClient.get<PlayerPositionHeatmapResponse>(
    `/api/analysis/${videoId}/analytics/player-heatmap?${params}`
  );
  return data;
};

/**
 * Get wall hits heatmap
 */
export const getWallHitsHeatmap = async (
  videoId: string,
  filters: AnalyticsFilters = {}
): Promise<WallHitHeatmapResponse> => {
  const params = buildQueryParams(filters);
  const { data } = await apiClient.get<WallHitHeatmapResponse>(
    `/api/analysis/${videoId}/analytics/wall-hits-heatmap?${params}`
  );
  return data;
};

// ============================================================================
// MOVEMENT & T-ZONE ANALYTICS
// ============================================================================

/**
 * Get movement metrics
 */
export const getMovementMetrics = async (
  videoId: string,
  filters: AnalyticsFilters = {}
): Promise<MovementMetricsResponse> => {
  const params = buildQueryParams(filters);
  const { data } = await apiClient.get<MovementMetricsResponse>(
    `/api/analysis/${videoId}/analytics/movement-metrics?${params}`
  );
  return data;
};

/**
 * Get T-zone occupancy metrics
 */
export const getTZoneOccupancy = async (
  videoId: string,
  filters: AnalyticsFilters = {}
): Promise<TZoneOccupancyResponse> => {
  const params = buildQueryParams(filters);
  const { data } = await apiClient.get<TZoneOccupancyResponse>(
    `/api/analysis/${videoId}/analytics/t-zone-occupancy?${params}`
  );
  return data;
};

/**
 * Get shot effectiveness metrics for a specific player
 */
export const getShotEffectiveness = async (
  videoId: string,
  playerId: 1 | 2,
  filters: AnalyticsFilters = {}
): Promise<ShotEffectivenessResponse> => {
  const params = buildQueryParams(filters);
  const { data } = await apiClient.get<ShotEffectivenessResponse>(
    `/api/analysis/${videoId}/analytics/shot-effectiveness/${playerId}?${params}`
  );
  return data;
};

// ============================================================================
// TIME-SERIES ANALYTICS
// ============================================================================

/**
 * Get rally timeline with per-rally metrics
 */
export const getRallyTimeline = async (
  videoId: string,
  filters: AnalyticsFilters = {}
): Promise<RallyTimelineResponse> => {
  const params = buildQueryParams(filters);
  const { data } = await apiClient.get<RallyTimelineResponse>(
    `/api/analysis/${videoId}/analytics/rally-timeline?${params}`
  );
  return data;
};

/**
 * Get momentum timeline (cumulative score progression)
 */
export const getMomentumTimeline = async (
  videoId: string,
  filters: AnalyticsFilters = {}
): Promise<MomentumTimelineResponse> => {
  const params = buildQueryParams(filters);
  const { data } = await apiClient.get<MomentumTimelineResponse>(
    `/api/analysis/${videoId}/analytics/momentum-timeline?${params}`
  );
  return data;
};

/**
 * Get time-to-T timeline (per-rally T-zone recovery metrics)
 */
export const getTimeToTTimeline = async (
  videoId: string,
  filters: AnalyticsFilters = {}
): Promise<TimeToTTimelineResponse> => {
  const params = buildQueryParams(filters);
  const { data } = await apiClient.get<TimeToTTimelineResponse>(
    `/api/analysis/${videoId}/analytics/time-to-t-timeline?${params}`
  );
  return data;
};

// ============================================================================
// MATCH HIGHLIGHTS
// ============================================================================

/**
 * Get the longest rally in the match
 */
export const getLongestRally = async (
  videoId: string,
  filters: AnalyticsFilters = {}
): Promise<LongestRallyResponse> => {
  const params = buildQueryParams(filters);
  const { data } = await apiClient.get<LongestRallyResponse>(
    `/api/analysis/${videoId}/analytics/longest-rally?${params}`
  );
  return data;
};

/**
 * Get the fastest shot in the match
 */
export const getFastestShot = async (
  videoId: string,
  filters: AnalyticsFilters = {}
): Promise<FastestShotResponse> => {
  const params = buildQueryParams(filters);
  const { data } = await apiClient.get<FastestShotResponse>(
    `/api/analysis/${videoId}/analytics/fastest-shot?${params}`
  );
  return data;
};

/**
 * Get let/replay statistics
 */
export const getLetStats = async (
  videoId: string,
  filters: AnalyticsFilters = {}
): Promise<LetStatsResponse> => {
  const params = buildQueryParams(filters);
  const { data } = await apiClient.get<LetStatsResponse>(
    `/api/analysis/${videoId}/analytics/let-stats?${params}`
  );
  return data;
};

/**
 * Get break time statistics between rallies
 */
export const getBreakTime = async (
  videoId: string,
  filters: AnalyticsFilters = {}
): Promise<BreakTimeResponse> => {
  const params = buildQueryParams(filters);
  const { data } = await apiClient.get<BreakTimeResponse>(
    `/api/analysis/${videoId}/analytics/break-time?${params}`
  );
  return data;
};

/**
 * Get ball speed per rally with both players' data
 */
export const getBallSpeedPerRally = async (
  videoId: string,
  filters: AnalyticsFilters = {}
): Promise<BallSpeedPerRallyResponse> => {
  const params = buildQueryParams(filters);
  const { data } = await apiClient.get<BallSpeedPerRallyResponse>(
    `/api/analysis/${videoId}/analytics/ball-speed/per-rally?${params}`
  );
  return data;
};

/**
 * Get shot effectiveness per rally with both players' data
 */
export const getShotEffectivenessPerRally = async (
  videoId: string,
  filters: AnalyticsFilters = {}
): Promise<ShotEffectivenessPerRallyResponse> => {
  const params = buildQueryParams(filters);
  const { data } = await apiClient.get<ShotEffectivenessPerRallyResponse>(
    `/api/analysis/${videoId}/analytics/shot-effectiveness/per-rally?${params}`
  );
  return data;
};

/**
 * Get T-zone occupancy per rally with both players' data
 */
export const getTZoneOccupancyPerRally = async (
  videoId: string,
  filters: AnalyticsFilters = {}
): Promise<TZoneOccupancyPerRallyResponse> => {
  const params = buildQueryParams(filters);
  const { data } = await apiClient.get<TZoneOccupancyPerRallyResponse>(
    `/api/analysis/${videoId}/analytics/t-zone-occupancy/per-rally?${params}`
  );
  return data;
};
