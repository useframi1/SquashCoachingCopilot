// TypeScript types for Squash Coaching Copilot API
// Generated from backend Pydantic schemas

// ============================================================================
// JOB & VIDEO TYPES
// ============================================================================

export enum JobStatus {
  PENDING = 'pending',
  PROCESSING = 'processing',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

export interface JobStatusResponse {
  id: string;
  status: JobStatus;
  progress: number; // 0-100
  current_stage: string | null;
}

export interface JobResponse extends JobStatusResponse {
  video_id: string;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface VideoUploadResponse {
  id: string;
  filename: string;
  message: string;
}

export interface VideoMetadata {
  id: string;
  filename: string;
  original_filename: string;
  fps: number;
  total_frames: number;
  width: number;
  height: number;
  duration_seconds: number;
  file_size_bytes: number;
  has_annotated_video: boolean;
  uploaded_at: string;
  processed_at: string | null;
}

// ============================================================================
// MATCH & GAME TYPES
// ============================================================================

export interface GameSchema {
  id: number;
  video_id: string;
  game_number: number;
  winner: 1 | 2 | null;
  player_1_score: number;
  player_2_score: number;
  start_rally_id: number;
  end_rally_id: number;
  start_time: number | null;
  end_time: number | null;
}

export interface MatchSchema {
  id: number;
  video_id: string;
  winner: 1 | 2 | null;
  player_1_games_won: number;
  player_2_games_won: number;
  total_rallies: number;
  total_games: number;
  scoring_system: string;
  best_of: number;
}

export interface MatchSummaryResponse {
  video_id: string;
  match: MatchSchema;
  games: GameSchema[];
}

// ============================================================================
// ANALYTICS BASE TYPES
// ============================================================================

export interface AnalyticsFilters {
  game_number?: number;
  player_id?: 1 | 2;
  start_time?: number;
  end_time?: number;
}

interface AnalyticsResponseBase {
  video_id: string;
  filters: AnalyticsFilters | null;
}

// ============================================================================
// DISTRIBUTION PATTERN
// ============================================================================

export interface DistributionItem {
  label: string;
  count: number;
  percentage: number;
}

export interface SingleDistribution {
  distribution: DistributionItem[];
  total: number;
}

export interface StrokeDistributionResponse extends AnalyticsResponseBase {
  data: SingleDistribution;
}

export interface ShotTypeDistributionResponse extends AnalyticsResponseBase {
  data: SingleDistribution;
  all_shot_types: string[];
}

export interface CourtQuadrantResponse extends AnalyticsResponseBase {
  data: SingleDistribution;
  quadrant_boundaries: {
    x_cut: number;
    y_cut: number;
  };
}

export interface WallQuadrantResponse extends AnalyticsResponseBase {
  data: SingleDistribution;
  quadrant_boundaries: {
    x_cut: number;
    y_cut: number;
  };
}

// ============================================================================
// AGGREGATE PATTERN
// ============================================================================

export interface BallSpeedData {
  mean_speed: number;
  min_speed: number;
  max_speed: number;
  std_dev: number;
  shot_count: number;
}

export interface BallSpeedResponse extends AnalyticsResponseBase {
  data: BallSpeedData;
}

export interface RhythmDisruptionData {
  ball_speed_cv: number;
  ball_speed_variance: number;
  wall_hit_height_cv: number;
  wall_hit_height_variance: number;
  shot_count: number;
}

export interface RhythmDisruptionResponse extends AnalyticsResponseBase {
  data: RhythmDisruptionData;
}

export interface ShotPlacementData {
  avg_opponent_distance_moved: number;
  min_opponent_distance_moved: number;
  max_opponent_distance_moved: number;
  std_dev: number;
  shot_count: number;
}

export interface ShotPlacementResponse extends AnalyticsResponseBase {
  data: ShotPlacementData;
}

export interface WinningStatsData {
  efficiency: number;
  points_won: number;
  total_shots: number;
  points_per_rally: number;
  rallies_played: number;
}

export interface WinningStatsResponse extends AnalyticsResponseBase {
  data: WinningStatsData;
}

export interface RallyIntensityData {
  avg_seconds_per_shot: number;
  min_seconds_per_shot: number;
  max_seconds_per_shot: number;
  std_dev: number;
  rally_count: number;
}

export interface RallyIntensityResponse extends AnalyticsResponseBase {
  data: RallyIntensityData;
}

// ============================================================================
// SPATIAL PATTERN (HEATMAPS)
// ============================================================================

export interface HeatmapGrid {
  grid: number[][]; // 2D array [height][width] with density percentages
  width: number;
  height: number;
  bounds: {
    x_min: number;
    x_max: number;
    y_min: number;
    y_max: number;
    units?: string;
  };
}

export interface SpatialData {
  heatmap_grid: HeatmapGrid;
}

export interface PlayerPositionHeatmapResponse extends AnalyticsResponseBase {
  data: SpatialData;
}

export interface WallHitHeatmapResponse extends AnalyticsResponseBase {
  data: SpatialData;
}

// ============================================================================
// MOVEMENT & T-ZONE ANALYTICS
// ============================================================================

export interface SingleMovementMetrics {
  total_distance: number;
  avg_distance_per_rally: number;
  avg_distance_to_ball: number;
  min_distance_to_ball: number | null;
  max_distance_to_ball: number | null;
  shot_count: number;
}

export interface MovementMetricsResponse extends AnalyticsResponseBase {
  data: SingleMovementMetrics;
}

export interface SingleTZoneMetrics {
  pct_time_in_t: number;
  avg_time_to_t: number | null;
  min_time_to_t: number | null;
  max_time_to_t: number | null;
  time_to_t_variance: number | null;
  t_zone_success_rate: number | null;
  total_shots_taken: number;
  successful_returns: number;
}

export interface TZoneOccupancyResponse extends AnalyticsResponseBase {
  data: SingleTZoneMetrics;
}

export interface SingleShotEffectivenessMetrics {
  avg_displacement_from_t: number | null;
  max_displacement_from_t: number | null;
  displacement_variance: number | null;
  depth_dominance_pct: number | null;
  avg_depth_difference: number | null;
  min_depth_difference: number | null;
  max_depth_difference: number | null;
  straight_shot_quality_pct: number | null;
  straight_shots_count: number;
  shots_close_to_wall: number;
}

export interface ShotEffectivenessResponse extends AnalyticsResponseBase {
  data: SingleShotEffectivenessMetrics;
}

// ============================================================================
// TIME-SERIES PATTERN
// ============================================================================

export interface RallyTimelineItem {
  rally_id: number;
  rally_start_time: number;
  rally_duration: number;
  shot_count: number;
  point_winner: 1 | 2 | null;
  avg_ball_speed: number | null;
  ball_speed_variance: number | null;
  wall_hit_count: number;
}

export interface RallyTimelineResponse extends AnalyticsResponseBase {
  data: RallyTimelineItem[];
  total_rallies: number;
}

export interface MomentumTimelineItem {
  rally_id: number;
  timestamp: number;
  point_winner: 1 | 2 | null;
  player_1_score: number;
  player_2_score: number;
  score_differential: number;
}

export interface MomentumTimelineResponse extends AnalyticsResponseBase {
  data: MomentumTimelineItem[];
}

export interface TimeToTTimelineItem {
  rally_id: number;
  rally_start_time: number;
  player_1_avg_time_to_t: number | null;
  player_1_min_time_to_t: number | null;
  player_1_max_time_to_t: number | null;
  player_1_measurements: number;
  player_2_avg_time_to_t: number | null;
  player_2_min_time_to_t: number | null;
  player_2_max_time_to_t: number | null;
  player_2_measurements: number;
}

export interface TimeToTTimelineResponse extends AnalyticsResponseBase {
  data: TimeToTTimelineItem[];
  total_rallies: number;
}
