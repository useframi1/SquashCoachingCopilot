"""Pydantic schemas for analysis-related API operations - Redesigned for consistency."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# BASE SCHEMAS
# ============================================================================


class AnalyticsFilters(BaseModel):
    """Common filters for analytics queries."""

    game_number: Optional[int] = Field(None, ge=1, description="Filter by specific game number")
    player_id: Optional[int] = Field(
        None, ge=1, le=2, description="Filter by player (1 or 2)"
    )
    start_time: Optional[float] = Field(
        None, ge=0, description="Start timestamp in seconds"
    )
    end_time: Optional[float] = Field(
        None, ge=0, description="End timestamp in seconds"
    )

    @field_validator("end_time")
    @classmethod
    def end_after_start(cls, v: Optional[float], info) -> Optional[float]:
        """Validate that end_time is after start_time."""
        if v is not None and info.data.get("start_time") is not None:
            if v <= info.data["start_time"]:
                raise ValueError("end_time must be after start_time")
        return v


class AnalyticsResponseBase(BaseModel):
    """Base schema for all analytics responses - minimal envelope."""

    video_id: str
    filters: Optional[AnalyticsFilters] = None


# ============================================================================
# PATTERN 1: DISTRIBUTION PATTERN
# For: Stroke distribution, Shot types, Court quadrants
# ============================================================================


class DistributionItem(BaseModel):
    """Single item in a distribution (chart-ready)."""

    label: str = Field(
        description="Category label (e.g., 'forehand', 'straight_drive')"
    )
    count: int = Field(description="Raw count")
    percentage: float = Field(description="Pre-computed percentage")




# ============================================================================
# PATTERN 3: SPATIAL PATTERN
# For: Player heatmap, Wall hits, Shot placement
# ============================================================================


class HeatmapGrid(BaseModel):
    """Pre-computed heatmap grid for visualization (32x50 for squash court)."""

    grid: List[List[float]] = Field(
        description="2D array of density values (percentages), shape: height x width"
    )
    width: int = Field(default=32, description="Grid width (columns)")
    height: int = Field(default=50, description="Grid height (rows)")
    bounds: Dict[str, float] = Field(
        description="Spatial boundaries (x_min, x_max, y_min, y_max, units)"
    )


class SpatialData(BaseModel):
    """Spatial heatmap data for a single player or aggregated totals."""

    heatmap_grid: HeatmapGrid


# ============================================================================
# ENDPOINT-SPECIFIC RESPONSE SCHEMAS
# ============================================================================


class SingleDistribution(BaseModel):
    """Distribution data for a single player or aggregated totals."""

    distribution: List[DistributionItem] = Field(
        description="Array of items (chart-ready)"
    )
    total: int = Field(description="Total count across all items")


class StrokeDistributionResponse(AnalyticsResponseBase):
    """Stroke distribution analytics (forehand vs backhand)."""

    data: SingleDistribution


class ShotTypeDistributionResponse(AnalyticsResponseBase):
    """Shot type distribution analytics."""

    data: SingleDistribution
    all_shot_types: List[str] = Field(
        description="List of all shot types found (for consistent chart colors)"
    )


class SingleAggregate(BaseModel):
    """Aggregate statistics for a single player or aggregated totals."""

    mean: float = Field(description="Mean value")
    min: float = Field(description="Minimum value")
    max: float = Field(description="Maximum value")
    std_dev: float = Field(description="Standard deviation")
    count: int = Field(description="Number of data points")


class BallSpeedData(BaseModel):
    """Ball speed aggregate statistics."""

    mean_speed: float = Field(description="Mean ball speed (m/s)")
    min_speed: float = Field(description="Minimum ball speed (m/s)")
    max_speed: float = Field(description="Maximum ball speed (m/s)")
    std_dev: float = Field(description="Standard deviation of ball speed")
    shot_count: int = Field(description="Number of shots analyzed")


class BallSpeedResponse(AnalyticsResponseBase):
    """Ball speed aggregate statistics (no time series)."""

    data: BallSpeedData


class RhythmDisruptionData(BaseModel):
    """Rhythm disruption aggregate statistics."""

    ball_speed_cv: float = Field(
        description="Coefficient of variation for ball speed (higher = more unpredictable)"
    )
    ball_speed_variance: float = Field(description="Variance of ball speed")
    wall_hit_height_cv: float = Field(
        description="Coefficient of variation for wall hit height (higher = more unpredictable)"
    )
    wall_hit_height_variance: float = Field(description="Variance of wall hit height")
    shot_count: int = Field(description="Number of shots analyzed")


class RhythmDisruptionResponse(AnalyticsResponseBase):
    """Rhythm disruption analytics (aggregate only)."""

    data: RhythmDisruptionData


class PlayerPositionHeatmapResponse(AnalyticsResponseBase):
    """Player position heatmap data."""

    data: SpatialData


# ShotPlacementData and ShotPlacementResponse removed - use ShotEffectiveness instead


class CourtQuadrantResponse(AnalyticsResponseBase):
    """Court quadrant distribution analytics."""

    data: SingleDistribution
    quadrant_boundaries: Dict[str, float] = Field(
        description="Quadrant boundaries (x_cut, y_cut) in meters"
    )


class WallHitHeatmapResponse(AnalyticsResponseBase):
    """Wall hit position heatmap data."""

    data: SpatialData


class WinningEfficiencyData(BaseModel):
    """Winning efficiency aggregate data."""

    shots_per_point_won: float = Field(
        description="Average shots needed to win a point (main efficiency metric, lower is better)"
    )
    points_won: int = Field(description="Total points won")
    total_shots: int = Field(description="Total shots taken")
    win_rate: float = Field(description="Percentage of rallies won")
    rallies_played: int = Field(description="Total rallies played")


class WinningEfficiencyResponse(AnalyticsResponseBase):
    """Winning efficiency analytics (aggregate only)."""

    data: WinningEfficiencyData


class WallQuadrantResponse(AnalyticsResponseBase):
    """Wall quadrant distribution analytics."""

    data: SingleDistribution
    quadrant_boundaries: Dict[str, float] = Field(
        description="Quadrant boundaries (x_cut, y_cut) in meters"
    )


# ============================================================================
# PATTERN 5: EXTENDED ANALYTICS (Movement, T-Zone, Shot Effectiveness, Intensity)
# ============================================================================


class SingleMovementMetrics(BaseModel):
    """Movement metrics for a single player or aggregated totals."""

    total_distance: float = Field(description="Total distance covered in meters")
    avg_distance_per_rally: float = Field(
        description="Average distance per rally in meters"
    )


class SingleTZoneMetrics(BaseModel):
    """T-zone occupancy metrics for a single player or aggregated totals."""

    pct_time_in_t: float = Field(description="% of frames in T-zone")
    avg_time_to_t: Optional[float] = Field(
        description="Average time to reach T-zone after opponent shot (seconds)"
    )
    min_time_to_t: Optional[float] = Field(
        description="Minimum time to reach T-zone (seconds)"
    )
    max_time_to_t: Optional[float] = Field(
        description="Maximum time to reach T-zone (seconds)"
    )
    time_to_t_variance: Optional[float] = Field(
        description="Variance in time-to-T measurements"
    )
    t_zone_success_rate: Optional[float] = Field(
        description="% of opponent shots where player reached T-zone"
    )
    total_shots_taken: int = Field(description="Total shots taken by player")
    successful_returns: int = Field(
        description="Number of opponent shots where player reached T"
    )


class SingleShotEffectivenessMetrics(BaseModel):
    """Shot effectiveness metrics for a single player (without player_id)."""

    avg_displacement_from_t: Optional[float] = Field(
        description="Average distance opponent moved away from T after shots (meters)"
    )
    max_displacement_from_t: Optional[float] = Field(
        description="Maximum opponent displacement from T (meters)"
    )
    displacement_variance: Optional[float] = Field(
        description="Variance in opponent displacement from T"
    )
    avg_opponent_distance_moved: Optional[float] = Field(
        None, description="Average absolute distance opponent had to move to return shot (meters)"
    )
    max_opponent_distance_moved: Optional[float] = Field(
        None, description="Maximum distance opponent had to move to return shot (meters)"
    )
    opponent_distance_moved_variance: Optional[float] = Field(
        None, description="Variance in opponent distance moved"
    )
    depth_dominance_pct: Optional[float] = Field(
        description="% of shots where opponent was deeper (closer to back wall)"
    )
    avg_depth_difference: Optional[float] = Field(
        description="Average Y-coordinate difference (opponent_y - player_y) in meters"
    )
    min_depth_difference: Optional[float] = Field(
        description="Minimum depth difference in meters"
    )
    max_depth_difference: Optional[float] = Field(
        description="Maximum depth difference in meters"
    )
    straight_shot_quality_pct: Optional[float] = Field(
        description="% of straight shots hit close to wall (<1.2m)"
    )
    straight_shots_count: int = Field(description="Total straight shots taken")
    shots_close_to_wall: int = Field(
        description="Straight shots within 1.2m of wall"
    )


# ============================================================================
# EXTENDED ANALYTICS RESPONSE SCHEMAS
# ============================================================================


class MovementMetricsResponse(AnalyticsResponseBase):
    """Movement and distance analytics."""

    data: SingleMovementMetrics


class TZoneOccupancyResponse(AnalyticsResponseBase):
    """T-zone occupancy and positioning analytics."""

    data: SingleTZoneMetrics


class ShotEffectivenessResponse(AnalyticsResponseBase):
    """Shot effectiveness and placement quality analytics."""

    data: SingleShotEffectivenessMetrics


class RallyIntensityData(BaseModel):
    """Rally intensity aggregate statistics."""

    avg_seconds_per_shot: float = Field(
        description="Average seconds per shot (lower = faster/more intense)"
    )
    min_seconds_per_shot: float = Field(
        description="Minimum seconds per shot (fastest rally)"
    )
    max_seconds_per_shot: float = Field(
        description="Maximum seconds per shot (slowest rally)"
    )
    std_dev: float = Field(description="Standard deviation of seconds per shot")
    rally_count: int = Field(description="Number of rallies analyzed")


class RallyIntensityResponse(AnalyticsResponseBase):
    """Rally intensity and pace analytics (aggregate only)."""

    data: RallyIntensityData


class LetStatsData(BaseModel):
    """Let/replay statistics."""

    total_lets: int = Field(description="Total number of lets/replays")
    total_rallies: int = Field(description="Total number of rallies")
    let_percentage: float = Field(description="Percentage of rallies that were lets")


class LetStatsResponse(AnalyticsResponseBase):
    """Let/replay statistics response."""

    data: LetStatsData


class BreakTimeData(BaseModel):
    """Break time between rallies statistics."""

    avg_break_time: float = Field(description="Average break time between rallies (seconds)")
    min_break_time: float = Field(description="Minimum break time between rallies (seconds)")
    max_break_time: float = Field(description="Maximum break time between rallies (seconds)")
    std_dev: float = Field(description="Standard deviation of break times")
    total_breaks: int = Field(description="Number of breaks analyzed")


class BreakTimeResponse(AnalyticsResponseBase):
    """Break time between rallies response."""

    data: BreakTimeData


# ============================================================================
# PATTERN 6: TIME-SERIES PATTERN
# For: Rally timeline, Momentum timeline (per-rally granularity)
# ============================================================================


class RallyTimelineItem(BaseModel):
    """Single rally's timeline data (streamlined - use dedicated endpoints for ball speed and rhythm)."""

    rally_id: int
    rally_start_time: float = Field(description="Timestamp when rally started (seconds)")
    rally_duration: float = Field(description="Rally length in seconds")
    shot_count: int = Field(description="Total shots in rally")
    point_winner: Optional[int] = Field(None, description="Player who won (1 or 2)")
    wall_hit_count: int = Field(description="Number of wall hits in rally")
    # Removed fields (use dedicated endpoints instead):
    # - avg_ball_speed: Use /ball-speed/per-rally endpoint
    # - ball_speed_variance: Use /rhythm-disruption/per-rally endpoint


class RallyTimelineResponse(AnalyticsResponseBase):
    """Rally timeline response with metadata."""

    data: List[RallyTimelineItem]
    total_rallies: int = Field(description="Total number of rallies returned")


class MomentumTimelineItem(BaseModel):
    """Cumulative score at each rally."""

    rally_id: int
    timestamp: float = Field(description="Rally timestamp")
    point_winner: Optional[int] = Field(None, description="Who won this rally (1 or 2)")
    player_1_score: int = Field(description="Player 1 cumulative score")
    player_2_score: int = Field(description="Player 2 cumulative score")
    score_differential: int = Field(description="Player 1 score - Player 2 score")


class MomentumTimelineResponse(AnalyticsResponseBase):
    """Momentum timeline with cumulative scores."""

    data: List[MomentumTimelineItem]


# ============================================================================
# PATTERN 7: PER-GAME TIME-SERIES PATTERN
# Base schemas for per-game time-series endpoints (dual-player data)
# ============================================================================


class PerGameTimeSeriesItem(BaseModel):
    """Base class for per-game time-series items with game metadata and dual-player data."""

    game_number: int = Field(description="Game number (1-5)")
    start_rally_id: int = Field(description="First rally ID in this game")
    end_rally_id: int = Field(description="Last rally ID in this game")
    start_time: float = Field(description="Game start timestamp (seconds)")
    end_time: float = Field(description="Game end timestamp (seconds)")
    duration: float = Field(description="Game duration (seconds)")
    rally_count: int = Field(description="Number of rallies in this game")


class PerGameTimeSeriesResponse(AnalyticsResponseBase):
    """Base class for per-game time-series responses."""

    total_games: int = Field(description="Total number of games returned")


# ============================================================================
# PATTERN 8: PER-RALLY TIME-SERIES PATTERN
# Base schemas for per-rally time-series endpoints (dual-player data)
# ============================================================================


class PerRallyTimeSeriesItem(BaseModel):
    """Base class for per-rally time-series items with rally metadata and dual-player data."""

    rally_id: int = Field(description="Rally identifier")
    game_number: Optional[int] = Field(None, description="Game number this rally belongs to (null if games not tracked)")
    rally_start_time: float = Field(description="Rally start timestamp (seconds)")
    rally_duration: float = Field(description="Rally duration (seconds)")
    shot_count: int = Field(description="Total shots in rally")
    point_winner: Optional[int] = Field(None, description="Player who won (1 or 2)")


class PerRallyTimeSeriesResponse(AnalyticsResponseBase):
    """Base class for per-rally time-series responses."""

    total_rallies: int = Field(description="Total number of rallies returned")


# ============================================================================
# METRIC-SPECIFIC TIME-SERIES SCHEMAS
# For each of the 11 metric categories, define per-game and per-rally schemas
# ============================================================================

# --- 1. Stroke Distribution ---

class StrokeDistributionPerGameItem(PerGameTimeSeriesItem):
    """Per-game stroke distribution with both players' data."""

    player_1: SingleDistribution = Field(description="Player 1 stroke distribution")
    player_2: SingleDistribution = Field(description="Player 2 stroke distribution")


class StrokeDistributionPerGameResponse(PerGameTimeSeriesResponse):
    """Per-game stroke distribution response."""

    data: List[StrokeDistributionPerGameItem]


class StrokeDistributionPerRallyItem(PerRallyTimeSeriesItem):
    """Per-rally stroke distribution with both players' data."""

    player_1: SingleDistribution = Field(description="Player 1 stroke distribution")
    player_2: SingleDistribution = Field(description="Player 2 stroke distribution")


class StrokeDistributionPerRallyResponse(PerRallyTimeSeriesResponse):
    """Per-rally stroke distribution response."""

    data: List[StrokeDistributionPerRallyItem]


# --- 2. Shot Type Distribution ---

class ShotTypeDistributionPerGameItem(PerGameTimeSeriesItem):
    """Per-game shot type distribution with both players' data."""

    player_1: SingleDistribution = Field(description="Player 1 shot type distribution")
    player_2: SingleDistribution = Field(description="Player 2 shot type distribution")


class ShotTypeDistributionPerGameResponse(PerGameTimeSeriesResponse):
    """Per-game shot type distribution response."""

    data: List[ShotTypeDistributionPerGameItem]
    all_shot_types: List[str] = Field(
        description="List of all shot types found (for consistent chart colors)"
    )


class ShotTypeDistributionPerRallyItem(PerRallyTimeSeriesItem):
    """Per-rally shot type distribution with both players' data."""

    player_1: SingleDistribution = Field(description="Player 1 shot type distribution")
    player_2: SingleDistribution = Field(description="Player 2 shot type distribution")


class ShotTypeDistributionPerRallyResponse(PerRallyTimeSeriesResponse):
    """Per-rally shot type distribution response."""

    data: List[ShotTypeDistributionPerRallyItem]
    all_shot_types: List[str] = Field(
        description="List of all shot types found (for consistent chart colors)"
    )


# --- 3. Ball Speed ---

class BallSpeedPerGameItem(PerGameTimeSeriesItem):
    """Per-game ball speed with both players' data."""

    player_1: BallSpeedData = Field(description="Player 1 ball speed stats")
    player_2: BallSpeedData = Field(description="Player 2 ball speed stats")


class BallSpeedPerGameResponse(PerGameTimeSeriesResponse):
    """Per-game ball speed response."""

    data: List[BallSpeedPerGameItem]


class BallSpeedPerRallyItem(PerRallyTimeSeriesItem):
    """Per-rally ball speed with both players' data."""

    player_1: BallSpeedData = Field(description="Player 1 ball speed stats")
    player_2: BallSpeedData = Field(description="Player 2 ball speed stats")


class BallSpeedPerRallyResponse(PerRallyTimeSeriesResponse):
    """Per-rally ball speed response."""

    data: List[BallSpeedPerRallyItem]


# --- 4. Rhythm Disruption ---

class RhythmDisruptionPerGameItem(PerGameTimeSeriesItem):
    """Per-game rhythm disruption with both players' data."""

    player_1: RhythmDisruptionData = Field(description="Player 1 rhythm disruption stats")
    player_2: RhythmDisruptionData = Field(description="Player 2 rhythm disruption stats")


class RhythmDisruptionPerGameResponse(PerGameTimeSeriesResponse):
    """Per-game rhythm disruption response."""

    data: List[RhythmDisruptionPerGameItem]


class RhythmDisruptionPerRallyItem(PerRallyTimeSeriesItem):
    """Per-rally rhythm disruption with both players' data."""

    player_1: RhythmDisruptionData = Field(description="Player 1 rhythm disruption stats")
    player_2: RhythmDisruptionData = Field(description="Player 2 rhythm disruption stats")


class RhythmDisruptionPerRallyResponse(PerRallyTimeSeriesResponse):
    """Per-rally rhythm disruption response."""

    data: List[RhythmDisruptionPerRallyItem]


# --- 5. Court Quadrants ---

class CourtQuadrantPerGameItem(PerGameTimeSeriesItem):
    """Per-game court quadrant distribution with both players' data."""

    player_1: SingleDistribution = Field(description="Player 1 court quadrant distribution")
    player_2: SingleDistribution = Field(description="Player 2 court quadrant distribution")


class CourtQuadrantPerGameResponse(PerGameTimeSeriesResponse):
    """Per-game court quadrant response."""

    data: List[CourtQuadrantPerGameItem]
    quadrant_boundaries: Dict[str, float] = Field(
        description="Quadrant boundaries (x_cut, y_cut) in meters"
    )


class CourtQuadrantPerRallyItem(PerRallyTimeSeriesItem):
    """Per-rally court quadrant distribution with both players' data."""

    player_1: SingleDistribution = Field(description="Player 1 court quadrant distribution")
    player_2: SingleDistribution = Field(description="Player 2 court quadrant distribution")


class CourtQuadrantPerRallyResponse(PerRallyTimeSeriesResponse):
    """Per-rally court quadrant response."""

    data: List[CourtQuadrantPerRallyItem]
    quadrant_boundaries: Dict[str, float] = Field(
        description="Quadrant boundaries (x_cut, y_cut) in meters"
    )


# --- 6. Wall Quadrants ---

class WallQuadrantPerGameItem(PerGameTimeSeriesItem):
    """Per-game wall quadrant distribution with both players' data."""

    player_1: SingleDistribution = Field(description="Player 1 wall quadrant distribution")
    player_2: SingleDistribution = Field(description="Player 2 wall quadrant distribution")


class WallQuadrantPerGameResponse(PerGameTimeSeriesResponse):
    """Per-game wall quadrant response."""

    data: List[WallQuadrantPerGameItem]
    quadrant_boundaries: Dict[str, float] = Field(
        description="Quadrant boundaries (x_cut, y_cut) in meters"
    )


class WallQuadrantPerRallyItem(PerRallyTimeSeriesItem):
    """Per-rally wall quadrant distribution with both players' data."""

    player_1: SingleDistribution = Field(description="Player 1 wall quadrant distribution")
    player_2: SingleDistribution = Field(description="Player 2 wall quadrant distribution")


class WallQuadrantPerRallyResponse(PerRallyTimeSeriesResponse):
    """Per-rally wall quadrant response."""

    data: List[WallQuadrantPerRallyItem]
    quadrant_boundaries: Dict[str, float] = Field(
        description="Quadrant boundaries (x_cut, y_cut) in meters"
    )


# --- 7. Movement Metrics ---

class MovementMetricsPerGameItem(PerGameTimeSeriesItem):
    """Per-game movement metrics with both players' data."""

    player_1: SingleMovementMetrics = Field(description="Player 1 movement metrics")
    player_2: SingleMovementMetrics = Field(description="Player 2 movement metrics")


class MovementMetricsPerGameResponse(PerGameTimeSeriesResponse):
    """Per-game movement metrics response."""

    data: List[MovementMetricsPerGameItem]


class MovementMetricsPerRallyItem(PerRallyTimeSeriesItem):
    """Per-rally movement metrics with both players' data."""

    player_1: SingleMovementMetrics = Field(description="Player 1 movement metrics")
    player_2: SingleMovementMetrics = Field(description="Player 2 movement metrics")


class MovementMetricsPerRallyResponse(PerRallyTimeSeriesResponse):
    """Per-rally movement metrics response."""

    data: List[MovementMetricsPerRallyItem]


# --- 8. T-Zone Occupancy ---

class TZoneOccupancyPerGameItem(PerGameTimeSeriesItem):
    """Per-game T-zone occupancy with both players' data."""

    player_1: SingleTZoneMetrics = Field(description="Player 1 T-zone metrics")
    player_2: SingleTZoneMetrics = Field(description="Player 2 T-zone metrics")


class TZoneOccupancyPerGameResponse(PerGameTimeSeriesResponse):
    """Per-game T-zone occupancy response."""

    data: List[TZoneOccupancyPerGameItem]


class TZoneOccupancyPerRallyItem(PerRallyTimeSeriesItem):
    """Per-rally T-zone occupancy with both players' data."""

    player_1: SingleTZoneMetrics = Field(description="Player 1 T-zone metrics")
    player_2: SingleTZoneMetrics = Field(description="Player 2 T-zone metrics")


class TZoneOccupancyPerRallyResponse(PerRallyTimeSeriesResponse):
    """Per-rally T-zone occupancy response."""

    data: List[TZoneOccupancyPerRallyItem]


# --- 9. Shot Effectiveness (both players) ---

class ShotEffectivenessPerGameItem(PerGameTimeSeriesItem):
    """Per-game shot effectiveness with both players' data."""

    player_1: SingleShotEffectivenessMetrics = Field(description="Player 1 shot effectiveness metrics")
    player_2: SingleShotEffectivenessMetrics = Field(description="Player 2 shot effectiveness metrics")


class ShotEffectivenessPerGameResponse(PerGameTimeSeriesResponse):
    """Per-game shot effectiveness response."""

    data: List[ShotEffectivenessPerGameItem]


class ShotEffectivenessPerRallyItem(PerRallyTimeSeriesItem):
    """Per-rally shot effectiveness with both players' data."""

    player_1: SingleShotEffectivenessMetrics = Field(description="Player 1 shot effectiveness metrics")
    player_2: SingleShotEffectivenessMetrics = Field(description="Player 2 shot effectiveness metrics")


class ShotEffectivenessPerRallyResponse(PerRallyTimeSeriesResponse):
    """Per-rally shot effectiveness response."""

    data: List[ShotEffectivenessPerRallyItem]


# --- 10. Winning Efficiency (both players) ---

class WinningEfficiencyPerGameItem(PerGameTimeSeriesItem):
    """Per-game winning efficiency with both players' data."""

    player_1: WinningEfficiencyData = Field(description="Player 1 winning efficiency statistics")
    player_2: WinningEfficiencyData = Field(description="Player 2 winning efficiency statistics")


class WinningEfficiencyPerGameResponse(PerGameTimeSeriesResponse):
    """Per-game winning efficiency response."""

    data: List[WinningEfficiencyPerGameItem]


class WinningEfficiencyPerRallyItem(PerRallyTimeSeriesItem):
    """Per-rally winning efficiency with both players' data."""

    player_1: WinningEfficiencyData = Field(description="Player 1 winning efficiency statistics")
    player_2: WinningEfficiencyData = Field(description="Player 2 winning efficiency statistics")


class WinningEfficiencyPerRallyResponse(PerRallyTimeSeriesResponse):
    """Per-rally winning efficiency response."""

    data: List[WinningEfficiencyPerRallyItem]


# --- 11. Rally Intensity ---

class RallyIntensityPerGameItem(PerGameTimeSeriesItem):
    """Per-game rally intensity (not split by player)."""

    data: RallyIntensityData = Field(description="Rally intensity metrics")


class RallyIntensityPerGameResponse(PerGameTimeSeriesResponse):
    """Per-game rally intensity response."""

    data: List[RallyIntensityPerGameItem]


class RallyIntensityPerRallyItem(PerRallyTimeSeriesItem):
    """Per-rally intensity (seconds per shot for this rally)."""

    seconds_per_shot: Optional[float] = Field(
        None, description="Seconds per shot in this rally (lower = faster)"
    )


class RallyIntensityPerRallyResponse(PerRallyTimeSeriesResponse):
    """Per-rally intensity response."""

    data: List[RallyIntensityPerRallyItem]


# ============================================================================
# MATCH HIGHLIGHTS SCHEMAS
# ============================================================================


class LongestRallyData(BaseModel):
    """Longest rally in the match."""

    rally_id: int = Field(description="Rally identifier")
    game_number: Optional[int] = Field(None, description="Game number this rally belongs to")
    rally_start_time: float = Field(description="Rally start timestamp (seconds)")
    rally_duration: float = Field(description="Rally duration (seconds)")
    shot_count: int = Field(description="Total shots in rally")
    point_winner: Optional[int] = Field(None, description="Player who won (1 or 2)")


class LongestRallyResponse(AnalyticsResponseBase):
    """Longest rally in the match by duration and shot count."""

    data: LongestRallyData


class FastestShotData(BaseModel):
    """Fastest shot in the match."""

    frame_number: int = Field(description="Frame number where the shot occurred")
    timestamp: float = Field(description="Timestamp when the shot occurred (seconds)")
    rally_id: Optional[int] = Field(None, description="Rally identifier")
    game_number: Optional[int] = Field(None, description="Game number")
    player_id: int = Field(description="Player who hit the shot (1 or 2)")
    ball_speed: float = Field(description="Ball speed (m/s)")
    stroke_type: Optional[str] = Field(None, description="Stroke type (forehand/backhand)")
    shot_type: Optional[str] = Field(None, description="Shot type classification")


class FastestShotResponse(AnalyticsResponseBase):
    """Fastest shot in the match."""

    data: FastestShotData
