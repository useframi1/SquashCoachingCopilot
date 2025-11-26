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


class ShotPlacementData(BaseModel):
    """Shot placement effectiveness aggregate statistics."""

    avg_opponent_distance_moved: float = Field(
        description="Average distance opponent moved after each shot (meters)"
    )
    min_opponent_distance_moved: float = Field(
        description="Minimum distance opponent moved (meters)"
    )
    max_opponent_distance_moved: float = Field(
        description="Maximum distance opponent moved (meters)"
    )
    std_dev: float = Field(description="Standard deviation of distance moved")
    shot_count: int = Field(description="Number of shots analyzed")


class ShotPlacementResponse(AnalyticsResponseBase):
    """Shot placement effectiveness analytics (aggregate only)."""

    data: ShotPlacementData


class CourtQuadrantResponse(AnalyticsResponseBase):
    """Court quadrant distribution analytics."""

    data: SingleDistribution
    quadrant_boundaries: Dict[str, float] = Field(
        description="Quadrant boundaries (x_cut, y_cut) in meters"
    )


class WallHitHeatmapResponse(AnalyticsResponseBase):
    """Wall hit position heatmap data."""

    data: SpatialData


class WinningStatsData(BaseModel):
    """Winning statistics aggregate data."""

    efficiency: float = Field(
        description="Points won per shot ratio (main efficiency metric)"
    )
    points_won: int = Field(description="Total points won")
    total_shots: int = Field(description="Total shots taken")
    points_per_rally: float = Field(description="Average points won per rally")
    rallies_played: int = Field(description="Total rallies played")


class WinningStatsResponse(AnalyticsResponseBase):
    """Winning statistics analytics (aggregate only)."""

    data: WinningStatsData


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
    avg_distance_to_ball: float = Field(
        description="Average distance moved to reach ball per shot in meters"
    )
    min_distance_to_ball: Optional[float] = Field(
        description="Minimum distance to ball in meters"
    )
    max_distance_to_ball: Optional[float] = Field(
        description="Maximum distance to ball in meters"
    )
    shot_count: int = Field(description="Number of shots taken")


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


# ============================================================================
# PATTERN 6: TIME-SERIES PATTERN
# For: Rally timeline, Momentum timeline (per-rally granularity)
# ============================================================================


class RallyTimelineItem(BaseModel):
    """Single rally's timeline data."""

    rally_id: int
    rally_start_time: float = Field(description="Timestamp when rally started (seconds)")
    rally_duration: float = Field(description="Rally length in seconds")
    shot_count: int = Field(description="Total shots in rally")
    point_winner: Optional[int] = Field(None, description="Player who won (1 or 2)")
    avg_ball_speed: Optional[float] = Field(None, description="Average ball speed in m/s")
    ball_speed_variance: Optional[float] = Field(None, description="Ball speed variance")
    wall_hit_count: int = Field(description="Number of wall hits in rally")


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


class TimeToTTimelineItem(BaseModel):
    """Per-rally time-to-T metrics."""

    rally_id: int
    rally_start_time: float = Field(description="Timestamp when rally started (seconds)")
    player_1_avg_time_to_t: Optional[float] = Field(
        None, description="Player 1 average time to return to T (seconds)"
    )
    player_1_min_time_to_t: Optional[float] = Field(
        None, description="Player 1 fastest time to return to T (seconds)"
    )
    player_1_max_time_to_t: Optional[float] = Field(
        None, description="Player 1 slowest time to return to T (seconds)"
    )
    player_1_measurements: int = Field(
        description="Number of time-to-T measurements for player 1"
    )
    player_2_avg_time_to_t: Optional[float] = Field(
        None, description="Player 2 average time to return to T (seconds)"
    )
    player_2_min_time_to_t: Optional[float] = Field(
        None, description="Player 2 fastest time to return to T (seconds)"
    )
    player_2_max_time_to_t: Optional[float] = Field(
        None, description="Player 2 slowest time to return to T (seconds)"
    )
    player_2_measurements: int = Field(
        description="Number of time-to-T measurements for player 2"
    )


class TimeToTTimelineResponse(AnalyticsResponseBase):
    """Time-to-T timeline response with metadata."""

    data: List[TimeToTTimelineItem]
    total_rallies: int = Field(description="Total number of rallies returned")
