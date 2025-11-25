"""Pydantic schemas for analysis-related API operations - Redesigned for consistency."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# BASE SCHEMAS
# ============================================================================


class AnalyticsFilters(BaseModel):
    """Common filters for analytics queries."""

    rally_id: Optional[int] = Field(None, description="Filter by specific rally")
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


class PlayerDistribution(BaseModel):
    """Distribution data for a single player."""

    player_id: int
    distribution: List[DistributionItem] = Field(
        description="Array of items (chart-ready)"
    )
    total: int = Field(description="Total count across all items")


class DistributionData(BaseModel):
    """Distribution data for both players (always both present)."""

    player_1: PlayerDistribution
    player_2: PlayerDistribution


# ============================================================================
# PATTERN 2: AGGREGATE PATTERN
# For: Ball speed, other summary statistics
# ============================================================================


class PlayerAggregates(BaseModel):
    """Aggregate statistics for a single player."""

    player_id: int
    mean: float = Field(description="Mean value")
    min: float = Field(description="Minimum value")
    max: float = Field(description="Maximum value")
    std_dev: float = Field(description="Standard deviation")
    count: int = Field(description="Number of data points")


class AggregateData(BaseModel):
    """Aggregate data for both players (always both present)."""

    player_1: PlayerAggregates
    player_2: PlayerAggregates


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
    """Spatial heatmap data for a single player."""

    player_id: int
    heatmap_grid: HeatmapGrid


# ============================================================================
# PATTERN 4: COMPARATIVE PATTERN
# For: Rally stats, Winning stats, Rhythm disruption
# ============================================================================


class RallyItem(BaseModel):
    """Single rally with metrics."""

    rally_id: int
    duration_seconds: float
    stroke_count: int
    player_1_shots: int
    player_2_shots: int
    start_timestamp: float
    end_timestamp: float


class RallyStatsData(BaseModel):
    """Rally statistics data with items and summaries."""

    items: List[RallyItem] = Field(description="List of individual rally metrics")
    summary: Dict[str, Any] = Field(
        description="Overall summary (total_rallies, avg_duration, avg_stroke_count)"
    )
    player_1_summary: Dict[str, Any] = Field(
        description="Player 1 summary (total_shots, avg_shots_per_rally)"
    )
    player_2_summary: Dict[str, Any] = Field(
        description="Player 2 summary (total_shots, avg_shots_per_rally)"
    )


class RhythmRallyItem(BaseModel):
    """Single rally with rhythm disruption metrics."""

    rally_id: int
    player_id: int
    ball_speed_variance: Optional[float] = None
    ball_speed_cv: Optional[float] = None
    wall_hit_height_variance: Optional[float] = None
    wall_hit_height_cv: Optional[float] = None
    shot_count: int


class RhythmDisruptionData(BaseModel):
    """Rhythm disruption data with rally-level metrics."""

    items: List[RhythmRallyItem] = Field(description="Per-rally rhythm metrics")
    player_1_summary: Dict[str, Optional[float]] = Field(
        description="Player 1 averages (avg_ball_speed_cv, avg_height_cv)"
    )
    player_2_summary: Dict[str, Optional[float]] = Field(
        description="Player 2 averages (avg_ball_speed_cv, avg_height_cv)"
    )


class WinningRallyItem(BaseModel):
    """Single rally with winning statistics."""

    rally_id: int
    player_id: int
    total_shots: int
    points_won: int
    points_per_shot: float


class WinningStatsData(BaseModel):
    """Winning statistics data."""

    items: List[WinningRallyItem] = Field(description="Per-rally winning metrics")
    player_1_summary: Dict[str, Any] = Field(
        description="Player 1 totals (total_points, total_shots, efficiency)"
    )
    player_2_summary: Dict[str, Any] = Field(
        description="Player 2 totals (total_points, total_shots, efficiency)"
    )


class ShotPlacementItem(BaseModel):
    """Single shot placement detail."""

    frame_number: int
    timestamp: float
    player_id: int
    player_x: float
    player_y: float
    opponent_x_before: float
    opponent_y_before: float
    opponent_x_after: Optional[float] = None
    opponent_y_after: Optional[float] = None
    distance_moved: Optional[float] = None


class ShotPlacementData(BaseModel):
    """Shot placement effectiveness data."""

    player_id: int
    items: List[ShotPlacementItem] = Field(description="Individual shot placements")
    summary: Dict[str, Optional[float]] = Field(
        description="Summary statistics (avg_distance_moved, max_distance_moved)"
    )


# ============================================================================
# ENDPOINT-SPECIFIC RESPONSE SCHEMAS
# ============================================================================


class StrokeDistributionResponse(AnalyticsResponseBase):
    """Stroke distribution analytics (forehand vs backhand)."""

    data: DistributionData


class ShotTypeDistributionResponse(AnalyticsResponseBase):
    """Shot type distribution analytics."""

    data: DistributionData
    all_shot_types: List[str] = Field(
        description="List of all shot types found (for consistent chart colors)"
    )


class BallSpeedResponse(AnalyticsResponseBase):
    """Ball speed aggregate statistics (no time series)."""

    data: AggregateData


class RhythmDisruptionResponse(AnalyticsResponseBase):
    """Rhythm disruption analytics."""

    data: RhythmDisruptionData


class PlayerPositionHeatmapResponse(AnalyticsResponseBase):
    """Player position heatmap data."""

    data: SpatialData


class ShotPlacementResponse(AnalyticsResponseBase):
    """Shot placement effectiveness analytics."""

    data: ShotPlacementData


class CourtQuadrantResponse(AnalyticsResponseBase):
    """Court quadrant distribution analytics."""

    data: DistributionData
    quadrant_boundaries: Dict[str, float] = Field(
        description="Quadrant boundaries (x_cut, y_cut) in meters"
    )


class RallyStatsResponse(AnalyticsResponseBase):
    """Rally statistics analytics."""

    data: RallyStatsData


class WallHitHeatmapResponse(AnalyticsResponseBase):
    """Wall hit position heatmap data."""

    data: SpatialData


class WinningStatsResponse(AnalyticsResponseBase):
    """Winning statistics analytics."""

    data: WinningStatsData


class WallQuadrantResponse(AnalyticsResponseBase):
    """Wall quadrant distribution analytics."""

    data: DistributionData
    quadrant_boundaries: Dict[str, float] = Field(
        description="Quadrant boundaries (x_cut, y_cut) in meters"
    )
