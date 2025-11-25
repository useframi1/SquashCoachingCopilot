"""Pydantic schemas for analysis-related API operations."""

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# ANALYTICS SCHEMAS
# ============================================================================


class AnalyticsFilters(BaseModel):
    """Common filters for analytics queries."""

    rally_id: Optional[int] = Field(None, description="Filter by specific rally")
    player_id: Optional[int] = Field(None, ge=1, le=2, description="Filter by player (1 or 2)")
    start_time: Optional[float] = Field(None, ge=0, description="Start timestamp in seconds")
    end_time: Optional[float] = Field(None, ge=0, description="End timestamp in seconds")

    @field_validator("end_time")
    @classmethod
    def end_after_start(cls, v: Optional[float], info) -> Optional[float]:
        """Validate that end_time is after start_time."""
        if v is not None and info.data.get("start_time") is not None:
            if v <= info.data["start_time"]:
                raise ValueError("end_time must be after start_time")
        return v


class AnalyticsResponseBase(BaseModel):
    """Base schema for all analytics responses."""

    video_id: str
    filters_applied: Optional[AnalyticsFilters] = None
    computed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data_points: int = Field(description="Number of data points used in calculation")


class PlayerStrokeStats(BaseModel):
    """Stroke statistics for a single player."""

    player_id: int
    forehand_count: int = 0
    backhand_count: int = 0
    total_shots: int = 0

    @property
    def forehand_percentage(self) -> float:
        """Calculate forehand percentage."""
        return (self.forehand_count / self.total_shots * 100) if self.total_shots > 0 else 0.0

    @property
    def backhand_percentage(self) -> float:
        """Calculate backhand percentage."""
        return (self.backhand_count / self.total_shots * 100) if self.total_shots > 0 else 0.0


class StrokeDistributionResponse(AnalyticsResponseBase):
    """Stroke distribution analytics (forehand vs backhand)."""

    player_1: Optional[PlayerStrokeStats] = None
    player_2: Optional[PlayerStrokeStats] = None


class ShotTypeStats(BaseModel):
    """Shot type statistics for a single player."""

    player_id: int
    shot_counts: dict[str, int] = Field(
        default_factory=dict, description="Shot type to count mapping"
    )
    total_shots: int = 0


class ShotTypeDistributionResponse(AnalyticsResponseBase):
    """Shot type distribution analytics."""

    player_1: Optional[ShotTypeStats] = None
    player_2: Optional[ShotTypeStats] = None
    all_shot_types: List[str] = Field(description="List of all shot types found")


class BallSpeedDataPoint(BaseModel):
    """Single data point for ball speed time series."""

    timestamp: float
    player_id: int
    speed: float
    frame_number: int


class BallSpeedAnalyticsResponse(AnalyticsResponseBase):
    """Ball speed analytics with time series."""

    time_series: List[BallSpeedDataPoint]
    player_1_avg_speed: Optional[float] = None
    player_1_max_speed: Optional[float] = None
    player_1_min_speed: Optional[float] = None
    player_2_avg_speed: Optional[float] = None
    player_2_max_speed: Optional[float] = None
    player_2_min_speed: Optional[float] = None


class RallyRhythmStats(BaseModel):
    """Rhythm disruption metrics for a single rally."""

    rally_id: int
    player_id: int
    ball_speed_variance: Optional[float] = None
    ball_speed_cv: Optional[float] = None
    wall_hit_height_variance: Optional[float] = None
    wall_hit_height_cv: Optional[float] = None
    shot_count: int = 0


class RhythmDisruptionResponse(AnalyticsResponseBase):
    """Rhythm disruption analytics (variance and CV)."""

    rallies: List[RallyRhythmStats]
    player_1_avg_ball_speed_cv: Optional[float] = None
    player_2_avg_ball_speed_cv: Optional[float] = None
    player_1_avg_height_cv: Optional[float] = None
    player_2_avg_height_cv: Optional[float] = None


class PositionPoint(BaseModel):
    """Player position point for heatmap."""

    x: float
    y: float
    timestamp: float
    frame_number: int


class PlayerPositionHeatmapResponse(AnalyticsResponseBase):
    """Player position heatmap data."""

    player_id: int
    positions: List[PositionPoint]
    court_bounds: dict[str, float] = Field(
        description="Court boundaries for visualization (x_min, x_max, y_min, y_max)"
    )


class ShotPlacementDetail(BaseModel):
    """Shot placement effectiveness detail."""

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


class ShotPlacementResponse(AnalyticsResponseBase):
    """Shot placement effectiveness analytics."""

    player_id: int
    placements: List[ShotPlacementDetail]
    avg_distance_moved: Optional[float] = None
    max_distance_moved: Optional[float] = None


class QuadrantStats(BaseModel):
    """Statistics for a single court quadrant."""

    quadrant: str
    frame_count: int
    percentage: float
    avg_time_seconds: float


class CourtQuadrantResponse(AnalyticsResponseBase):
    """Court quadrant distribution analytics."""

    player_1_quadrants: List[QuadrantStats]
    player_2_quadrants: List[QuadrantStats]
    quadrant_boundaries: dict[str, float] = Field(
        description="Quadrant boundaries (x_cut, y_cut) in meters"
    )


class RallyStatsDetail(BaseModel):
    """Statistics for a single rally."""

    rally_id: int
    duration_seconds: float
    stroke_count: int
    player_1_shots: int
    player_2_shots: int
    start_timestamp: float
    end_timestamp: float


class RallyStatsResponse(AnalyticsResponseBase):
    """Rally statistics analytics."""

    rallies: List[RallyStatsDetail]
    avg_rally_duration: float
    avg_stroke_count: float
    total_rallies: int


class WallHitPoint(BaseModel):
    """Wall hit position for shot placement heatmap."""

    x: float
    y: float
    timestamp: float
    frame_number: int
    player_id: int


class WallHitDistributionResponse(AnalyticsResponseBase):
    """Wall hit distribution for shot placement heatmaps."""

    wall_hits: List[WallHitPoint]
    player_id: Optional[int] = None
    wall_bounds: dict[str, float] = Field(
        description="Wall boundaries for visualization"
    )


class WinningStatsDetail(BaseModel):
    """Winning statistics for a rally."""

    rally_id: int
    player_id: int
    total_shots: int
    points_won: int
    points_per_shot: float


class WinningStatsResponse(AnalyticsResponseBase):
    """Winning statistics analytics."""

    rally_stats: List[WinningStatsDetail]
    player_1_total_points: int = 0
    player_1_total_shots: int = 0
    player_1_efficiency: float = 0.0
    player_2_total_points: int = 0
    player_2_total_shots: int = 0
    player_2_efficiency: float = 0.0
