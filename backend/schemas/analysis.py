"""Pydantic schemas for analysis-related API operations."""

from typing import Optional

from pydantic import BaseModel, Field


class FrameDataResponse(BaseModel):
    """Single frame analysis data."""

    frame_number: int
    timestamp: float
    ball_x: Optional[float] = None
    ball_y: Optional[float] = None
    player_1_x_meter: Optional[float] = None
    player_1_y_meter: Optional[float] = None
    player_2_x_meter: Optional[float] = None
    player_2_y_meter: Optional[float] = None
    is_rally_frame: bool = False
    rally_id: Optional[int] = None
    is_wall_hit: bool = False
    is_racket_hit: bool = False
    stroke_type: Optional[str] = None
    shot_type: Optional[str] = None
    shot_direction: Optional[str] = None
    shot_depth: Optional[str] = None

    class Config:
        from_attributes = True


class FrameDataListResponse(BaseModel):
    """Paginated frame data response."""

    frames: list[FrameDataResponse]
    total: int
    page: int = 1
    page_size: int = 100


class RallySummary(BaseModel):
    """Summary of a single rally."""

    rally_id: int
    start_frame: int
    end_frame: int
    start_timestamp: float
    end_timestamp: float
    duration_seconds: float
    total_shots: int
    wall_hits: int
    player_1_shots: int
    player_2_shots: int


class RallyDetailResponse(BaseModel):
    """Detailed rally analysis."""

    rally_id: int
    start_frame: int
    end_frame: int
    duration_seconds: float
    shots: list["ShotDetail"]
    player_1_stats: "RallyPlayerStats"
    player_2_stats: "RallyPlayerStats"


class ShotDetail(BaseModel):
    """Details of a single shot."""

    frame_number: int
    timestamp: float
    player_id: int
    stroke_type: Optional[str] = None
    shot_type: Optional[str] = None
    shot_direction: Optional[str] = None
    shot_depth: Optional[str] = None


class RallyPlayerStats(BaseModel):
    """Per-player stats within a rally."""

    total_shots: int
    forehand_count: int
    backhand_count: int
    shot_types: dict[str, int]  # e.g., {"straight_drive": 3, "cross_court_drop": 1}


class MatchSummaryResponse(BaseModel):
    """Overall match statistics."""

    video_id: str
    total_frames: int
    duration_seconds: float
    total_rallies: int
    total_shots: int
    total_wall_hits: int
    avg_rally_duration: float
    longest_rally_id: Optional[int] = None
    longest_rally_shots: int = 0
    player_1_total_shots: int
    player_2_total_shots: int
    stroke_distribution: dict[str, int]  # forehand/backhand counts
    shot_type_distribution: dict[str, int]  # by shot type


class ShotAnalysisResponse(BaseModel):
    """All shots in a video."""

    shots: list[ShotDetail]
    total: int
    stroke_distribution: dict[str, int]
    shot_type_distribution: dict[str, int]
    direction_distribution: dict[str, int]
    depth_distribution: dict[str, int]


class HeatmapPoint(BaseModel):
    """Single point for heatmap visualization."""

    x: float
    y: float
    count: int = 1


class HeatmapDataResponse(BaseModel):
    """Player position heatmap data."""

    video_id: str
    player_id: int
    points: list[HeatmapPoint]
    bounds: dict[str, float]  # min_x, max_x, min_y, max_y


class PlayerStatsResponse(BaseModel):
    """Per-player statistics."""

    video_id: str
    player_id: int
    total_shots: int
    forehand_count: int
    backhand_count: int
    forehand_percentage: float
    backhand_percentage: float
    shot_types: dict[str, int]
    avg_position_x: float
    avg_position_y: float
    court_coverage_area: float  # approximate area covered


# Update forward references
RallyDetailResponse.model_rebuild()
