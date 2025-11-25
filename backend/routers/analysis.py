"""Analysis API endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.models.database import get_db
from backend.schemas.analysis import (
    # Analytics schemas
    AnalyticsFilters,
    StrokeDistributionResponse,
    ShotTypeDistributionResponse,
    BallSpeedAnalyticsResponse,
    RhythmDisruptionResponse,
    PlayerPositionHeatmapResponse,
    ShotPlacementResponse,
    CourtQuadrantResponse,
    RallyStatsResponse,
    WallHitDistributionResponse,
    WinningStatsResponse,
)
from backend.services.analysis_service import AnalysisService

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


def get_analysis_service(db: Session = Depends(get_db)) -> AnalysisService:
    return AnalysisService(db)


# ============================================================================
# ANALYTICS ENDPOINTS
# ============================================================================


@router.get("/{video_id}/analytics/stroke-distribution", response_model=StrokeDistributionResponse)
async def get_stroke_distribution(
    video_id: str,
    rally_id: Optional[int] = None,
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Get stroke distribution analytics (forehand vs backhand).

    Returns counts and percentages of forehand and backhand shots for each player,
    with optional filtering by rally, player, or time range.
    """
    try:
        filters = AnalyticsFilters(
            rally_id=rally_id,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_stroke_distribution(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{video_id}/analytics/shot-types", response_model=ShotTypeDistributionResponse)
async def get_shot_type_distribution(
    video_id: str,
    rally_id: Optional[int] = None,
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Get shot type distribution analytics.

    Returns breakdown of shot types (drives, drops, etc.) for each player,
    useful for pie charts and bar graphs on the dashboard.
    """
    try:
        filters = AnalyticsFilters(
            rally_id=rally_id,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_shot_type_distribution(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{video_id}/analytics/ball-speed", response_model=BallSpeedAnalyticsResponse)
async def get_ball_speed_analytics(
    video_id: str,
    rally_id: Optional[int] = None,
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Get ball speed analytics with time series data.

    Calculates ball speed from racket hit to wall hit for each shot.
    Returns time series data for line charts and aggregate stats (avg, max, min).
    """
    try:
        filters = AnalyticsFilters(
            rally_id=rally_id,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_ball_speed_analytics(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{video_id}/analytics/rhythm-disruption", response_model=RhythmDisruptionResponse)
async def get_rhythm_disruption(
    video_id: str,
    rally_id: Optional[int] = None,
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Get rhythm disruption analytics (variance and coefficient of variation).

    Analyzes shot variability through variance and CV of ball speed and shot height.
    Higher CV indicates more unpredictable, rhythm-disrupting play.
    """
    try:
        filters = AnalyticsFilters(
            rally_id=rally_id,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_rhythm_disruption(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{video_id}/analytics/player-heatmap/{player_id}", response_model=PlayerPositionHeatmapResponse)
async def get_player_position_heatmap(
    video_id: str,
    player_id: int,
    rally_id: Optional[int] = None,
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Get player position data for heatmap visualization.

    Returns all position points for the specified player with timestamps.
    Use this data to generate heatmaps showing court coverage and positioning patterns.
    """
    if player_id not in [1, 2]:
        raise HTTPException(status_code=400, detail="player_id must be 1 or 2")
    try:
        filters = AnalyticsFilters(
            rally_id=rally_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_player_position_heatmap(video_id, player_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{video_id}/analytics/shot-placement/{player_id}", response_model=ShotPlacementResponse)
async def get_shot_placement_effectiveness(
    video_id: str,
    player_id: int,
    rally_id: Optional[int] = None,
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Analyze shot placement effectiveness.

    Tracks opponent position before and after each shot to measure how much
    the opponent had to move. Greater distance indicates more effective shot placement.
    """
    if player_id not in [1, 2]:
        raise HTTPException(status_code=400, detail="player_id must be 1 or 2")
    try:
        filters = AnalyticsFilters(
            rally_id=rally_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_shot_placement_effectiveness(video_id, player_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{video_id}/analytics/court-quadrants", response_model=CourtQuadrantResponse)
async def get_court_quadrant_distribution(
    video_id: str,
    rally_id: Optional[int] = None,
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Get court quadrant distribution analytics.

    Analyzes time spent in each of the four court quadrants (Front-Left, Front-Right,
    Back-Left, Back-Right). Shows positioning preferences and court coverage patterns.
    """
    try:
        filters = AnalyticsFilters(
            rally_id=rally_id,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_court_quadrant_distribution(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{video_id}/analytics/rally-stats", response_model=RallyStatsResponse)
async def get_rally_stats(
    video_id: str,
    rally_id: Optional[int] = None,
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Get rally statistics analytics.

    Returns duration and stroke count for each rally, plus aggregate statistics.
    Useful for analyzing match pace and rally intensity.
    """
    try:
        filters = AnalyticsFilters(
            rally_id=rally_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_rally_stats(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{video_id}/analytics/wall-hits", response_model=WallHitDistributionResponse)
async def get_wall_hit_distribution(
    video_id: str,
    rally_id: Optional[int] = None,
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    quadrant: Optional[str] = Query(None, description="Filter by court quadrant (Front-Left, Front-Right, Back-Left, Back-Right)"),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Get wall hit distribution for shot placement heatmaps.

    Returns positions where the ball hit the wall. Use this to visualize
    shot placement patterns and targeting strategies on a wall heatmap.
    Optionally filter by court quadrant.
    """
    try:
        filters = AnalyticsFilters(
            rally_id=rally_id,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_wall_hit_distribution(video_id, filters, quadrant=quadrant)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{video_id}/analytics/winning-stats", response_model=WinningStatsResponse)
async def get_winning_stats(
    video_id: str,
    rally_id: Optional[int] = None,
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Get winning statistics and efficiency metrics.

    Calculates points won per shot ratio for each player and rally.
    Shows shot efficiency and point-winning effectiveness.
    """
    try:
        filters = AnalyticsFilters(
            rally_id=rally_id,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_winning_stats(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
