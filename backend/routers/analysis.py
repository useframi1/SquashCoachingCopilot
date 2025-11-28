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
    BallSpeedResponse,
    RhythmDisruptionResponse,
    PlayerPositionHeatmapResponse,
    CourtQuadrantResponse,
    WallHitHeatmapResponse,
    WallQuadrantResponse,
    WinningEfficiencyResponse,
    # Extended analytics schemas
    MovementMetricsResponse,
    TZoneOccupancyResponse,
    ShotEffectivenessResponse,
    RallyIntensityResponse,
    # Time-series analytics schemas
    RallyTimelineResponse,
    MomentumTimelineResponse,
    # Per-game time-series schemas
    StrokeDistributionPerGameResponse,
    ShotTypeDistributionPerGameResponse,
    BallSpeedPerGameResponse,
    RhythmDisruptionPerGameResponse,
    CourtQuadrantPerGameResponse,
    WallQuadrantPerGameResponse,
    MovementMetricsPerGameResponse,
    TZoneOccupancyPerGameResponse,
    ShotEffectivenessPerGameResponse,
    WinningEfficiencyPerGameResponse,
    RallyIntensityPerGameResponse,
    # Per-rally time-series schemas
    StrokeDistributionPerRallyResponse,
    ShotTypeDistributionPerRallyResponse,
    BallSpeedPerRallyResponse,
    RhythmDisruptionPerRallyResponse,
    CourtQuadrantPerRallyResponse,
    WallQuadrantPerRallyResponse,
    MovementMetricsPerRallyResponse,
    TZoneOccupancyPerRallyResponse,
    ShotEffectivenessPerRallyResponse,
    WinningEfficiencyPerRallyResponse,
    RallyIntensityPerRallyResponse,
    # Let and break time schemas
    LetStatsResponse,
    BreakTimeResponse,
    # Match highlights schemas
    LongestRallyResponse,
    FastestShotResponse,
)
from backend.schemas.match import MatchSummaryResponse
from backend.services.analysis_service import AnalysisService

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


def get_analysis_service(db: Session = Depends(get_db)) -> AnalysisService:
    return AnalysisService(db)


# ============================================================================
# ANALYTICS ENDPOINTS
# ============================================================================


@router.get(
    "/{video_id}/analytics/stroke-distribution",
    response_model=StrokeDistributionResponse,
)
async def get_stroke_distribution(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Get stroke distribution analytics (forehand vs backhand).

    Returns counts and percentages of forehand and backhand shots for each player,
    with optional filtering by game, player, or time range.
    """
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_stroke_distribution(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/shot-types-distribution",
    response_model=ShotTypeDistributionResponse,
)
async def get_shot_type_distribution(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
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
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_shot_type_distribution(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{video_id}/analytics/ball-speed", response_model=BallSpeedResponse)
async def get_ball_speed_analytics(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Get ball speed aggregate statistics (no time series).

    Calculates ball speed from racket hit to wall hit for each shot.
    Returns aggregate stats (mean, min, max, std_dev, count) for both players.
    """
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_ball_speed_analytics(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/rhythm-disruption", response_model=RhythmDisruptionResponse
)
async def get_rhythm_disruption(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
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
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_rhythm_disruption(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/player-heatmap",
    response_model=PlayerPositionHeatmapResponse,
)
async def get_player_position_heatmap(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Get player position data for heatmap visualization.

    Returns aggregated position points if player_id is not specified,
    otherwise returns data for the specified player only.
    Use this data to generate heatmaps showing court coverage and positioning patterns.
    """
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_player_position_heatmap(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/court-quadrants", response_model=CourtQuadrantResponse
)
async def get_court_quadrant_distribution(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
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
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_court_quadrant_distribution(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/wall-hits-heatmap",
    response_model=WallHitHeatmapResponse,
)
async def get_wall_hit_heatmap(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Get wall hit position data for heatmap visualization.

    Returns positions where the ball hit the wall. Use this to visualize
    shot placement patterns and targeting strategies on a wall heatmap.
    """
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_wall_hit_heatmap(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{video_id}/analytics/wall-quadrants", response_model=WallQuadrantResponse)
async def get_wall_quadrant_distribution(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Get wall quadrant distribution analytics.

    Analyzes where the ball hits the front wall across four quadrants
    (Top-Left, Top-Right, Bottom-Left, Bottom-Right). Shows shot placement
    patterns and targeting strategies on the front wall.
    """
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_wall_quadrant_distribution(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# NOTE: Per-game and per-rally endpoints must come BEFORE /{player_id} endpoints
# to avoid route matching conflicts (FastAPI matches routes in order)


@router.get(
    "/{video_id}/analytics/winning-efficiency/per-game",
    response_model=WinningEfficiencyPerGameResponse,
)
async def get_winning_efficiency_per_game(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get winning efficiency metrics per game with both players' data."""
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_winning_efficiency_per_game(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/winning-efficiency/per-rally",
    response_model=WinningEfficiencyPerRallyResponse,
)
async def get_winning_efficiency_per_rally(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get winning efficiency metrics per rally with both players' data."""
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_winning_efficiency_per_rally(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/winning-efficiency/{player_id}",
    response_model=WinningEfficiencyResponse,
)
async def get_winning_efficiency(
    video_id: str,
    player_id: int,
    game_number: Optional[int] = Query(None, ge=1),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Get winning efficiency metrics for a specific player.

    Calculates how many shots the player needed to make to win each point.
    Lower values indicate better efficiency (winning points with fewer shots).
    Shows shot efficiency and point-winning effectiveness.
    """
    if player_id not in [1, 2]:
        raise HTTPException(status_code=400, detail="player_id must be 1 or 2")
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_winning_efficiency(video_id, player_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/movement-metrics", response_model=MovementMetricsResponse
)
async def get_movement_metrics(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Get movement and distance metrics.

    Analyzes total distance covered, distance per rally, and distance moved to reach
    the ball for each shot. Provides comprehensive movement analytics for both players.
    """
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_movement_metrics(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/t-zone-occupancy", response_model=TZoneOccupancyResponse
)
async def get_t_zone_occupancy(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Get T-zone occupancy and positioning analytics.

    Analyzes percentage of time spent in T-zone, time taken to reach T-zone after
    opponent shots, and success rate of reaching T-zone. Key metrics for court
    positioning and recovery analysis.
    """
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_t_zone_occupancy(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# NOTE: Per-game and per-rally endpoints must come BEFORE /{player_id} endpoints
# to avoid route matching conflicts (FastAPI matches routes in order)


@router.get(
    "/{video_id}/analytics/shot-effectiveness/per-game",
    response_model=ShotEffectivenessPerGameResponse,
)
async def get_shot_effectiveness_per_game(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get shot effectiveness metrics per game with both players' data."""
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_shot_effectiveness_per_game(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/shot-effectiveness/per-rally",
    response_model=ShotEffectivenessPerRallyResponse,
)
async def get_shot_effectiveness_per_rally(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get shot effectiveness metrics per rally with both players' data."""
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_shot_effectiveness_per_rally(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/shot-effectiveness/{player_id}",
    response_model=ShotEffectivenessResponse,
)
async def get_shot_effectiveness(
    video_id: str,
    player_id: int,
    game_number: Optional[int] = Query(None, ge=1),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Get shot effectiveness and placement quality metrics for a specific player.

    Combines displacement from T-zone (how far opponent moves after shot),
    depth dominance (percentage of shots where opponent is deeper), and
    straight shot quality (percentage of straight shots hit close to wall).
    Comprehensive offensive performance analytics.
    """
    if player_id not in [1, 2]:
        raise HTTPException(status_code=400, detail="player_id must be 1 or 2")
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_shot_effectiveness(video_id, player_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/rally-intensity", response_model=RallyIntensityResponse
)
async def get_rally_intensity(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Get rally intensity and pace metrics.

    Analyzes seconds per shot for each rally (lower = faster/more intense).
    Returns per-rally breakdowns plus aggregate statistics (average, min, max).
    Useful for understanding match tempo and rally patterns.
    """
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_rally_intensity(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================================
# TIME-SERIES ANALYTICS ENDPOINTS
# ============================================================================


@router.get(
    "/{video_id}/analytics/rally-timeline",
    response_model=RallyTimelineResponse,
)
async def get_rally_timeline(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Get rally-by-rally timeline with key metrics.

    Returns chronological sequence of rallies with:
    - Rally duration and shot count
    - Average ball speed and variance (rhythm)
    - Point winner for each rally
    - Wall hit count

    Useful for:
    - Line charts of rally duration over time
    - Bar charts of shots per rally
    - Ball speed trend visualization
    """
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_rally_timeline(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/momentum-timeline",
    response_model=MomentumTimelineResponse,
)
async def get_momentum_timeline(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Get cumulative score progression and momentum shifts.

    Returns running scoreboard after each rally with:
    - Cumulative scores for both players
    - Point winner for each rally
    - Score differential (lead changes)

    Useful for:
    - Dual-line chart showing score progression
    - Area chart showing lead changes
    - Momentum shift identification
    """
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_momentum_timeline(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================================
# PER-GAME TIME-SERIES ENDPOINTS
# ============================================================================


@router.get(
    "/{video_id}/analytics/stroke-distribution/per-game",
    response_model=StrokeDistributionPerGameResponse,
)
async def get_stroke_distribution_per_game(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get stroke distribution per game with both players' data."""
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_stroke_distribution_per_game(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/shot-type-distribution/per-game",
    response_model=ShotTypeDistributionPerGameResponse,
)
async def get_shot_type_distribution_per_game(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get shot type distribution per game with both players' data."""
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_shot_type_distribution_per_game(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/rhythm-disruption/per-game",
    response_model=RhythmDisruptionPerGameResponse,
)
async def get_rhythm_disruption_per_game(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get rhythm disruption statistics per game with both players' data."""
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_rhythm_disruption_per_game(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/court-quadrants/per-game",
    response_model=CourtQuadrantPerGameResponse,
)
async def get_court_quadrant_per_game(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get court quadrant distribution per game with both players' data."""
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_court_quadrant_per_game(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/wall-quadrants/per-game",
    response_model=WallQuadrantPerGameResponse,
)
async def get_wall_quadrant_per_game(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get wall quadrant distribution per game with both players' data."""
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_wall_quadrant_per_game(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/movement-metrics/per-game",
    response_model=MovementMetricsPerGameResponse,
)
async def get_movement_metrics_per_game(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get movement metrics per game with both players' data."""
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_movement_metrics_per_game(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/t-zone-occupancy/per-game",
    response_model=TZoneOccupancyPerGameResponse,
)
async def get_t_zone_occupancy_per_game(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get T-zone occupancy per game with both players' data."""
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_t_zone_occupancy_per_game(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/ball-speed/per-game",
    response_model=BallSpeedPerGameResponse,
)
async def get_ball_speed_per_game(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get ball speed statistics per game with both players' data."""
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_ball_speed_per_game(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/rally-intensity/per-game",
    response_model=RallyIntensityPerGameResponse,
)
async def get_rally_intensity_per_game(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get rally intensity metrics per game (not player-specific)."""
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_rally_intensity_per_game(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================================
# PER-RALLY TIME-SERIES ENDPOINTS
# ============================================================================


@router.get(
    "/{video_id}/analytics/stroke-distribution/per-rally",
    response_model=StrokeDistributionPerRallyResponse,
)
async def get_stroke_distribution_per_rally(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get stroke distribution per rally with both players' data."""
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_stroke_distribution_per_rally(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/shot-type-distribution/per-rally",
    response_model=ShotTypeDistributionPerRallyResponse,
)
async def get_shot_type_distribution_per_rally(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get shot type distribution per rally with both players' data."""
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_shot_type_distribution_per_rally(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/rhythm-disruption/per-rally",
    response_model=RhythmDisruptionPerRallyResponse,
)
async def get_rhythm_disruption_per_rally(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get rhythm disruption statistics per rally with both players' data."""
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_rhythm_disruption_per_rally(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/court-quadrants/per-rally",
    response_model=CourtQuadrantPerRallyResponse,
)
async def get_court_quadrant_per_rally(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get court quadrant distribution per rally with both players' data."""
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_court_quadrant_per_rally(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/wall-quadrants/per-rally",
    response_model=WallQuadrantPerRallyResponse,
)
async def get_wall_quadrant_per_rally(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get wall quadrant distribution per rally with both players' data."""
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_wall_quadrant_per_rally(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/movement-metrics/per-rally",
    response_model=MovementMetricsPerRallyResponse,
)
async def get_movement_metrics_per_rally(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get movement metrics per rally with both players' data."""
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_movement_metrics_per_rally(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/t-zone-occupancy/per-rally",
    response_model=TZoneOccupancyPerRallyResponse,
)
async def get_t_zone_occupancy_per_rally(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get T-zone occupancy per rally with both players' data."""
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_t_zone_occupancy_per_rally(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/ball-speed/per-rally",
    response_model=BallSpeedPerRallyResponse,
)
async def get_ball_speed_per_rally(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get ball speed statistics per rally with both players' data."""
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_ball_speed_per_rally(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{video_id}/analytics/rally-intensity/per-rally",
    response_model=RallyIntensityPerRallyResponse,
)
async def get_rally_intensity_per_rally(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get rally intensity metrics per rally (not player-specific)."""
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_rally_intensity_per_rally(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================================
# MATCH SUMMARY ENDPOINT
# ============================================================================


@router.get(
    "/{video_id}/match-summary",
    response_model=MatchSummaryResponse,
)
async def get_match_summary(
    video_id: str,
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Get match summary including game results and overall match winner.

    Returns complete match information following squash scoring rules:
    - Individual game scores
    - Match winner and games won by each player
    - Rally counts and scoring system used
    """
    try:
        return service.get_match_summary(video_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{video_id}/analytics/let-stats", response_model=LetStatsResponse)
async def get_let_stats(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Get let/replay statistics.

    Counts rallies where point_winner = 0 (indicates a let/replay).
    Returns total number of lets, total rallies, and let percentage.
    """
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_let_stats(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{video_id}/analytics/break-time", response_model=BreakTimeResponse)
async def get_break_time(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Get break time statistics between rallies.

    Calculates the time between the end of one rally and the start of the next rally.
    Returns average, minimum, maximum, standard deviation, and total number of breaks.
    """
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_break_time(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================================
# MATCH HIGHLIGHTS ENDPOINTS
# ============================================================================


@router.get("/{video_id}/analytics/longest-rally", response_model=LongestRallyResponse)
async def get_longest_rally(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Get the longest rally in the match.

    Returns the rally with the longest duration (in seconds) and number of shots.
    Includes rally metadata like start time, duration, shot count, and point winner.
    """
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_longest_rally(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{video_id}/analytics/fastest-shot", response_model=FastestShotResponse)
async def get_fastest_shot(
    video_id: str,
    game_number: Optional[int] = Query(None, ge=1),
    player_id: Optional[int] = Query(None, ge=1, le=2),
    start_time: Optional[float] = Query(None, ge=0),
    end_time: Optional[float] = Query(None, ge=0),
    service: AnalysisService = Depends(get_analysis_service),
):
    """
    Get the fastest shot in the match.

    Returns the shot with the highest ball speed (m/s).
    Includes frame number, timestamp, player who hit it, stroke type, and shot type.
    """
    try:
        filters = AnalyticsFilters(
            game_number=game_number,
            player_id=player_id,
            start_time=start_time,
            end_time=end_time,
        )
        return service.get_fastest_shot(video_id, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
