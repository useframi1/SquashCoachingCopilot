"""Analysis API endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.models.database import get_db
from backend.schemas.analysis import (
    FrameDataListResponse,
    FrameDataResponse,
    RallySummary,
    RallyDetailResponse,
    MatchSummaryResponse,
    ShotAnalysisResponse,
    HeatmapDataResponse,
    PlayerStatsResponse,
)
from backend.services.analysis_service import AnalysisService

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


def get_analysis_service(db: Session = Depends(get_db)) -> AnalysisService:
    return AnalysisService(db)


@router.get("/{video_id}/summary", response_model=MatchSummaryResponse)
async def get_match_summary(
    video_id: str,
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get overall match statistics."""
    try:
        return service.get_match_summary(video_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{video_id}/rallies", response_model=list[RallySummary])
async def get_rallies(
    video_id: str,
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get summary of all rallies in a video."""
    try:
        return service.get_rallies(video_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{video_id}/rallies/{rally_id}", response_model=RallyDetailResponse)
async def get_rally_detail(
    video_id: str,
    rally_id: int,
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get detailed analysis of a specific rally."""
    try:
        return service.get_rally_detail(video_id, rally_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{video_id}/frames", response_model=FrameDataListResponse)
async def get_frames(
    video_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    rally_id: Optional[int] = None,
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get frame-by-frame data with pagination."""
    try:
        frames, total = service.get_frames(
            video_id, page=page, page_size=page_size, rally_id=rally_id
        )
        return FrameDataListResponse(
            frames=frames,
            total=total,
            page=page,
            page_size=page_size,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{video_id}/shots", response_model=ShotAnalysisResponse)
async def get_shots(
    video_id: str,
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get all shots in a video with analysis."""
    try:
        return service.get_shots(video_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{video_id}/heatmap/{player_id}", response_model=HeatmapDataResponse)
async def get_heatmap(
    video_id: str,
    player_id: int,
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get player position heatmap data."""
    if player_id not in [1, 2]:
        raise HTTPException(status_code=400, detail="player_id must be 1 or 2")
    try:
        return service.get_heatmap(video_id, player_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{video_id}/player/{player_id}/stats", response_model=PlayerStatsResponse)
async def get_player_stats(
    video_id: str,
    player_id: int,
    service: AnalysisService = Depends(get_analysis_service),
):
    """Get per-player statistics."""
    if player_id not in [1, 2]:
        raise HTTPException(status_code=400, detail="player_id must be 1 or 2")
    try:
        return service.get_player_stats(video_id, player_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
