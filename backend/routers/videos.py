"""Video management API endpoints."""

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from backend.models.database import get_db
from backend.schemas.video import (
    VideoUploadResponse,
    VideoMetadataResponse,
    VideoListResponse,
    VideoDeleteResponse,
)
from backend.services.video_service import VideoService

router = APIRouter(prefix="/api/videos", tags=["videos"])


def get_video_service(db: Session = Depends(get_db)) -> VideoService:
    return VideoService(db)


@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    service: VideoService = Depends(get_video_service),
):
    """Upload a new video for analysis."""
    video = await service.upload_video(file)
    return VideoUploadResponse(
        id=video.id,
        filename=video.original_filename,
        message="Video uploaded successfully",
    )


@router.get("", response_model=VideoListResponse)
async def list_videos(
    page: int = 1,
    page_size: int = 20,
    service: VideoService = Depends(get_video_service),
):
    """List all uploaded videos."""
    videos, total = service.list_videos(page=page, page_size=page_size)

    return VideoListResponse(
        videos=[
            VideoMetadataResponse(
                id=v.id,
                filename=v.filename,
                original_filename=v.original_filename,
                fps=v.fps,
                total_frames=v.total_frames,
                width=v.width,
                height=v.height,
                duration_seconds=v.duration_seconds,
                file_size_bytes=v.file_size_bytes,
                has_annotated_video=v.annotated_video_path is not None,
                uploaded_at=v.uploaded_at,
                processed_at=v.processed_at,
            )
            for v in videos
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{video_id}", response_model=VideoMetadataResponse)
async def get_video(
    video_id: str,
    service: VideoService = Depends(get_video_service),
):
    """Get video metadata."""
    video = service.get_video(video_id)
    return VideoMetadataResponse(
        id=video.id,
        filename=video.filename,
        original_filename=video.original_filename,
        fps=video.fps,
        total_frames=video.total_frames,
        width=video.width,
        height=video.height,
        duration_seconds=video.duration_seconds,
        file_size_bytes=video.file_size_bytes,
        has_annotated_video=video.annotated_video_path is not None,
        uploaded_at=video.uploaded_at,
        processed_at=video.processed_at,
    )


@router.get("/{video_id}/stream")
async def stream_video(
    video_id: str,
    service: VideoService = Depends(get_video_service),
):
    """Stream the original video file."""
    video_path = service.get_video_path(video_id)
    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=video_path.name,
    )


@router.get("/{video_id}/annotated")
async def stream_annotated_video(
    video_id: str,
    service: VideoService = Depends(get_video_service),
):
    """Stream the annotated video file."""
    video_path = service.get_annotated_video_path(video_id)
    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=f"annotated_{video_path.name}",
    )


@router.delete("/{video_id}", response_model=VideoDeleteResponse)
async def delete_video(
    video_id: str,
    service: VideoService = Depends(get_video_service),
):
    """Delete a video and all associated data."""
    service.delete_video(video_id)
    return VideoDeleteResponse(id=video_id, message="Video deleted successfully")
