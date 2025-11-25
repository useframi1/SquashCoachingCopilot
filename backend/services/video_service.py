"""Video handling service."""

import shutil
import subprocess
import uuid
from pathlib import Path
from typing import BinaryIO

import cv2
from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.config import settings
from backend.models.video import Video
from backend.storage.local import LocalStorage


class VideoService:
    """Service for video upload, storage, and metadata management."""

    def __init__(self, db: Session):
        self.db = db
        self.storage = LocalStorage()

    async def upload_video(self, file: UploadFile) -> Video:
        """Upload and process a new video file."""
        # Validate file extension
        filename = file.filename or "video.mp4"
        extension = Path(filename).suffix.lower()

        if extension not in settings.allowed_video_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {settings.allowed_video_extensions}",
            )

        # Generate unique ID
        video_id = str(uuid.uuid4())

        # Save file to storage
        file_path = self.storage.save_uploaded_video(video_id, file.file, filename)

        # Extract video metadata
        metadata = self._extract_video_metadata(file_path)

        # Convert to max_fps if needed and enabled in settings
        original_fps = metadata.get("fps", 0)
        if settings.enable_fps_conversion and original_fps > settings.max_fps:
            print(f"Video has {original_fps} fps (max allowed: {settings.max_fps} fps)")
            print(f"Attempting to convert video to {settings.max_fps} fps...")
            converted_path = self._convert_to_target_fps(file_path, video_id, settings.max_fps)

            # Check if conversion actually happened (or was skipped due to missing ffmpeg)
            if converted_path != file_path:
                file_path = converted_path
                # Re-extract metadata after conversion
                metadata = self._extract_video_metadata(file_path)
                print(f"Video converted successfully. New fps: {metadata.get('fps')}")
            else:
                print(f"WARNING: Video uploaded at {original_fps} fps (exceeds {settings.max_fps} fps limit)")
                print("FPS conversion was skipped - ffmpeg not available")

        # Create database record
        video = Video(
            id=video_id,
            filename=f"original{extension}",
            original_filename=filename,
            filepath=str(file_path),
            fps=metadata.get("fps"),
            total_frames=metadata.get("total_frames"),
            width=metadata.get("width"),
            height=metadata.get("height"),
            duration_seconds=metadata.get("duration_seconds"),
            file_size_bytes=self.storage.get_file_size(file_path),
        )

        self.db.add(video)
        self.db.commit()
        self.db.refresh(video)

        return video

    def get_video(self, video_id: str) -> Video:
        """Get video by ID."""
        video = self.db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        return video

    def get_video_path(self, video_id: str) -> Path:
        """Get the filesystem path to a video."""
        video = self.get_video(video_id)
        path = Path(video.filepath)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Video file not found on disk")
        return path

    def get_annotated_video_path(self, video_id: str) -> Path:
        """Get the path to the annotated video."""
        video = self.get_video(video_id)
        if not video.annotated_video_path:
            raise HTTPException(status_code=404, detail="Annotated video not available")

        path = Path(video.annotated_video_path)
        if not path.exists():
            raise HTTPException(
                status_code=404, detail="Annotated video file not found on disk"
            )
        return path

    def list_videos(self, page: int = 1, page_size: int = 20) -> tuple[list[Video], int]:
        """List all videos with pagination."""
        total = self.db.query(Video).count()
        videos = (
            self.db.query(Video)
            .order_by(Video.uploaded_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )
        return videos, total

    def delete_video(self, video_id: str) -> None:
        """Delete a video and all associated data."""
        video = self.get_video(video_id)

        # Delete from storage
        self.storage.delete_video(video_id)

        # Delete from database (cascades to jobs and frame_data)
        self.db.delete(video)
        self.db.commit()

    def update_processing_outputs(
        self,
        video_id: str,
        annotated_video_path: str | None = None,
        csv_path: str | None = None,
        stats_path: str | None = None,
    ) -> Video:
        """Update video with processing output paths."""
        video = self.get_video(video_id)

        if annotated_video_path:
            video.annotated_video_path = annotated_video_path
        if csv_path:
            video.csv_path = csv_path
        if stats_path:
            video.stats_path = stats_path

        self.db.commit()
        self.db.refresh(video)
        return video

    def _extract_video_metadata(self, video_path: Path) -> dict:
        """Extract metadata from video file using OpenCV."""
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_seconds = total_frames / fps if fps > 0 else 0

            return {
                "fps": fps,
                "total_frames": total_frames,
                "width": width,
                "height": height,
                "duration_seconds": duration_seconds,
            }
        finally:
            cap.release()

    def _convert_to_target_fps(self, input_path: Path, video_id: str, target_fps: int) -> Path:
        """Convert video to target fps using ffmpeg.

        Args:
            input_path: Path to the original video file
            video_id: Unique video ID
            target_fps: Target frame rate

        Returns:
            Path to the converted video file
        """
        # Create output path (same directory, with _XXfps suffix before extension)
        output_path = input_path.parent / f"{input_path.stem}_{target_fps}fps{input_path.suffix}"

        try:
            # Check if ffmpeg is available using shutil.which
            ffmpeg_path = shutil.which('ffmpeg')

            if not ffmpeg_path:
                # Log warning and skip conversion instead of failing
                print("WARNING: ffmpeg not found. Skipping FPS conversion.")
                print("To enable FPS conversion, install ffmpeg or disable with enable_fps_conversion=False in config")
                return input_path

            # Use ffmpeg to convert fps
            # -filter:v fps=X: Set output frame rate (drops/duplicates frames as needed)
            # -c:v libx264: Re-encode with H.264 codec
            # -crf 18: High quality (lower = better quality, 18 is visually lossless)
            # -preset fast: Encoding speed preset
            # -c:a copy: Copy audio stream without re-encoding
            cmd = [
                ffmpeg_path,
                '-i', str(input_path),
                '-filter:v', f'fps={target_fps}',  # Filter to set fps
                '-c:v', 'libx264',                  # Video codec
                '-crf', '18',                       # Quality
                '-preset', 'fast',                  # Encoding speed
                '-c:a', 'copy',                     # Copy audio
                '-y',                               # Overwrite output file
                str(output_path)
            ]

            # Run ffmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # Delete original file
            input_path.unlink()

            return output_path

        except subprocess.CalledProcessError as e:
            # If conversion fails, clean up and raise error
            if output_path.exists():
                output_path.unlink()
            raise HTTPException(
                status_code=500,
                detail=f"Video conversion failed: {e.stderr}"
            )
        except Exception as e:
            # Clean up on any error
            if output_path.exists():
                output_path.unlink()
            raise HTTPException(
                status_code=500,
                detail=f"Video conversion failed: {str(e)}"
            )
