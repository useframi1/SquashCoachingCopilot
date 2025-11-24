"""Local filesystem storage implementation."""

import shutil
from pathlib import Path
from typing import BinaryIO

from backend.config import settings


class LocalStorage:
    """Local filesystem storage for videos and outputs."""

    def __init__(self):
        self.upload_dir = settings.upload_dir
        self.temp_dir = settings.temp_dir
        settings.setup_directories()

    def get_video_directory(self, video_id: str) -> Path:
        """Get the directory for a specific video."""
        video_dir = self.upload_dir / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        return video_dir

    def save_uploaded_video(self, video_id: str, file: BinaryIO, filename: str) -> Path:
        """Save an uploaded video file."""
        video_dir = self.get_video_directory(video_id)

        # Preserve original extension
        extension = Path(filename).suffix.lower()
        dest_path = video_dir / f"original{extension}"

        with open(dest_path, "wb") as dest:
            shutil.copyfileobj(file, dest)

        return dest_path

    def get_video_path(self, video_id: str) -> Path | None:
        """Get the path to the original video."""
        video_dir = self.upload_dir / video_id
        if not video_dir.exists():
            return None

        # Look for original video with any extension
        for ext in settings.allowed_video_extensions:
            path = video_dir / f"original{ext}"
            if path.exists():
                return path
        return None

    def get_annotated_video_path(self, video_id: str) -> Path | None:
        """Get the path to the annotated video."""
        video_dir = self.upload_dir / video_id
        annotated_path = video_dir / "annotated.mp4"
        return annotated_path if annotated_path.exists() else None

    def get_output_directory(self, video_id: str) -> Path:
        """Get the output directory for pipeline results."""
        output_dir = self.upload_dir / video_id / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def delete_video(self, video_id: str) -> bool:
        """Delete all files associated with a video."""
        video_dir = self.upload_dir / video_id
        if video_dir.exists():
            shutil.rmtree(video_dir)
            return True
        return False

    def get_file_size(self, path: Path) -> int:
        """Get file size in bytes."""
        return path.stat().st_size if path.exists() else 0

    def file_exists(self, path: Path) -> bool:
        """Check if a file exists."""
        return path.exists()
