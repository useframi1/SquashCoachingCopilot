"""
Application configuration using Pydantic Settings.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "SquashCoachingCopilot API"
    debug: bool = False

    # Database
    database_url: str = "sqlite:///./backend/data/squash.db"

    # Storage paths
    upload_dir: Path = Path("backend/storage/uploads")
    temp_dir: Path = Path("backend/storage/temp")

    # Video processing
    max_video_size_mb: int = 500
    allowed_video_extensions: set = {".mp4", ".mov", ".avi", ".mkv"}

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # Pipeline settings
    pipeline_timeout_seconds: int = 3600  # 1 hour max for processing

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def setup_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        # Ensure database directory exists
        db_path = Path(self.database_url.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
