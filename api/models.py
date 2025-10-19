"""Database models for the squash video analysis API."""

import uuid
from datetime import datetime
from sqlalchemy import (
    create_engine,
    Column,
    String,
    DateTime,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

Base = declarative_base()


class Job(Base):
    """Job model representing a video analysis task."""

    __tablename__ = "jobs"

    job_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_filename = Column(String(255), nullable=False)
    status = Column(
        String(50), nullable=False, default="uploaded"
    )  # uploaded, pending, processing, completed, failed
    upload_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    start_time = Column(DateTime, nullable=True)  # When processing started
    complete_time = Column(DateTime, nullable=True)  # When processing completed
    results_json = Column(JSONB, nullable=True)  # Full analysis results
    error_message = Column(Text, nullable=True)  # Error details if failed

    def to_dict(self):
        """Convert job to dictionary."""
        return {
            "job_id": str(self.job_id),
            "video_filename": self.video_filename,
            "status": self.status,
            "upload_time": self.upload_time.isoformat() if self.upload_time else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "complete_time": (
                self.complete_time.isoformat() if self.complete_time else None
            ),
            "error_message": self.error_message,
        }


# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://squash:squash@localhost:5432/squash_db"
)

# Create engine and session factory
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database by creating all tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
