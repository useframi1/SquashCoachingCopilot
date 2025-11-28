"""Video metadata database model."""

import uuid
from datetime import datetime

from sqlalchemy import Column, String, Float, Integer, DateTime
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import relationship

from backend.models.database import Base


class Video(Base):
    """Video metadata and storage information."""

    __tablename__ = "videos"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    filepath = Column(String(512), nullable=False)

    # Video properties
    fps = Column(Float, nullable=True)
    total_frames = Column(Integer, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    file_size_bytes = Column(Integer, nullable=True)

    # Processing outputs
    annotated_video_path = Column(String(512), nullable=True)
    csv_path = Column(String(512), nullable=True)
    stats_path = Column(String(512), nullable=True)
    first_frame_path = Column(String(512), nullable=True)

    # Player metadata
    player_1_name = Column(String(255), nullable=True)
    player_2_name = Column(String(255), nullable=True)

    # Court calibration data (stored as JSON)
    calibration_data = Column(JSON, nullable=True)

    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    processed_at = Column(DateTime, nullable=True)

    # Relationships
    jobs = relationship("Job", back_populates="video", cascade="all, delete-orphan")
    frames = relationship("FrameData", back_populates="video", cascade="all, delete-orphan")
    games = relationship("Game", back_populates="video", cascade="all, delete-orphan")
    match = relationship("Match", back_populates="video", uselist=False, cascade="all, delete-orphan")
    llm_conversations = relationship("LLMConversation", back_populates="video", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Video(id={self.id}, filename={self.filename})>"
