"""Frame-by-frame analysis data model."""

from sqlalchemy import Column, String, Float, Integer, Boolean, ForeignKey, Index
from sqlalchemy.orm import relationship

from backend.models.database import Base


class FrameData(Base):
    """Frame-by-frame analysis results from pipeline."""

    __tablename__ = "frame_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String(36), ForeignKey("videos.id"), nullable=False, index=True)
    frame_number = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)

    # Ball tracking
    ball_x = Column(Float, nullable=True)
    ball_y = Column(Float, nullable=True)

    # Player 1 tracking (meters)
    player_1_x_meter = Column(Float, nullable=True)
    player_1_y_meter = Column(Float, nullable=True)

    # Player 2 tracking (meters)
    player_2_x_meter = Column(Float, nullable=True)
    player_2_y_meter = Column(Float, nullable=True)

    # Rally information
    is_rally_frame = Column(Boolean, default=False, nullable=False)
    rally_id = Column(Integer, nullable=True, index=True)

    # Hit detection
    is_wall_hit = Column(Boolean, default=False, nullable=False)
    wall_hit_x_meter = Column(Float, nullable=True)
    wall_hit_y_meter = Column(Float, nullable=True)
    is_racket_hit = Column(Boolean, default=False, nullable=False)
    racket_hit_player_id = Column(Integer, nullable=True)

    # Shot classification
    stroke_type = Column(String(50), nullable=True)  # forehand, backhand
    shot_type = Column(String(50), nullable=True)  # straight_drive, cross_court_drop, etc.
    shot_direction = Column(String(50), nullable=True)  # straight, cross_court
    shot_depth = Column(String(50), nullable=True)  # drop, long

    # Relationships
    video = relationship("Video", back_populates="frames")

    # Composite index for efficient queries
    __table_args__ = (
        Index("idx_video_frame", "video_id", "frame_number"),
        Index("idx_video_rally", "video_id", "rally_id"),
    )

    def __repr__(self) -> str:
        return f"<FrameData(video_id={self.video_id}, frame={self.frame_number})>"
