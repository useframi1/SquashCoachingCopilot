"""Pipeline orchestration service."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
from sqlalchemy.orm import Session

from backend.models.job import Job, JobStatus
from backend.models.video import Video
from backend.models.frame_data import FrameData
from backend.storage.local import LocalStorage

logger = logging.getLogger(__name__)


class PipelineService:
    """Service for managing pipeline jobs and execution."""

    def __init__(self, db: Session):
        self.db = db
        self.storage = LocalStorage()

    def create_job(self, video_id: str) -> Job:
        """Create a new processing job for a video."""
        # Verify video exists
        video = self.db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise ValueError(f"Video {video_id} not found")

        # Check for existing pending/processing jobs
        existing = (
            self.db.query(Job)
            .filter(
                Job.video_id == video_id,
                Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING]),
            )
            .first()
        )
        if existing:
            raise ValueError(f"Job {existing.id} already in progress for this video")

        # Create new job
        job = Job(video_id=video_id)
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)

        return job

    def get_job(self, job_id: str) -> Job:
        """Get job by ID."""
        job = self.db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise ValueError(f"Job {job_id} not found")
        return job

    def list_jobs(
        self,
        video_id: str | None = None,
        status: JobStatus | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Job], int]:
        """List jobs with optional filters."""
        query = self.db.query(Job)

        if video_id:
            query = query.filter(Job.video_id == video_id)
        if status:
            query = query.filter(Job.status == status)

        total = query.count()
        jobs = (
            query.order_by(Job.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )

        return jobs, total

    def update_job_progress(self, job_id: str, stage: str, progress: float) -> None:
        """Update job progress (called from background worker)."""
        job = self.get_job(job_id)
        job.update_progress(stage, progress)
        self.db.commit()

    def start_job(self, job_id: str) -> None:
        """Mark job as started."""
        job = self.get_job(job_id)
        job.start()
        self.db.commit()

    def complete_job(self, job_id: str) -> None:
        """Mark job as completed."""
        job = self.get_job(job_id)
        job.complete()
        self.db.commit()

    def fail_job(self, job_id: str, error_message: str) -> None:
        """Mark job as failed."""
        job = self.get_job(job_id)
        job.fail(error_message)
        self.db.commit()

    def cancel_job(self, job_id: str) -> Job:
        """Cancel a pending or processing job."""
        job = self.get_job(job_id)

        if job.status not in [JobStatus.PENDING, JobStatus.PROCESSING]:
            raise ValueError(f"Cannot cancel job with status {job.status}")

        job.cancel()
        self.db.commit()
        self.db.refresh(job)

        return job

    def run_pipeline(
        self,
        job_id: str,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> None:
        """
        Execute the pipeline for a job.

        This method is designed to be called from a background worker.
        """
        from squashcopilot.pipeline import Pipeline

        job = self.get_job(job_id)
        video = job.video

        try:
            # Mark job as started
            self.start_job(job_id)

            # Get video path
            video_path = Path(video.filepath)
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # Get output directory
            output_dir = self.storage.get_output_directory(video.id)

            # Create progress callback wrapper
            def pipeline_progress(stage: str, percent: float):
                self.update_job_progress(job_id, stage, percent)
                if progress_callback:
                    progress_callback(stage, percent)

            # Build pipeline config
            config = {
                "video_path": str(video_path),
                "output": {
                    "base_directory": str(output_dir),
                    "create_video_subdirectory": False,
                    "save_annotated_video": True,
                    "save_csv": True,
                    "save_statistics": True,
                },
            }

            # Run pipeline with context manager for proper cleanup
            with Pipeline(config=config, progress_callback=pipeline_progress) as pipeline:
                result = pipeline.run()

            # Store results in database
            self._store_pipeline_results(video.id, result, output_dir)

            # Update video with output paths
            video.annotated_video_path = str(output_dir / f"{video_path.stem}_annotated.mp4")
            video.csv_path = str(output_dir / f"{video_path.stem}_analysis.csv")
            video.stats_path = str(output_dir / f"{video_path.stem}_stats.json")
            video.processed_at = datetime.utcnow()
            self.db.commit()

            # Mark job as completed
            self.complete_job(job_id)

        except Exception as e:
            logger.exception(f"Pipeline failed for job {job_id}")
            self.fail_job(job_id, str(e))
            raise

    def _store_pipeline_results(self, video_id: str, result: dict, output_dir: Path) -> None:
        """Store pipeline results (frame data) in the database."""
        # Try to load the CSV output
        csv_files = list(output_dir.glob("*_analysis.csv"))
        if not csv_files:
            logger.warning(f"No CSV output found for video {video_id}")
            return

        csv_path = csv_files[0]
        df = pd.read_csv(csv_path)

        # Delete existing frame data for this video
        self.db.query(FrameData).filter(FrameData.video_id == video_id).delete()

        # Insert new frame data in batches
        batch_size = 1000
        frame_records = []

        for _, row in df.iterrows():
            frame_data = FrameData(
                video_id=video_id,
                frame_number=int(row.get("frame_number", 0)),
                timestamp=float(row.get("timestamp", 0)),
                ball_x=row.get("ball_x") if pd.notna(row.get("ball_x")) else None,
                ball_y=row.get("ball_y") if pd.notna(row.get("ball_y")) else None,
                player_1_x_meter=row.get("player_1_x_meter") if pd.notna(row.get("player_1_x_meter")) else None,
                player_1_y_meter=row.get("player_1_y_meter") if pd.notna(row.get("player_1_y_meter")) else None,
                player_2_x_meter=row.get("player_2_x_meter") if pd.notna(row.get("player_2_x_meter")) else None,
                player_2_y_meter=row.get("player_2_y_meter") if pd.notna(row.get("player_2_y_meter")) else None,
                is_rally_frame=bool(row.get("is_rally_frame", False)),
                rally_id=int(row.get("rally_id")) if pd.notna(row.get("rally_id")) else None,
                is_wall_hit=bool(row.get("is_wall_hit", False)),
                wall_hit_x_meter=row.get("wall_hit_x_meter") if pd.notna(row.get("wall_hit_x_meter")) else None,
                wall_hit_y_meter=row.get("wall_hit_y_meter") if pd.notna(row.get("wall_hit_y_meter")) else None,
                is_racket_hit=bool(row.get("is_racket_hit", False)),
                racket_hit_player_id=int(row.get("racket_hit_player_id")) if pd.notna(row.get("racket_hit_player_id")) else None,
                stroke_type=row.get("stroke_type") if pd.notna(row.get("stroke_type")) else None,
                shot_type=row.get("shot_type") if pd.notna(row.get("shot_type")) else None,
                shot_direction=row.get("shot_direction") if pd.notna(row.get("shot_direction")) else None,
                shot_depth=row.get("shot_depth") if pd.notna(row.get("shot_depth")) else None,
            )
            frame_records.append(frame_data)

            if len(frame_records) >= batch_size:
                self.db.bulk_save_objects(frame_records)
                self.db.commit()
                frame_records = []

        # Insert remaining records
        if frame_records:
            self.db.bulk_save_objects(frame_records)
            self.db.commit()

        logger.info(f"Stored {len(df)} frames for video {video_id}")
