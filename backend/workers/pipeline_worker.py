"""Pipeline worker for background processing.

This module provides a worker class that can be used for more advanced
background processing scenarios (e.g., with Celery or as a standalone worker).
"""

import logging
from typing import Callable, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.config import settings

logger = logging.getLogger(__name__)


class PipelineWorker:
    """
    Worker for processing pipeline jobs.

    Can be used with:
    - FastAPI BackgroundTasks (simple async)
    - Celery (distributed processing)
    - Standalone worker process
    """

    def __init__(self, database_url: Optional[str] = None):
        """Initialize worker with database connection."""
        self.database_url = database_url or settings.database_url
        self._engine = None
        self._session_factory = None

    def _get_session(self):
        """Get a new database session."""
        if self._engine is None:
            self._engine = create_engine(
                self.database_url,
                connect_args=(
                    {"check_same_thread": False}
                    if "sqlite" in self.database_url
                    else {}
                ),
            )
            self._session_factory = sessionmaker(
                autocommit=False, autoflush=False, bind=self._engine
            )
        return self._session_factory()

    def process_job(
        self,
        job_id: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> dict:
        """
        Process a single job.

        Args:
            job_id: The job ID to process
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with processing results
        """
        from backend.services.pipeline_service import PipelineService

        db = self._get_session()
        try:
            service = PipelineService(db)
            service.run_pipeline(job_id, progress_callback=progress_callback)
            return {"status": "completed", "job_id": job_id}
        except Exception as e:
            logger.exception(f"Error processing job {job_id}")
            return {"status": "failed", "job_id": job_id, "error": str(e)}
        finally:
            db.close()

    def poll_and_process(self, max_jobs: int = 1) -> list[dict]:
        """
        Poll for pending jobs and process them.

        This method is useful for standalone worker processes.

        Args:
            max_jobs: Maximum number of jobs to process in one poll

        Returns:
            List of processing results
        """
        from backend.models.job import Job, JobStatus

        db = self._get_session()
        results = []

        try:
            # Get pending jobs
            pending_jobs = (
                db.query(Job)
                .filter(Job.status == JobStatus.PENDING)
                .order_by(Job.created_at)
                .limit(max_jobs)
                .all()
            )

            for job in pending_jobs:
                logger.info(f"Processing job {job.id}")
                result = self.process_job(job.id)
                results.append(result)

        finally:
            db.close()

        return results


def run_worker_loop(poll_interval: int = 5):
    """
    Run a worker loop that continuously polls for jobs.

    This is useful for running the worker as a standalone process.

    Args:
        poll_interval: Seconds to wait between polls when no jobs are found
    """
    import time

    worker = PipelineWorker()
    logger.info("Starting pipeline worker loop...")

    while True:
        try:
            results = worker.poll_and_process(max_jobs=1)

            if results:
                for result in results:
                    logger.info(f"Job {result['job_id']}: {result['status']}")
            else:
                # No jobs found, wait before polling again
                time.sleep(poll_interval)

        except KeyboardInterrupt:
            logger.info("Worker interrupted, shutting down...")
            break
        except Exception as e:
            logger.exception(f"Worker error: {e}")
            time.sleep(poll_interval)


if __name__ == "__main__":
    # Allow running worker directly: python -m backend.workers.pipeline_worker
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run_worker_loop()
