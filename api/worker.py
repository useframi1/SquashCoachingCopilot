"""Background worker for processing video analysis jobs."""

import os
import time
import traceback
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from models import Job, SessionLocal, init_db
from pipeline import Pipeline
from config import PipelineConfig


# Configure paths
VIDEOS_DIR = Path(os.getenv("VIDEOS_DIR", "/app/volumes/videos"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/volumes/output"))

# Worker configuration
POLL_INTERVAL = int(os.getenv("WORKER_POLL_INTERVAL", "5"))  # seconds


def process_job(job: Job, db: Session):
    """
    Process a single job by running the pipeline.

    Args:
        job: Job object to process
        db: Database session
    """
    print(f"[Worker] Processing job {job.job_id}...")

    try:
        # Update status to processing
        job.status = "processing"
        job.start_time = datetime.utcnow()
        db.commit()

        # Construct paths
        video_path = str(VIDEOS_DIR / f"{job.job_id}.mp4")
        output_path = str(OUTPUT_DIR / str(job.job_id))

        # Ensure output directory exists
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # Configure pipeline
        config = PipelineConfig(
            video_path=video_path,
            base_output_path=output_path,
            output_codec="mp4v",  # Use software codec for Docker compatibility
        )

        print(f"[Worker] Running pipeline for {job.video_filename}...")
        print(f"[Worker] Video path: {video_path}")
        print(f"[Worker] Output path: {output_path}")

        # Run the pipeline
        pipeline = Pipeline(config=config)
        results = pipeline.run()

        # Update job with results
        job.status = "completed"
        job.complete_time = datetime.utcnow()
        job.results_json = results
        db.commit()

        print(f"[Worker] Job {job.job_id} completed successfully!")
        print(f"[Worker] Total rallies: {results.get('total_rallies', 0)}")

    except Exception as e:
        # Handle errors
        error_message = f"{str(e)}\n\n{traceback.format_exc()}"
        print(f"[Worker] Job {job.job_id} failed: {error_message}")

        job.status = "failed"
        job.complete_time = datetime.utcnow()
        job.error_message = error_message
        db.commit()


def worker_loop():
    """
    Main worker loop that continuously checks for pending jobs.
    """
    print("[Worker] Starting worker loop...")
    print(f"[Worker] Polling interval: {POLL_INTERVAL} seconds")
    print(f"[Worker] Videos directory: {VIDEOS_DIR}")
    print(f"[Worker] Output directory: {OUTPUT_DIR}")

    while True:
        try:
            # Create database session
            db = SessionLocal()

            # Find pending jobs
            pending_jobs = db.query(Job).filter(Job.status == "pending").all()

            if pending_jobs:
                print(f"[Worker] Found {len(pending_jobs)} pending job(s)")

                # Process each pending job
                for job in pending_jobs:
                    process_job(job, db)

            else:
                # No pending jobs, wait before checking again
                time.sleep(POLL_INTERVAL)

            # Close database session
            db.close()

        except KeyboardInterrupt:
            print("[Worker] Received interrupt signal, shutting down...")
            break

        except Exception as e:
            print(f"[Worker] Error in worker loop: {e}")
            print(traceback.format_exc())
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    # Initialize database
    print("[Worker] Initializing database...")
    init_db()

    # Start worker loop
    worker_loop()
