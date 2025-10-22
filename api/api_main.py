"""FastAPI application for squash video analysis."""

import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from models import Job, get_db, init_db

# Load environment variables from .env file
load_dotenv()

# Configure paths
VIDEOS_DIR = Path(os.getenv("VIDEOS_DIR", "/app/volumes/videos"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/volumes/output"))


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    yield
    # Shutdown (add cleanup code here if needed)


# Initialize FastAPI app
app = FastAPI(
    title="Squash Coaching API",
    description="API for analyzing squash match videos",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Squash Coaching API is running"}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload a video file for analysis.

    Args:
        file: Video file to upload

    Returns:
        Job details with job_id
    """
    # Validate file type
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    # Generate unique job ID
    job_id = uuid.uuid4()

    # Save video file
    video_path = VIDEOS_DIR / f"{job_id}.mp4"
    try:
        with open(video_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")

    # Create job in database
    job = Job(
        job_id=job_id,
        video_filename=file.filename,
        status="uploaded",
        upload_time=datetime.utcnow(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    return {
        "job_id": str(job.job_id),
        "filename": job.video_filename,
        "status": job.status,
        "upload_time": job.upload_time.isoformat(),
        "message": "Video uploaded successfully. Call POST /analyze/{job_id} to start processing.",
    }


@app.post("/analyze/{job_id}")
async def analyze_video(job_id: str, db: Session = Depends(get_db)):
    """
    Trigger analysis for an uploaded video.

    Args:
        job_id: UUID of the job

    Returns:
        Confirmation message
    """
    # Validate job_id format
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id format")

    # Get job from database
    job = db.query(Job).filter(Job.job_id == job_uuid).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Check if job is in valid state
    if job.status != "uploaded":
        raise HTTPException(
            status_code=400,
            detail=f"Job cannot be analyzed. Current status: {job.status}",
        )

    # Update status to pending
    job.status = "pending"
    db.commit()

    return {
        "job_id": str(job.job_id),
        "status": job.status,
        "message": "Analysis triggered. Worker will process the video shortly.",
    }


@app.get("/status/{job_id}")
async def get_status(job_id: str, db: Session = Depends(get_db)):
    """
    Get the current status of a job.

    Args:
        job_id: UUID of the job

    Returns:
        Job status and metadata
    """
    # Validate job_id format
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id format")

    # Get job from database
    job = db.query(Job).filter(Job.job_id == job_uuid).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job.to_dict()


@app.get("/results/{job_id}")
async def get_results(job_id: str, db: Session = Depends(get_db)):
    """
    Get the analysis results for a completed job.

    Args:
        job_id: UUID of the job

    Returns:
        Complete analysis results including rallies and statistics
    """
    # Validate job_id format
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id format")

    # Get job from database
    job = db.query(Job).filter(Job.job_id == job_uuid).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Check if job is completed
    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Results not available. Current status: {job.status}",
        )

    # Return results
    return {
        "job_id": str(job.job_id),
        "video_filename": job.video_filename,
        "status": job.status,
        "complete_time": job.complete_time.isoformat() if job.complete_time else None,
        "results": job.results_json,
    }


@app.get("/videos/{job_id}/rallies/{rally_num}")
async def get_rally_video(job_id: str, rally_num: int, db: Session = Depends(get_db)):
    """
    Stream a specific rally video.

    Args:
        job_id: UUID of the job
        rally_num: Rally number (1-indexed)

    Returns:
        Video file stream
    """
    # Validate job_id format
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id format")

    # Get job from database
    job = db.query(Job).filter(Job.job_id == job_uuid).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Check if job is completed
    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Rally videos not available. Current status: {job.status}",
        )

    # Construct video path
    rally_path = OUTPUT_DIR / str(job.job_id) / f"rally_{rally_num}.mp4"

    # Check if video exists
    if not rally_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Rally video {rally_num} not found"
        )

    # Stream video file
    return FileResponse(
        path=rally_path,
        media_type="video/mp4",
        filename=f"rally_{rally_num}.mp4",
    )


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str, db: Session = Depends(get_db)):
    """
    Delete a job and its associated files.

    Args:
        job_id: UUID of the job

    Returns:
        Confirmation message
    """
    # Validate job_id format
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id format")

    # Get job from database
    job = db.query(Job).filter(Job.job_id == job_uuid).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Delete video file
    video_path = VIDEOS_DIR / f"{job_id}.mp4"
    if video_path.exists():
        video_path.unlink()

    # Delete output directory
    output_path = OUTPUT_DIR / str(job_id)
    if output_path.exists():
        import shutil

        shutil.rmtree(output_path)

    # Delete job from database
    db.delete(job)
    db.commit()

    return {
        "job_id": str(job_uuid),
        "message": "Job and associated files deleted successfully",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
