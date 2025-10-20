# Squash Coaching API

A REST API for analyzing squash match videos using computer vision and machine learning.

## Architecture

-   **FastAPI**: REST API with 5 endpoints
-   **PostgreSQL**: Database for job tracking and results storage
-   **Worker**: Background processor for video analysis
-   **Docker Compose**: Orchestration of all services

## Features

-   Upload squash match videos
-   Trigger video analysis on-demand
-   Monitor processing status in real-time
-   Retrieve comprehensive analysis results (rallies, strokes, statistics)
-   Stream annotated rally videos

## API Endpoints

### 1. Upload Video

```bash
POST /upload
Content-Type: multipart/form-data

Response:
{
  "job_id": "uuid",
  "filename": "match.mp4",
  "status": "uploaded",
  "upload_time": "2025-10-19T16:00:00",
  "message": "Video uploaded successfully. Call POST /analyze/{job_id} to start processing."
}
```

### 2. Trigger Analysis

```bash
POST /analyze/{job_id}

Response:
{
  "job_id": "uuid",
  "status": "pending",
  "message": "Analysis triggered. Worker will process the video shortly."
}
```

### 3. Check Status

```bash
GET /status/{job_id}

Response:
{
  "job_id": "uuid",
  "video_filename": "match.mp4",
  "status": "processing",  # uploaded, pending, processing, completed, failed
  "upload_time": "2025-10-19T16:00:00",
  "start_time": "2025-10-19T16:01:00",
  "complete_time": null
}
```

### 4. Get Results

```bash
GET /results/{job_id}

Response:
{
  "job_id": "uuid",
  "video_filename": "match.mp4",
  "status": "completed",
  "complete_time": "2025-10-19T16:05:00",
  "results": {
    "total_rallies": 5,
    "avg_rally_duration": 12.3,
    "rallies": [
        {
            "start_frame": 123,
            "end_frame": 456,
            "rally_frames": [
                {
                    "frame_number": 123,
                    "timestamp": float,
                    "player1": {
                        "player_id": 1,
                        "position": (x, y),
                        "real_position": (x, y),
                        "stroke_type": "backhand",
                    }
                    "player2": {
                        "player_id": 2,
                        "position": (x, y),
                        "real_position": (x, y),
                        "stroke_type": "forehand",
                    }
                    "ball": {
                        "position": (x, y),
                        "is_wall_hit": true,
                        "ball_hit_real_position": (x, y)
                    }
                }
            ]
        }
    ]
  }
}
```

### 5. Stream Rally Video

```bash
GET /videos/{job_id}/rallies/{rally_num}

Response: Video file stream (MP4)
```

### 6. Delete Job (Bonus)

```bash
DELETE /jobs/{job_id}

Response:
{
  "job_id": "uuid",
  "message": "Job and associated files deleted successfully"
}
```

## Setup & Running

### Prerequisites

-   Docker
-   Docker Compose

### 1. Build and Start Services

```bash
cd api
docker-compose up --build
```

This will start:

-   PostgreSQL on port 5432
-   API on port 8000
-   Worker (background processor)

### 2. Test the API

```bash
# Health check
curl http://localhost:8000/

# Upload video
curl -X POST http://localhost:8000/upload \
  -F "file=@path/to/video.mp4"

# Trigger analysis
curl -X POST http://localhost:8000/analyze/{job_id}

# Check status
curl http://localhost:8000/status/{job_id}

# Get results (when completed)
curl http://localhost:8000/results/{job_id}

# Download rally video
curl http://localhost:8000/videos/{job_id}/rallies/1 -o rally_1.mp4
```

## Directory Structure

```
api/
├── api_main.py           # FastAPI application
├── worker.py             # Background job processor
├── models.py             # Database models
├── pipeline.py           # Video analysis pipeline
├── config.py             # Pipeline configuration
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container image
├── docker-compose.yml    # Service orchestration
├── packages/             # ML model packages (.whl files)
│   ├── ball_detection_pipeline-0.1.4-py3-none-any.whl
│   ├── court_detection_pipeline-0.1.2-py3-none-any.whl
│   ├── player_tracking_pipeline-0.2.2-py3-none-any.whl
│   ├── rally_state_pipeline-0.1.12-py3-none-any.whl
│   └── stroke_detection_pipeline-0.1.1-py3-none-any.whl
└── volumes/              # Runtime data (created automatically)
    ├── videos/           # Uploaded videos
    ├── output/           # Processed rally videos
    └── postgres-data/    # Database persistence
```

## Job Lifecycle

1. **uploaded** - Video uploaded, awaiting user trigger
2. **pending** - User triggered analysis, waiting for worker
3. **processing** - Pipeline currently running
4. **completed** - Analysis finished successfully
5. **failed** - Error occurred during processing

## Configuration

Environment variables (set in docker-compose.yml):

-   `DATABASE_URL`: PostgreSQL connection string
-   `VIDEOS_DIR`: Directory for uploaded videos
-   `OUTPUT_DIR`: Directory for processed outputs
-   `WORKER_POLL_INTERVAL`: Worker polling interval in seconds (default: 5)

## Development

### Run without Docker (for development)

```bash
# Install dependencies
pip install -r requirements.txt
pip install packages/*.whl

# Start PostgreSQL (or use existing instance)
# Update DATABASE_URL in models.py if needed

# Run API
uvicorn api_main:app --reload --host 0.0.0.0 --port 8000

# Run worker (in separate terminal)
python worker.py
```

## Troubleshooting

### Check logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f worker
docker-compose logs -f postgres
```

### Restart services

```bash
docker-compose restart
```

### Clean volumes and restart

```bash
docker-compose down -v
docker-compose up --build
```

## Notes

-   First build will take several minutes due to large ML packages (~143MB)
-   Docker image size will be ~2-3GB (normal for ML applications)
-   Videos and outputs persist in `./volumes/` directory
-   Database data persists in Docker volume `postgres-data`
