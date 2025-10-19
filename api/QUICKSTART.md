# Quick Start Guide

Get your Squash Coaching API running in 3 steps!

## Step 1: Start the Services

```bash
cd api
docker-compose up --build
```

Wait for the services to start. You should see:
- `squash-postgres` - Database ready
- `squash-api` - API running on http://localhost:8000
- `squash-worker` - Worker polling for jobs

## Step 2: Upload and Analyze a Video

### Using the Test Script (Easiest)
```bash
./test_api.sh tests/video-5.mp4
```

### Using cURL (Manual)
```bash
# Upload
curl -X POST http://localhost:8000/upload -F "file=@tests/video-5.mp4"
# Copy the job_id from response

# Trigger analysis
curl -X POST http://localhost:8000/analyze/{job_id}

# Check status
curl http://localhost:8000/status/{job_id}

# Get results (when status is "completed")
curl http://localhost:8000/results/{job_id}
```

## Step 3: View Results

### API Response Example
```json
{
  "job_id": "uuid",
  "status": "completed",
  "results": {
    "total_rallies": 5,
    "avg_rally_duration": 12.3,
    "rallies": [
      {
        "start_frame": 100,
        "end_frame": 500,
        "strokes": [...],
        "ball_hits": [...]
      }
    ]
  }
}
```

### Download Rally Videos
```bash
curl http://localhost:8000/videos/{job_id}/rallies/1 -o rally_1.mp4
curl http://localhost:8000/videos/{job_id}/rallies/2 -o rally_2.mp4
# ... etc
```

## Next Steps

### For Frontend Development
Your frontend should:
1. Upload video via `POST /upload`
2. Trigger analysis via `POST /analyze/{job_id}`
3. Poll `GET /status/{job_id}` every 5 seconds until status is "completed"
4. Fetch results via `GET /results/{job_id}`
5. Display statistics and rally videos

### Example Frontend Flow (React/Vue/etc)
```javascript
// 1. Upload
const formData = new FormData();
formData.append('file', videoFile);
const uploadRes = await fetch('http://localhost:8000/upload', {
  method: 'POST',
  body: formData
});
const { job_id } = await uploadRes.json();

// 2. Trigger
await fetch(`http://localhost:8000/analyze/${job_id}`, { method: 'POST' });

// 3. Poll status
const pollStatus = setInterval(async () => {
  const statusRes = await fetch(`http://localhost:8000/status/${job_id}`);
  const { status } = await statusRes.json();

  if (status === 'completed') {
    clearInterval(pollStatus);
    // 4. Fetch results
    const resultsRes = await fetch(`http://localhost:8000/results/${job_id}`);
    const data = await resultsRes.json();
    displayDashboard(data.results);
  }
}, 5000);

// 5. Display rally videos
const videoUrl = `http://localhost:8000/videos/${job_id}/rallies/1`;
```

## Troubleshooting

### Services won't start
```bash
# Check if ports are in use
lsof -i :8000  # API port
lsof -i :5432  # PostgreSQL port

# Clean everything and restart
docker-compose down -v
docker-compose up --build
```

### Worker not processing
```bash
# Check worker logs
docker-compose logs -f worker

# Restart worker
docker-compose restart worker
```

### Database connection issues
```bash
# Check PostgreSQL health
docker-compose ps
docker-compose logs postgres
```

## Stop Services
```bash
# Stop but keep data
docker-compose stop

# Stop and remove containers (keeps volumes)
docker-compose down

# Stop and remove everything including data
docker-compose down -v
```
