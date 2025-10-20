# AI Coaching Copilot - Frontend Documentation

A professional Next.js application for AI-powered squash match video analysis and coaching insights.

## Features

### 1. Homepage - Video Upload
- **Drag-and-drop interface** for easy video uploading
- Visual feedback with animations during drag operations
- File validation for video formats (MP4, MOV, AVI, WebM)
- File size display after upload
- Clean, professional UI with custom color theme

### 2. Analysis Progress
- Real-time status tracking with loading animations
- Status messages for different analysis stages:
  - **uploaded**: Video uploaded, ready for analysis
  - **pending**: Analysis triggered, waiting for worker
  - **processing**: Pipeline currently analyzing the match
  - **completed**: Analysis finished successfully
  - **failed**: Error occurred during processing
- Status badge display

### 3. Professional Dashboard
- **Three key stat cards** with icons:
  - Total Rallies (primary card in dark red)
  - Average Rally Duration
  - Total Strokes Detected
- **Two interactive charts**:
  - **Rally Durations** (Bar Chart): Duration of each rally in seconds
  - **Stroke Type Distribution** (Bar Chart): Breakdown of forehand/backhand strokes
- **Rally Breakdown Grid**: Clickable cards for each rally showing:
  - Rally number
  - Duration in seconds
  - Frame count
  - Stroke count
- **Video Player**: Click any rally to watch the annotated video
- Responsive design for all screen sizes

## Design System

### Color Theme
- **Dark Red**: `#8B0000` - Primary action color, key stats
- **Black**: `#000000` - Text and secondary elements
- **White**: `#FFFFFF` - Cards and foreground
- **Cream**: `#faf8f5` - Background color

### Typography
- Uses Geist Sans and Geist Mono font families
- Large, bold headings for clear hierarchy
- Professional spacing and sizing

## Technology Stack

### Core
- **Next.js 15.5.6** - React framework with App Router
- **React 19.1.0** - UI library
- **TypeScript** - Type safety

### Styling
- **Tailwind CSS v4** - Utility-first CSS framework
- Custom color theme configuration

### Charts & Visualizations
- **Recharts** - Professional chart library
  - Bar charts for rally durations and stroke distribution

### Icons & UI
- **Lucide React** - Beautiful, consistent icon set
- **Framer Motion** - Animation library (available for enhancements)

## Project Structure

```
frontend/
├── app/
│   ├── dashboard/
│   │   └── [id]/
│   │       └── page.tsx          # Dashboard page (dynamic route)
│   ├── globals.css                # Global styles with custom theme
│   ├── layout.tsx                 # Root layout with metadata
│   └── page.tsx                   # Homepage with upload & progress
├── components/
│   ├── AnalysisProgress.tsx       # Progress tracking component
│   └── VideoUpload.tsx            # Drag-and-drop upload component
├── lib/
│   ├── api.ts                     # API utilities for backend communication
│   └── types.ts                   # TypeScript type definitions
├── .env.local                     # Environment variables
├── .env.example                   # Environment template
├── BACKEND_ARCHITECTURE.md        # Backend API documentation
└── PROJECT_DOCUMENTATION.md       # This file
```

## Backend Integration

This frontend integrates with the **Squash Coaching API** backend built with FastAPI, PostgreSQL, and background workers.

### Backend Architecture
- **FastAPI**: REST API with 6 endpoints
- **PostgreSQL**: Database for job tracking and results storage
- **Worker**: Background processor for video analysis
- **Docker Compose**: Orchestration of all services

### API Endpoints Used

#### 1. Upload Video
```
POST /upload
Content-Type: multipart/form-data
Body: { file: File }

Response:
{
  "job_id": "uuid",
  "filename": "match.mp4",
  "status": "uploaded",
  "upload_time": "2025-10-19T16:00:00",
  "message": "Video uploaded successfully. Call POST /analyze/{job_id} to start processing."
}
```

#### 2. Trigger Analysis
```
POST /analyze/{job_id}

Response:
{
  "job_id": "uuid",
  "status": "pending",
  "message": "Analysis triggered. Worker will process the video shortly."
}
```

#### 3. Check Status (Polling)
```
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

#### 4. Get Results
```
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
              "position": [x, y],
              "real_position": [x, y],
              "stroke_type": "backhand"
            },
            "player2": {
              "player_id": 2,
              "position": [x, y],
              "real_position": [x, y],
              "stroke_type": "forehand"
            },
            "ball": {
              "position": [x, y],
              "is_wall_hit": true,
              "ball_hit_real_position": [x, y]
            }
          }
        ]
      }
    ]
  }
}
```

#### 5. Stream Rally Video
```
GET /videos/{job_id}/rallies/{rally_num}

Response: Video file stream (MP4)
```

#### 6. Delete Job (Optional)
```
DELETE /jobs/{job_id}

Response:
{
  "job_id": "uuid",
  "message": "Job and associated files deleted successfully"
}
```

### Configuration

Set your backend API URL in `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Getting Started

### Prerequisites
Make sure the backend is running first:

```bash
cd api
docker-compose up --build
```

The backend will be available at `http://localhost:8000`

### Frontend Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the application.

### Build for Production

```bash
npm run build
npm start
```

## User Flow

1. **Upload**: User lands on homepage and drags/drops a squash match video file
2. **Confirmation**: After upload succeeds, "Start Analysis" button appears
3. **Initiate**: Click button to trigger analysis via `/analyze/{job_id}` endpoint
4. **Progress**: Frontend polls `/status/{job_id}` every 2 seconds showing:
   - uploaded → pending → processing → completed
5. **Results**: Automatically redirected to dashboard when status is "completed"
6. **Dashboard**: View comprehensive analysis with:
   - Rally statistics
   - Duration charts
   - Stroke type distribution
   - Individual rally videos
7. **Navigate**: Click "Back to Home" to analyze another video

## Component Details

### VideoUpload Component
- Handles drag-and-drop events
- Validates file types (must be video/*)
- Shows upload status with icons
- Displays file metadata (name, size)

### AnalysisProgress Component
- Animated loading spinner
- Dynamic status messages based on job lifecycle
- Status badge showing current state
- User-friendly messages

### Dashboard Page
Features:
- **StatCard** component for metrics display
- **RallyCard** component for individual rally selection
- Responsive chart layouts with Recharts
- Embedded video player for rally playback
- Data processing to calculate:
  - Rally durations from frame counts (assumes 30 FPS)
  - Stroke type aggregation from frame data
  - Total stroke counts per rally

## Best Practices Implemented

1. **TypeScript**: Full type safety with interfaces matching backend API
2. **Client Components**: Using 'use client' directive appropriately
3. **Error Handling**: Try-catch blocks with user-friendly alerts
4. **Loading States**: Proper loading indicators throughout
5. **Responsive Design**: Mobile-first approach with Tailwind
6. **Code Organization**: Separated concerns (components, lib, app)
7. **Environment Variables**: Secure API URL configuration
8. **Polling Strategy**: Efficient status polling (2s interval) with cleanup
9. **Accessibility**: Semantic HTML and proper button elements
10. **Performance**: Optimized charts with ResponsiveContainer

## Data Processing

### Rally Duration Calculation
```typescript
duration = (end_frame - start_frame) / 30  // assumes 30 FPS
```

### Stroke Type Aggregation
The dashboard iterates through all rally frames and counts each stroke type (forehand, backhand, etc.) from both players to create the stroke distribution chart.

### Rally Selection
Users can click any rally card to load and play the corresponding annotated video from the backend's `/videos/{job_id}/rallies/{rally_num}` endpoint.

## Future Enhancements

- Add player performance comparison charts
- Implement heatmap visualization for player positions
- Export reports as PDF
- Historical match analysis and comparison
- User authentication and match history
- Real-time WebSocket updates instead of polling
- Advanced filtering by stroke type, player, or rally duration
- Slow-motion playback controls
- Frame-by-frame analysis view
- Downloadable rally videos
- Share analysis results via link

## Machine Learning Pipeline

The backend uses 5 specialized ML models (provided as .whl packages):
1. **Ball Detection**: Tracks ball position and wall hits
2. **Court Detection**: Identifies court boundaries
3. **Player Tracking**: Tracks player positions
4. **Rally State**: Detects rally start/end
5. **Stroke Detection**: Classifies stroke types (forehand/backhand)

All analysis is performed by background workers, allowing the API to remain responsive.

## Troubleshooting

### Video Upload Fails
- Check backend is running on port 8000
- Verify `NEXT_PUBLIC_API_URL` in `.env.local`
- Ensure file is a valid video format

### Analysis Stuck in "pending"
- Check worker container is running: `docker-compose logs worker`
- Worker polls database every 5 seconds
- Large videos may take several minutes

### Dashboard Shows No Results
- Ensure analysis status is "completed"
- Check browser console for API errors
- Verify job_id in URL matches uploaded video

### Rally Videos Won't Play
- Check backend video endpoint is accessible
- Some browsers require HTTPS for video playback
- Try different video codec/format

## Notes

- The application uses polling (every 2 seconds) to check analysis status
- Charts are fully responsive and adapt to screen size
- All colors follow the custom theme (dark red, black, white, cream)
- The dashboard uses dynamic routing with Next.js App Router
- Environment variables must start with `NEXT_PUBLIC_` to be accessible client-side
- Backend requires Docker and Docker Compose
- First backend build takes several minutes due to large ML packages (~143MB)
- Docker image size is ~2-3GB (normal for ML applications)
