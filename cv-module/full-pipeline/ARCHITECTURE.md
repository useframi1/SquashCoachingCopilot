# Architecture Documentation

## Overview

The pipeline has been refactored into a clean **three-layer architecture** following SOLID principles and separation of concerns.

```
┌─────────────────────────────────────────────────────────────┐
│                    Layer 3: Analysis                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         CoachingAnalyzer                            │   │
│  │  - High-level coaching insights                     │   │
│  │  - Performance comparisons                          │   │
│  │  - Export analysis results                          │   │
│  └──────────────┬──────────────────┬───────────────────┘   │
│                 │                  │                        │
│  ┌──────────────▼──────────┐  ┌───▼──────────────────┐    │
│  │   MovementAnalyzer      │  │   RallyAnalyzer      │    │
│  │  - Speed/acceleration   │  │  - Rally segmentation│    │
│  │  - Distance metrics     │  │  - Duration/intensity│    │
│  │  - Court coverage       │  │  - Shot counting     │    │
│  │  - Sprint detection     │  │  - Rally statistics  │    │
│  └─────────────────────────┘  └──────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ FrameData (cleaned)
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Layer 2: Data Collection                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              DataCollector                          │   │
│  │  - Aggregate raw outputs                            │   │
│  │  - Validate data quality                            │   │
│  │  - Apply temporal smoothing                         │   │
│  │  - Handle missing data                              │   │
│  │  - Structure into FrameData models                  │   │
│  └────────┬─────────────────┬──────────────────┬───────┘   │
│           │                 │                  │            │
│  ┌────────▼──────┐  ┌───────▼────────┐  ┌─────▼──────┐   │
│  │  Validators   │  │ PostProcessors │  │ DataModels │   │
│  │  - Confidence │  │  - Smoothing   │  │ - FrameData│   │
│  │  - Temporal   │  │  - Filtering   │  │ - PlayerData│  │
│  │  - Bounds     │  │  - Interpolate │  │ - BallData │   │
│  └───────────────┘  └────────────────┘  └────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ Raw pipeline outputs
                              │
┌─────────────────────────────────────────────────────────────┐
│                 Layer 1: Orchestration                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │          PipelineOrchestrator                       │   │
│  │  - Video I/O management                             │   │
│  │  - Sub-pipeline coordination                        │   │
│  │  - Frame-by-frame processing                        │   │
│  │  - Progress tracking                                │   │
│  └──────────┬───────────────────────────────────────────┘  │
│             │                                               │
│  ┌──────────▼──────────────────────────────────────────┐   │
│  │              Visualizer                             │   │
│  │  - Render bounding boxes                            │   │
│  │  - Draw keypoints & skeleton                        │   │
│  │  - Display ball & rally state                       │   │
│  │  - Court visualization                              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
                    ┌─────────┴─────────┐
                    │   Sub-Pipelines   │
                    ├───────────────────┤
                    │ • CourtCalibrator │
                    │ • PlayerTracker   │
                    │ • BallTracker     │
                    │ • RallyDetector   │
                    └───────────────────┘
```

## Layer Responsibilities

### Layer 1: Orchestration
**What it does:**
- Manages video reading/writing
- Initializes and coordinates sub-pipelines
- Handles display and progress tracking
- Passes raw outputs to DataCollector
- Manages visualization rendering

**What it does NOT do:**
- Process or validate data
- Compute metrics or insights
- Make decisions about data quality

**Key Classes:**
- `PipelineOrchestrator`: Main orchestration logic
- `Visualizer`: All rendering/drawing operations

### Layer 2: Data Collection
**What it does:**
- Aggregates outputs from all sub-pipelines
- Validates data quality (confidence, temporal consistency)
- Applies temporal smoothing to reduce noise
- Handles missing/invalid data through interpolation
- Structures data into clean, validated models
- Provides trajectory and history queries

**What it does NOT do:**
- Orchestrate pipeline execution
- Compute high-level metrics or insights
- Make coaching recommendations
- Render visualizations

**Key Classes:**
- `DataCollector`: Main collection and processing logic
- `DataValidator`: Validation rules
- `TemporalSmoother`: Smoothing algorithms
- `MissingDataHandler`: Interpolation and gap filling
- Data Models: `FrameData`, `PlayerData`, `BallData`, `CourtData`

### Layer 3: Analysis
**What it does:**
- Computes movement metrics (speed, distance, coverage)
- Analyzes rally patterns and statistics
- Generates coaching insights and recommendations
- Compares player performances
- Exports analysis results

**What it does NOT do:**
- Access raw pipeline outputs directly
- Handle data validation or smoothing
- Manage video processing
- Render visualizations

**Key Classes:**
- `CoachingAnalyzer`: High-level analysis and insights
- `MovementAnalyzer`: Movement-specific metrics
- `RallyAnalyzer`: Rally-level analysis
- `metrics.py`: Shared metric computation utilities

## Data Flow

### Frame Processing Flow
```
1. VideoReader yields (frame_number, timestamp, frame)
   ↓
2. PipelineOrchestrator.process_frame():
   - CourtCalibrator.process_frame() → homographies, keypoints
   - PlayerTracker.process_frame() → player results (bbox, keypoints, confidence)
   - BallTracker.process_frame() → (ball_x, ball_y)
   - RallyStateDetector.process_frame() → rally_state
   ↓
3. DataCollector.collect_frame_data():
   - Aggregates raw outputs into FrameData
   - Stores in raw_frame_history
   ↓
4. Visualizer.render_frame():
   - Draws court keypoints
   - Draws player bboxes, keypoints, skeleton
   - Draws ball position
   - Draws rally state
   ↓
5. VideoWriter.write(annotated_frame)
```

### Post-Processing Flow
```
1. PipelineOrchestrator completes all frames
   ↓
2. DataCollector.post_process():
   a. Validate all frames (confidence, temporal consistency)
   b. Interpolate missing data (gaps up to max_interpolation_frames)
   c. Apply temporal smoothing (moving average over smoothing_window)
   ↓
3. Processed frames stored in processed_frame_history
```

### Analysis Flow
```
1. orchestrator.get_collected_data() → List[FrameData]
   ↓
2. CoachingAnalyzer.analyze_match():
   - MovementAnalyzer.analyze_both_players()
     → speed, distance, coverage, sprints
   - RallyAnalyzer.get_rally_statistics()
     → rally count, durations, intensities
   - RallyAnalyzer.analyze_all_rallies()
     → per-rally metrics
   - Generate coaching insights
   ↓
3. Export to JSON and text summary
```

## Design Benefits

### 🎯 Separation of Concerns
Each layer has ONE responsibility, making code easier to understand and maintain.

### 🧪 Testability
Each layer can be unit tested independently with mock data.

### 🔧 Maintainability
Changes in one layer rarely affect others. Clear boundaries make debugging easier.

### 📈 Extensibility
Easy to add new analyzers, validators, or post-processors without touching core logic.

### ♻️ Reusability
DataCollector and Analyzers can work with different data sources (live feed, batch processing).

### 🚀 Scalability
Clear structure makes it easy to parallelize or distribute processing.

## SOLID Principles Applied

### Single Responsibility Principle (SRP)
- Orchestrator → manages flow
- DataCollector → manages data quality
- CoachingAnalyzer → computes insights
- Visualizer → handles rendering

### Open/Closed Principle (OCP)
- Easy to extend with new analyzers (open for extension)
- Don't need to modify existing classes (closed for modification)

### Liskov Substitution Principle (LSP)
- All data models follow consistent interfaces
- Analyzers can be swapped without breaking code

### Interface Segregation Principle (ISP)
- Small, focused interfaces between layers
- Each layer only exposes what others need

### Dependency Inversion Principle (DIP)
- Layers depend on data models (abstractions)
- Not on concrete pipeline implementations

## Usage Patterns

### Pattern 1: Simple Usage (main.py)
```python
from pipeline import Pipeline
from config import PipelineConfig

config = PipelineConfig(
    video_path="video.mp4",
    output_path="output.mp4",
    analysis_output_path="match_analysis",
    display=True
)

pipeline = Pipeline(config=config)
analysis = pipeline.run()
```

**When to use**: Quick analysis, prototyping, standard use cases.

### Pattern 2: Component Configuration
```python
from pipeline import Pipeline
from config import PipelineConfig, DataCollectorConfig, VisualizerConfig

# Custom configuration
config = PipelineConfig(
    video_path="video.mp4",
    data_collector=DataCollectorConfig(
        enable_smoothing=True,
        smoothing_window=7,
        min_confidence=0.5
    ),
    visualizer=VisualizerConfig(
        show_player_keypoints=True,
        keypoint_confidence_threshold=0.6
    )
)

pipeline = Pipeline(config=config)
analysis = pipeline.run()
```

**When to use**: Custom data processing parameters, different visualization settings.

### Pattern 3: Direct Architecture Access
```python
from orchestration import PipelineOrchestrator, Visualizer
from data import DataCollector
from analysis import CoachingAnalyzer
from video_io import VideoReader, VideoWriter

# Full control over each component
data_collector = DataCollector(enable_smoothing=True)
visualizer = Visualizer(show_player_keypoints=True)
orchestrator = PipelineOrchestrator(data_collector, visualizer)

# Custom video processing loop
with VideoReader("video.mp4") as reader:
    metadata = reader.get_metadata()
    with VideoWriter("output.mp4", metadata) as writer:
        orchestrator.process_frames(
            frames_iterator=reader.frames(),
            video_metadata=metadata.__dict__,
            display=True,
            on_frame_processed=lambda fd, frame: writer.write(frame)
        )

# Custom analysis
frames = orchestrator.get_collected_data()
analyzer = CoachingAnalyzer(fps=metadata.fps)
analysis = analyzer.analyze_match(frames)
```

**When to use**: Real-time processing, custom frame handling, integration with other systems.

## Implementation Details

### Key Classes and Methods

#### PipelineOrchestrator
- `__init__(data_collector, visualizer)`: Initialize with components
- `process_frames(frames_iterator, video_metadata, display, on_frame_processed)`: Process video frames
- `process_frame(frame, frame_number, timestamp)`: Process single frame through sub-pipelines
- `get_collected_data()`: Retrieve processed frame data
- `reset()`: Reset state

#### DataCollector
- `__init__(enable_smoothing, smoothing_window, enable_validation, ...)`: Configure processing
- `collect_frame_data(frame_number, timestamp, court_data, player_results, ball_position, rally_state)`: Aggregate frame data
- `post_process()`: Apply validation, interpolation, and smoothing
- `get_frame_history(num_frames, raw)`: Retrieve frame history

#### CoachingAnalyzer
- `__init__(fps)`: Initialize with video framerate
- `analyze_match(frames)`: Complete match analysis
- `analyze_player_performance(frames, player_id)`: Single player analysis
- `compare_players(frames)`: Player comparison
- `export_analysis(frames, output_path, format)`: Export results

#### Visualizer
- `__init__(show_court_keypoints, show_player_keypoints, ...)`: Configure visualization
- `render_frame(frame, frame_data)`: Draw all elements on frame
- Uses COCO skeleton format for pose visualization

### Data Models

#### FrameData
- `frame_number`: int
- `timestamp`: float
- `court`: CourtData
- `player1`: PlayerData
- `player2`: PlayerData
- `ball`: BallData
- `rally_state`: str

#### PlayerData
- `player_id`: int
- `position`: Tuple[float, float] (bottom-center of bbox)
- `real_position`: Tuple[float, float] (court coordinates)
- `bbox`: Tuple[float, float, float, float]
- `confidence`: float
- `keypoints`: Dict (xy array and confidence array)

## Future Enhancements

### Easily Extensible
- ✅ New analyzers (TacticalAnalyzer, ShotTypeAnalyzer, FatigueAnalyzer)
- ✅ New validators (CourtBoundsValidator, PhysicsValidator)
- ✅ New post-processors (KalmanFilter, OutlierRemoval)
- ✅ Different export formats (CSV, Excel, PDF reports)
- ✅ Real-time processing mode (webcam/stream)
- ✅ Multi-video batch processing
- ✅ Custom visualization overlays

### Architecture Supports
- Parallel frame processing (independent frames)
- Distributed analysis (multiple machines)
- Plugin system for custom analyzers
- REST API for coaching app integration
- Database storage for historical analysis
- Live streaming and real-time coaching
