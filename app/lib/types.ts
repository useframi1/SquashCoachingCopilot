// API Response Types
export interface HealthResponse {
  status: string;
  message: string;
}

export interface UploadResponse {
  job_id: string;
  filename: string;
  status: string;
  upload_time: string;
  message: string;
}

export interface AnalyzeResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface StatusResponse {
  job_id: string;
  video_filename: string;
  status: 'uploaded' | 'pending' | 'processing' | 'completed' | 'failed';
  upload_time: string;
  start_time: string | null;
  complete_time: string | null;
  error_message: string | null;
}

export interface PlayerData {
  player_id: number;
  position: [number, number];
  real_position: [number, number];
  bbox: [number, number, number, number];
  confidence: number;
  keypoints: {
    xy: [number, number][];
    conf: number[];
  };
  stroke_type: 'forehand' | 'backhand' | 'none' | null;
}

export interface BallData {
  position: [number, number] | null;
  real_position: [number, number] | null;
  confidence: number;
}

export interface CourtData {
  homographies: {
    't-boxes': number[][];
    wall: number[][];
  };
  keypoints: {
    't-boxes': [number, number][];
    wall: [number, number][];
  };
  is_calibrated: boolean;
}

export interface FrameData {
  frame_number: number;
  timestamp: number;
  court: CourtData;
  player1: PlayerData;
  player2: PlayerData;
  ball: BallData;
  rally_state: 'none' | 'rally' | 'between_rallies' | null;
}

export interface RallyData {
  rally_frames: FrameData[];
  start_frame: number;
  end_frame: number;
  duration?: number;
  start_timestamp?: number;
  end_timestamp?: number;
}

export interface AnalysisResults {
  total_rallies: number;
  avg_rally_duration: number;
  rallies: RallyData[];
}

export interface ResultsResponse {
  job_id: string;
  video_filename: string;
  status: string;
  complete_time: string | null;
  results: AnalysisResults;
}

// Frontend specific types
export interface StrokeStats {
  forehand: number;
  backhand: number;
  total: number;
}

export interface PlayerStats {
  player_id: number;
  strokes: StrokeStats;
  averagePosition: [number, number];
  movementDistance: number;
}

export interface RallySummary {
  rally_number: number;
  duration: number;
  start_time: number;
  end_time: number;
  player1_strokes: StrokeStats;
  player2_strokes: StrokeStats;
  total_strokes: number;
}

export interface DashboardData {
  jobId: string;
  filename: string;
  totalRallies: number;
  avgRallyDuration: number;
  longestRally: RallySummary;
  rallies: RallySummary[];
  player1Stats: PlayerStats;
  player2Stats: PlayerStats;
}
