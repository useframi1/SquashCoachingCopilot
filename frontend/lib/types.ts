// API Response Types
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
    status: "uploaded" | "pending" | "processing" | "completed" | "failed";
    upload_time: string;
    start_time: string | null;
    complete_time: string | null;
}

export interface ResultsResponse {
    job_id: string;
    video_filename: string;
    status: string;
    complete_time: string;
    results: AnalysisResults;
}

export interface AnalysisResults {
    fps: number;
    total_rallies: number;
    avg_rally_duration: number;
    rallies: Rally[];
}

export interface Rally {
    start_frame: number;
    end_frame: number;
    rally_frames: RallyFrame[];
}

export interface RallyFrame {
    frame_number: number;
    timestamp: number;
    player1: PlayerData;
    player2: PlayerData;
    ball: BallData;
}

export interface PlayerData {
    player_id: number;
    position: [number, number];
    real_position: [number, number];
    stroke_type: string | null;
}

export interface BallData {
    position: [number, number];
    is_wall_hit: boolean;
    ball_hit_real_position: [number, number] | null;
}
