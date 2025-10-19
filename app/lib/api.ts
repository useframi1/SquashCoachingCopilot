import type {
  HealthResponse,
  UploadResponse,
  AnalyzeResponse,
  StatusResponse,
  ResultsResponse,
} from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new ApiError(response.status, error.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

export const api = {
  /**
   * Health check endpoint
   */
  async healthCheck(): Promise<HealthResponse> {
    const response = await fetch(`${API_BASE_URL}/`);
    return handleResponse<HealthResponse>(response);
  },

  /**
   * Upload a video file
   */
  async uploadVideo(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    return handleResponse<UploadResponse>(response);
  },

  /**
   * Trigger analysis for an uploaded video
   */
  async analyzeVideo(jobId: string): Promise<AnalyzeResponse> {
    const response = await fetch(`${API_BASE_URL}/analyze/${jobId}`, {
      method: 'POST',
    });

    return handleResponse<AnalyzeResponse>(response);
  },

  /**
   * Get job status
   */
  async getStatus(jobId: string): Promise<StatusResponse> {
    const response = await fetch(`${API_BASE_URL}/status/${jobId}`);
    return handleResponse<StatusResponse>(response);
  },

  /**
   * Get analysis results
   */
  async getResults(jobId: string): Promise<ResultsResponse> {
    const response = await fetch(`${API_BASE_URL}/results/${jobId}`);
    return handleResponse<ResultsResponse>(response);
  },

  /**
   * Get rally video URL
   */
  getRallyVideoUrl(jobId: string, rallyNum: number): string {
    return `${API_BASE_URL}/videos/${jobId}/rallies/${rallyNum}`;
  },

  /**
   * Poll job status until completion or failure
   */
  async pollStatus(
    jobId: string,
    onUpdate?: (status: StatusResponse) => void,
    interval = 2000
  ): Promise<StatusResponse> {
    return new Promise((resolve, reject) => {
      const poll = async () => {
        try {
          const status = await this.getStatus(jobId);

          if (onUpdate) {
            onUpdate(status);
          }

          if (status.status === 'completed') {
            resolve(status);
          } else if (status.status === 'failed') {
            reject(new Error(status.error_message || 'Analysis failed'));
          } else {
            setTimeout(poll, interval);
          }
        } catch (error) {
          reject(error);
        }
      };

      poll();
    });
  },
};

export { ApiError };
