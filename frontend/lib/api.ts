import { UploadResponse, AnalyzeResponse, StatusResponse, ResultsResponse } from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function uploadVideo(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Failed to upload video');
  }

  return response.json();
}

export async function startAnalysis(jobId: string): Promise<AnalyzeResponse> {
  const response = await fetch(`${API_BASE_URL}/analyze/${jobId}`, {
    method: 'POST',
  });

  if (!response.ok) {
    throw new Error('Failed to start analysis');
  }

  return response.json();
}

export async function getJobStatus(jobId: string): Promise<StatusResponse> {
  const response = await fetch(`${API_BASE_URL}/status/${jobId}`);

  if (!response.ok) {
    throw new Error('Failed to get job status');
  }

  return response.json();
}

export async function getAnalysisResults(jobId: string): Promise<ResultsResponse> {
  const response = await fetch(`${API_BASE_URL}/results/${jobId}`);

  if (!response.ok) {
    throw new Error('Failed to get analysis results');
  }

  return response.json();
}

export function getRallyVideoUrl(jobId: string, rallyNum: number): string {
  return `${API_BASE_URL}/videos/${jobId}/rallies/${rallyNum}`;
}

export async function deleteJob(jobId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/jobs/${jobId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error('Failed to delete job');
  }
}
