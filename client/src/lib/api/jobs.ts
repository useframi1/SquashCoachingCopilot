import { apiClient } from './client';
import { JobResponse, JobStatusResponse } from '@/types/api';

/**
 * Create a new processing job for a video
 */
export const createJob = async (videoId: string): Promise<JobResponse> => {
  const { data } = await apiClient.post<JobResponse>('/api/pipeline/jobs', {
    video_id: videoId,
  });
  return data;
};

/**
 * Get lightweight job status (optimized for polling)
 */
export const getJobStatus = async (jobId: string): Promise<JobStatusResponse> => {
  const { data } = await apiClient.get<JobStatusResponse>(
    `/api/pipeline/jobs/${jobId}/status`
  );
  return data;
};

/**
 * Get full job details
 */
export const getJob = async (jobId: string): Promise<JobResponse> => {
  const { data} = await apiClient.get<JobResponse>(`/api/pipeline/jobs/${jobId}`);
  return data;
};

/**
 * Cancel a running or pending job
 */
export const cancelJob = async (jobId: string): Promise<JobResponse> => {
  const { data } = await apiClient.post<JobResponse>(
    `/api/pipeline/jobs/${jobId}/cancel`
  );
  return data;
};
