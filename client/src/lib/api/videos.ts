import { apiClient } from './client';
import { VideoUploadResponse, VideoMetadata, PlayerNamesUpdate, PlayerNamesResponse } from '@/types/api';

/**
 * Upload a video file for analysis
 */
export const uploadVideo = async (file: File): Promise<VideoUploadResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const { data } = await apiClient.post<VideoUploadResponse>(
    '/api/videos/upload',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );

  return data;
};

/**
 * Get metadata for a specific video
 */
export const getVideoMetadata = async (videoId: string): Promise<VideoMetadata> => {
  const { data } = await apiClient.get<VideoMetadata>(`/api/videos/${videoId}`);
  return data;
};

/**
 * Delete a video and all associated data
 */
export const deleteVideo = async (videoId: string): Promise<void> => {
  await apiClient.delete(`/api/videos/${videoId}`);
};

/**
 * Update player names for a video
 */
export const updatePlayerNames = async (
  videoId: string,
  playerNames: PlayerNamesUpdate
): Promise<PlayerNamesResponse> => {
  const { data } = await apiClient.patch<PlayerNamesResponse>(
    `/api/videos/${videoId}/player-names`,
    playerNames
  );
  return data;
};

/**
 * Get the URL for the first annotated frame of a video
 */
export const getFirstFrameUrl = (videoId: string): string => {
  return `/api/videos/${videoId}/first-frame`;
};
