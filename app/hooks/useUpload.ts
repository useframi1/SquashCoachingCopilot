'use client';

import { useState, useCallback } from 'react';
import { api } from '@/lib/api';
import type { StatusResponse } from '@/lib/types';

export function useUpload() {
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const uploadAndAnalyze = useCallback(async (file: File) => {
    try {
      setIsUploading(true);
      setError(null);

      // Upload video
      const uploadResponse = await api.uploadVideo(file);
      setJobId(uploadResponse.job_id);

      // Trigger analysis
      setIsAnalyzing(true);
      await api.analyzeVideo(uploadResponse.job_id);

      // Start polling for status
      await api.pollStatus(uploadResponse.job_id, (statusUpdate) => {
        setStatus(statusUpdate);
      });

      return uploadResponse.job_id;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Upload failed';
      setError(message);
      throw err;
    } finally {
      setIsUploading(false);
      setIsAnalyzing(false);
    }
  }, []);

  const reset = useCallback(() => {
    setIsUploading(false);
    setIsAnalyzing(false);
    setJobId(null);
    setStatus(null);
    setError(null);
  }, []);

  return {
    uploadAndAnalyze,
    isUploading,
    isAnalyzing,
    jobId,
    status,
    error,
    reset,
  };
}
