import { useQuery } from '@tanstack/react-query';
import { getJob } from '@/lib/api/jobs';

/**
 * Custom hook for fetching full job details
 * Use this to get video_id and other job metadata
 */
export function useJob(jobId: string | null) {
  return useQuery({
    queryKey: ['job', jobId],
    queryFn: () => getJob(jobId!),
    enabled: !!jobId,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}
