import { useQuery } from '@tanstack/react-query';
import { getJobStatus } from '@/lib/api/jobs';
import { JobStatus } from '@/types/api';

/**
 * Custom hook for polling job status
 * Automatically polls every 2 seconds when job is processing
 * Stops polling when job is completed, failed, or cancelled
 */
export function useJobPolling(jobId: string | null) {
  return useQuery({
    queryKey: ['job-status', jobId],
    queryFn: () => getJobStatus(jobId!),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;

      // Poll every 2 seconds if processing or pending
      if (status === JobStatus.PROCESSING || status === JobStatus.PENDING) {
        return 2000;
      }

      // Stop polling for terminal states
      return false;
    },
    retry: false, // Don't retry on error during polling
    staleTime: 0, // Always consider data stale to ensure fresh updates
  });
}
