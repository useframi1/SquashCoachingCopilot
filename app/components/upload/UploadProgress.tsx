'use client';

import { CheckCircle2, XCircle, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { StatusResponse } from '@/lib/types';

export interface UploadProgressProps {
  status: StatusResponse;
}

const statusConfig = {
  uploaded: {
    icon: CheckCircle2,
    color: 'text-green-500',
    bgColor: 'bg-green-500/10',
    label: 'Uploaded',
    description: 'Video uploaded successfully',
    animate: false,
  },
  pending: {
    icon: Loader2,
    color: 'text-yellow-500',
    bgColor: 'bg-yellow-500/10',
    label: 'Pending',
    description: 'Waiting for analysis to start',
    animate: true,
  },
  processing: {
    icon: Loader2,
    color: 'text-blue-500',
    bgColor: 'bg-blue-500/10',
    label: 'Processing',
    description: 'Analyzing your video...',
    animate: true,
  },
  completed: {
    icon: CheckCircle2,
    color: 'text-green-500',
    bgColor: 'bg-green-500/10',
    label: 'Completed',
    description: 'Analysis complete!',
    animate: false,
  },
  failed: {
    icon: XCircle,
    color: 'text-red-500',
    bgColor: 'bg-red-500/10',
    label: 'Failed',
    description: 'Analysis failed',
    animate: false,
  },
};

export function UploadProgress({ status }: UploadProgressProps) {
  const config = statusConfig[status.status];
  const Icon = config.icon;

  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--card-bg)] p-8">
      <div className="flex items-start gap-6">
        <div className={cn('rounded-full p-4', config.bgColor)}>
          <Icon
            className={cn('h-8 w-8', config.color, config.animate && 'animate-spin')}
          />
        </div>
        <div className="flex-1">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-xl font-semibold text-white">{config.label}</h3>
              <p className="mt-1 text-sm text-[var(--foreground-secondary)]">
                {config.description}
              </p>
            </div>
          </div>

          <div className="mt-4 space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-[var(--foreground-secondary)]">Video:</span>
              <span className="text-white">{status.video_filename}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-[var(--foreground-secondary)]">Job ID:</span>
              <span className="font-mono text-xs text-white">
                {status.job_id.slice(0, 8)}...
              </span>
            </div>
            {status.start_time && (
              <div className="flex justify-between text-sm">
                <span className="text-[var(--foreground-secondary)]">Started:</span>
                <span className="text-white">
                  {new Date(status.start_time).toLocaleTimeString()}
                </span>
              </div>
            )}
            {status.error_message && (
              <div className="mt-4 rounded-lg bg-red-500/10 border border-red-500/20 p-4">
                <p className="text-sm text-red-400 font-medium">Error:</p>
                <p className="mt-1 text-xs text-red-300 font-mono whitespace-pre-wrap">
                  {status.error_message}
                </p>
              </div>
            )}
          </div>

          {(status.status === 'pending' || status.status === 'processing') && (
            <div className="mt-6">
              <div className="h-2 w-full overflow-hidden rounded-full bg-[var(--border)]">
                <div className="h-full w-full origin-left animate-pulse bg-gradient-to-r from-[var(--primary)] to-[var(--accent)]" />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
