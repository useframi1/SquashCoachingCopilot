'use client';

import { Loader2 } from 'lucide-react';

interface AnalysisProgressProps {
  state: string;
}

const stateMessages: Record<string, string> = {
  uploaded: 'Preparing analysis...',
  pending: 'Waiting for worker...',
  processing: 'Analyzing squash match...',
  completed: 'Analysis complete!',
  failed: 'Analysis failed',
};

export default function AnalysisProgress({ state }: AnalysisProgressProps) {
  const message = stateMessages[state] || 'Initializing...';

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div className="bg-white rounded-xl shadow-2xl p-12">
        <div className="flex flex-col items-center">
          {/* Loading Spinner */}
          <div className="mb-8">
            <Loader2 className="w-24 h-24 text-dark-red animate-spin" />
          </div>

          {/* State Message */}
          <h2 className="text-3xl font-semibold text-black mb-4">
            {message}
          </h2>

          {/* Status Badge */}
          <div className="px-6 py-2 bg-dark-red/10 text-dark-red rounded-full font-medium">
            Status: {state}
          </div>

          <p className="mt-6 text-black/60 text-center">
            This may take a few moments. Please don&apos;t close this window.
          </p>
        </div>
      </div>
    </div>
  );
}
