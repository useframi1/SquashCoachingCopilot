'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { VideoDropzone } from '@/components/upload/VideoDropzone';
import { UploadProgress } from '@/components/upload/UploadProgress';
import { Button } from '@/components/ui/Button';
import { useUpload } from '@/hooks/useUpload';

export default function HomePage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const router = useRouter();
  const { uploadAndAnalyze, isUploading, isAnalyzing, status, error, reset } = useUpload();

  const handleUpload = async () => {
    if (!selectedFile) return;

    try {
      const jobId = await uploadAndAnalyze(selectedFile);
      // Redirect to dashboard when complete
      if (jobId) {
        router.push(`/dashboard/${jobId}`);
      }
    } catch (err) {
      console.error('Upload error:', err);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    reset();
  };

  const isProcessing = isUploading || isAnalyzing;
  const showProgress = status !== null;

  // Auto-redirect when analysis is complete
  useEffect(() => {
    if (status?.status === 'completed' && status.job_id) {
      setTimeout(() => {
        router.push(`/dashboard/${status.job_id}`);
      }, 1500);
    }
  }, [status, router]);

  return (
    <main className="min-h-screen bg-[var(--background)]">
      <div className="mx-auto max-w-4xl px-6 py-16">
        {/* Header */}
        <div className="text-center">
          <h1 className="text-5xl font-bold text-white">
            Squash Coaching{' '}
            <span className="text-[var(--primary)]">Copilot</span>
          </h1>
          <p className="mt-4 text-lg text-[var(--foreground-secondary)]">
            Upload your squash match video and get detailed performance analysis
          </p>
        </div>

        {/* Upload Section */}
        <div className="mt-12 space-y-6">
          <VideoDropzone
            onFileSelect={setSelectedFile}
            selectedFile={selectedFile}
            onClear={handleClear}
            disabled={isProcessing}
          />

          {selectedFile && !showProgress && (
            <div className="flex justify-center">
              <Button
                onClick={handleUpload}
                disabled={isProcessing}
                size="lg"
                className="min-w-[200px]"
              >
                {isUploading ? 'Uploading...' : 'Start Analysis'}
              </Button>
            </div>
          )}

          {showProgress && status && (
            <UploadProgress status={status} />
          )}

          {error && (
            <div className="rounded-xl border border-red-500/20 bg-red-500/10 p-6">
              <p className="text-center text-red-400">{error}</p>
              <div className="mt-4 flex justify-center">
                <Button onClick={handleClear} variant="outline">
                  Try Again
                </Button>
              </div>
            </div>
          )}
        </div>

        {/* Features */}
        <div className="mt-16 grid grid-cols-1 gap-6 md:grid-cols-3">
          <div className="rounded-xl border border-[var(--border)] bg-[var(--card-bg)] p-6">
            <div className="text-4xl mb-3">📊</div>
            <h3 className="text-lg font-semibold text-white">Detailed Analytics</h3>
            <p className="mt-2 text-sm text-[var(--foreground-secondary)]">
              Get comprehensive statistics about rallies, strokes, and player movements
            </p>
          </div>
          <div className="rounded-xl border border-[var(--border)] bg-[var(--card-bg)] p-6">
            <div className="text-4xl mb-3">🎥</div>
            <h3 className="text-lg font-semibold text-white">Video Breakdown</h3>
            <p className="mt-2 text-sm text-[var(--foreground-secondary)]">
              View annotated rally videos with player tracking and stroke detection
            </p>
          </div>
          <div className="rounded-xl border border-[var(--border)] bg-[var(--card-bg)] p-6">
            <div className="text-4xl mb-3">📈</div>
            <h3 className="text-lg font-semibold text-white">Performance Insights</h3>
            <p className="mt-2 text-sm text-[var(--foreground-secondary)]">
              Analyze patterns and improve your game with AI-powered insights
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
