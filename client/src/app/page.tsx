'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { VideoDropzone } from '@/components/upload/VideoDropzone';
import { uploadVideo } from '@/lib/api/videos';
import { createJob } from '@/lib/api/jobs';
import { Button } from '@/components/ui/button';

export default function Home() {
  const router = useRouter();
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setError(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setError(null);

    try {
      // Step 1: Upload video
      const uploadResponse = await uploadVideo(selectedFile);
      const videoId = uploadResponse.id;

      // Step 2: Create processing job
      const jobResponse = await createJob(videoId);
      const jobId = jobResponse.id;

      // Step 3: Navigate to processing page (replace to prevent back button)
      router.replace(`/processing/${jobId}`);
    } catch (err) {
      console.error('Upload failed:', err);
      setError('Failed to upload video. Please try again.');
      setIsUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-white flex flex-col">
      {/* Header */}
      <header className="py-8 px-12">
        <h1 className="text-3xl font-bold text-gray-900">
          Squash Coaching Copilot
        </h1>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex flex-col items-center justify-center px-12 pb-24">
        <div className="w-full max-w-4xl space-y-8">
          {/* Title and Description */}
          <div className="text-center space-y-4">
            <h2 className="text-2xl font-semibold text-gray-900">
              Upload Your Match Video
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Upload a squash match video to get comprehensive AI-powered analysis including
              shot types, player movement, T-zone control, and match momentum.
            </p>
          </div>

          {/* Upload Area */}
          <VideoDropzone
            onFileSelect={handleFileSelect}
            isUploading={isUploading}
          />

          {/* Error Message */}
          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}

          {/* Upload Button */}
          {selectedFile && !isUploading && (
            <div className="flex justify-center">
              <Button
                onClick={handleUpload}
                size="lg"
                className="bg-red-700 hover:bg-red-800"
              >
                Start Analysis
              </Button>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="py-6 px-12 border-t border-gray-200">
        <p className="text-sm text-gray-500 text-center">
          Powered by AI • Squash Coaching Copilot
        </p>
      </footer>
    </div>
  );
}
