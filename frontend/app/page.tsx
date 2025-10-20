'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import VideoUpload from '@/components/VideoUpload';
import AnalysisProgress from '@/components/AnalysisProgress';
import { uploadVideo, startAnalysis, getJobStatus } from '@/lib/api';

export default function Home() {
  const router = useRouter();
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisState, setAnalysisState] = useState<string>('');

  const handleVideoUpload = async (file: File) => {
    try {
      setUploadedFile(file);
      const response = await uploadVideo(file);
      setJobId(response.job_id);
    } catch (error) {
      console.error('Upload failed:', error);
      alert('Failed to upload video. Please try again.');
    }
  };

  const handleStartAnalysis = async () => {
    if (!jobId) return;

    try {
      setIsAnalyzing(true);
      await startAnalysis(jobId);

      // Poll for status
      const pollInterval = setInterval(async () => {
        const status = await getJobStatus(jobId);
        setAnalysisState(status.status);

        if (status.status === 'completed') {
          clearInterval(pollInterval);
          setTimeout(() => {
            router.push(`/dashboard/${jobId}`);
          }, 500);
        } else if (status.status === 'failed') {
          clearInterval(pollInterval);
          setIsAnalyzing(false);
          alert('Analysis failed. Please try again.');
        }
      }, 2000);
    } catch (error) {
      console.error('Analysis failed:', error);
      setIsAnalyzing(false);
      alert('Failed to start analysis. Please try again.');
    }
  };

  return (
    <div className="min-h-screen bg-cream flex flex-col items-center justify-center p-8">
      <div className="max-w-4xl w-full">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-6xl font-bold text-black mb-4">
            AI Coaching Copilot
          </h1>
          <p className="text-xl text-black/70">
            AI-powered squash video analysis for coaching insights
          </p>
        </div>

        {/* Main Content Area */}
        {!isAnalyzing ? (
          <div className="w-full">
            <VideoUpload
              onVideoUpload={handleVideoUpload}
              uploadedFile={uploadedFile}
            />
            {uploadedFile && (
              <div className="mt-8 flex justify-center">
                <button
                  onClick={handleStartAnalysis}
                  className="bg-dark-red text-white px-12 py-4 rounded-lg text-lg font-semibold hover:bg-[#a00000] transition-colors shadow-lg"
                >
                  Start Analysis
                </button>
              </div>
            )}
          </div>
        ) : (
          <AnalysisProgress state={analysisState} />
        )}
      </div>
    </div>
  );
}
