"use client";

import {useEffect} from "react";
import {useParams, useRouter} from "next/navigation";
import {Loader2, CheckCircle2, XCircle, AlertCircle} from "lucide-react";
import {useJobPolling} from "@/lib/hooks/useJobPolling";
import {useJob} from "@/lib/hooks/useJob";
import {JobStatus} from "@/types/api";
import {Button} from "@/components/ui/button";

export default function ProcessingPage() {
    const params = useParams();
    const router = useRouter();
    const jobId = params.jobId as string;

    // Poll job status every 2 seconds
    const {data: statusData, isLoading: statusLoading} = useJobPolling(jobId);

    // Fetch full job details to get video_id
    const {data: jobData} = useJob(jobId);

    // Navigate to dashboard when processing is complete
    useEffect(() => {
        if (statusData?.status === JobStatus.COMPLETED && jobData?.video_id) {
            // Wait a moment before redirecting for better UX
            setTimeout(() => {
                // Use replace to prevent going back to processing/upload pages
                // Add showNaming=true to trigger player naming modal
                router.replace(
                    `/dashboard/${jobData.video_id}/overview?showNaming=true`
                );
            }, 1500);
        }
    }, [statusData?.status, jobData?.video_id, router]);

    if (statusLoading) {
        return (
            <div className="min-h-screen bg-white flex items-center justify-center">
                <Loader2 className="w-12 h-12 text-red-700 animate-spin" />
            </div>
        );
    }

    const status = statusData?.status;
    const progress = statusData?.progress || 0;
    const currentStage = statusData?.current_stage;

    return (
        <div className="min-h-screen bg-white flex flex-col">
            {/* Header */}
            <header className="py-8 px-12 border-b border-gray-200">
                <h1 className="text-3xl font-bold text-gray-900">
                    Squash Coaching Copilot
                </h1>
            </header>

            {/* Main Content */}
            <main className="flex-1 flex flex-col items-center justify-center px-12">
                <div className="w-full max-w-2xl space-y-12">
                    {/* Status Icon */}
                    <div className="flex justify-center">
                        {status === JobStatus.PROCESSING ||
                        status === JobStatus.PENDING ? (
                            <div className="p-8 bg-red-50 rounded-full">
                                <Loader2 className="w-24 h-24 text-red-700 animate-spin" />
                            </div>
                        ) : status === JobStatus.COMPLETED ? (
                            <div className="p-8 bg-green-50 rounded-full">
                                <CheckCircle2 className="w-24 h-24 text-green-600" />
                            </div>
                        ) : status === JobStatus.FAILED ? (
                            <div className="p-8 bg-red-50 rounded-full">
                                <XCircle className="w-24 h-24 text-red-600" />
                            </div>
                        ) : (
                            <div className="p-8 bg-gray-50 rounded-full">
                                <AlertCircle className="w-24 h-24 text-gray-400" />
                            </div>
                        )}
                    </div>

                    {/* Status Text */}
                    <div className="text-center space-y-4">
                        <h2 className="text-3xl font-semibold text-gray-900">
                            {status === JobStatus.PENDING &&
                                "Preparing Analysis..."}
                            {status === JobStatus.PROCESSING &&
                                "Analyzing Match..."}
                            {status === JobStatus.COMPLETED &&
                                "Analysis Complete!"}
                            {status === JobStatus.FAILED && "Analysis Failed"}
                            {status === JobStatus.CANCELLED &&
                                "Analysis Cancelled"}
                        </h2>

                        {currentStage && status === JobStatus.PROCESSING && (
                            <p className="text-lg text-gray-600">
                                {currentStage}
                            </p>
                        )}

                        {status === JobStatus.COMPLETED && (
                            <p className="text-lg text-gray-600">
                                Redirecting to dashboard...
                            </p>
                        )}

                        {status === JobStatus.FAILED && (
                            <p className="text-lg text-red-600">
                                An error occurred during processing. Please try
                                uploading again.
                            </p>
                        )}
                    </div>

                    {/* Progress Bar */}
                    {(status === JobStatus.PROCESSING ||
                        status === JobStatus.PENDING) && (
                        <div className="space-y-3">
                            <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-red-700 transition-all duration-500 ease-out"
                                    style={{width: `${progress}%`}}
                                />
                            </div>
                            <p className="text-center text-sm text-gray-600">
                                {progress.toFixed(0)}% complete
                            </p>
                        </div>
                    )}

                    {/* Action Buttons */}
                    {status === JobStatus.FAILED && (
                        <div className="flex justify-center">
                            <Button
                                onClick={() => router.push("/")}
                                size="lg"
                                className="bg-red-700 hover:bg-red-800"
                            >
                                Upload New Video
                            </Button>
                        </div>
                    )}
                </div>
            </main>

            {/* Footer */}
            <footer className="py-6 px-12 border-t border-gray-200">
                <p className="text-sm text-gray-500 text-center">
                    Processing time is based on video length. It may take a few
                    minutes to complete.
                </p>
            </footer>
        </div>
    );
}
