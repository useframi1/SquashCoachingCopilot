"use client";

import {useState} from "react";
import {useRouter} from "next/navigation";
import {VideoDropzone} from "@/components/upload/VideoDropzone";
import {uploadVideo} from "@/lib/api/videos";
import {createJob} from "@/lib/api/jobs";
import {Button} from "@/components/ui/button";

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
            console.error("Upload failed:", err);
            setError("Failed to upload video. Please try again.");
            setIsUploading(false);
        }
    };

    return (
        <div className="h-screen bg-background flex flex-col">
            {/* Header */}
            <header className="py-4 px-8">
                <h1 className="text-xl font-bold">Squash Coaching Copilot</h1>
            </header>

            {/* Main Content */}
            <main className="flex-1 flex flex-col p-32 overflow-hidden">
                {/* Title and Description */}
                <div className="text-center mb-6">
                    <h2 className="text-lg font-semibold mb-2">
                        Upload Your Match Video
                    </h2>
                    <p className="text-muted-foreground text-sm">
                        Upload a squash match video to get comprehensive
                        AI-powered analysis including shot types, player
                        movement, T-zone control, and match momentum.
                    </p>
                </div>

                {/* Upload Area - Takes remaining space */}
                <div className="flex-1 min-h-0">
                    <VideoDropzone
                        onFileSelect={handleFileSelect}
                        isUploading={isUploading}
                    />
                </div>

                {/* Error Message */}
                {error && (
                    <div className="mt-6 p-4 bg-destructive/10 border border-destructive/20 rounded-lg max-w-2xl mx-auto">
                        <p className="text-sm text-destructive">{error}</p>
                    </div>
                )}

                {/* Upload Button */}
                {selectedFile && !isUploading && (
                    <div className="flex justify-center mt-6">
                        <Button onClick={handleUpload} size="lg">
                            Start Analysis
                        </Button>
                    </div>
                )}
            </main>

            {/* Footer */}
            <footer className="py-3 px-8 border-t border-border/20">
                <p className="text-xs text-muted-foreground text-center">
                    Powered by AI • Squash Coaching Copilot
                </p>
            </footer>
        </div>
    );
}
