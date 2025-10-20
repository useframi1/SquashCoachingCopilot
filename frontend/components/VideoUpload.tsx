'use client';

import { useState, useRef } from 'react';
import { Upload, Video } from 'lucide-react';

interface VideoUploadProps {
  onVideoUpload: (file: File) => void;
  uploadedFile: File | null;
}

export default function VideoUpload({ onVideoUpload, uploadedFile }: VideoUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (file.type.startsWith('video/')) {
        onVideoUpload(file);
      } else {
        alert('Please upload a video file');
      }
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (file.type.startsWith('video/')) {
        onVideoUpload(file);
      } else {
        alert('Please upload a video file');
      }
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleClick}
      className={`
        relative w-full aspect-video cursor-pointer
        border-4 border-dashed rounded-xl
        transition-all duration-300
        ${isDragging
          ? 'border-dark-red bg-dark-red/5 scale-[0.98]'
          : uploadedFile
            ? 'border-dark-red bg-white'
            : 'border-black/30 bg-white hover:border-dark-red hover:bg-dark-red/5'
        }
      `}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        onChange={handleFileSelect}
        className="hidden"
      />

      <div className="absolute inset-0 flex flex-col items-center justify-center p-8">
        {uploadedFile ? (
          <div className="text-center">
            <div className="mb-4 flex justify-center">
              <div className="bg-dark-red/10 p-6 rounded-full">
                <Video className="w-16 h-16 text-dark-red" />
              </div>
            </div>
            <h3 className="text-2xl font-semibold text-black mb-2">
              Video Uploaded
            </h3>
            <p className="text-lg text-black/60">{uploadedFile.name}</p>
            <p className="text-sm text-black/40 mt-2">
              {(uploadedFile.size / (1024 * 1024)).toFixed(2)} MB
            </p>
          </div>
        ) : (
          <div className="text-center">
            <div className="mb-6 flex justify-center">
              <div className="bg-black/5 p-8 rounded-full">
                <Upload className="w-20 h-20 text-black/40" />
              </div>
            </div>
            <h3 className="text-3xl font-semibold text-black mb-4">
              Drag your video here
            </h3>
            <p className="text-lg text-black/60">
              or click to browse
            </p>
            <p className="text-sm text-black/40 mt-4">
              Supported formats: MP4, MOV, AVI, WebM
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
