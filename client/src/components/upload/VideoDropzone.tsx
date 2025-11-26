'use client';

import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileVideo, X } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

interface VideoDropzoneProps {
  onFileSelect: (file: File) => void;
  isUploading?: boolean;
}

/**
 * Drag-and-drop video upload component
 * Spacious, minimalistic design with white background
 */
export function VideoDropzone({ onFileSelect, isUploading = false }: VideoDropzoneProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0];
        setSelectedFile(file);
        onFileSelect(file);
      }
    },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv'],
    },
    multiple: false,
    disabled: isUploading,
  });

  const handleRemove = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedFile(null);
  };

  return (
    <div
      {...getRootProps()}
      className={cn(
        'relative flex flex-col items-center justify-center',
        'w-full h-[500px] p-12',
        'border-2 border-dashed rounded-lg',
        'transition-all duration-200 cursor-pointer',
        'focus-within:ring-2 focus-within:ring-red-700 focus-within:ring-offset-2',
        isDragActive
          ? 'border-red-700 bg-red-50'
          : 'border-gray-300 bg-white hover:border-red-700 hover:bg-gray-50',
        isUploading && 'cursor-not-allowed opacity-60'
      )}
      role="button"
      aria-label="Upload video file"
      tabIndex={0}
    >
      <input {...getInputProps()} aria-label="File input" />

      {selectedFile ? (
        <div className="flex flex-col items-center space-y-6">
          <div className="relative">
            <FileVideo className="w-24 h-24 text-red-700" />
            {!isUploading && (
              <button
                onClick={handleRemove}
                className="absolute -top-2 -right-2 p-1 bg-red-700 text-white rounded-full hover:bg-red-800 transition-colors focus:outline-none focus:ring-2 focus:ring-red-700 focus:ring-offset-2"
                aria-label="Remove selected file"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>

          <div className="text-center space-y-2">
            <p className="text-lg font-semibold text-gray-900">{selectedFile.name}</p>
            <p className="text-sm text-gray-500">
              {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
            </p>
          </div>

          {isUploading && (
            <div className="flex flex-col items-center space-y-2">
              <div className="w-48 h-2 bg-gray-200 rounded-full overflow-hidden">
                <div className="h-full bg-red-700 animate-pulse" style={{ width: '100%' }} />
              </div>
              <p className="text-sm text-gray-600">Uploading...</p>
            </div>
          )}
        </div>
      ) : (
        <div className="flex flex-col items-center space-y-6">
          <div className="p-6 bg-gray-100 rounded-full">
            <Upload className="w-16 h-16 text-gray-400" />
          </div>

          <div className="text-center space-y-3">
            <p className="text-xl font-semibold text-gray-900">
              {isDragActive ? 'Drop video here' : 'Drag and drop your squash match video'}
            </p>
            <p className="text-sm text-gray-500">
              or click to browse
            </p>
            <p className="text-xs text-gray-400">
              Supported formats: MP4, AVI, MOV, MKV
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
