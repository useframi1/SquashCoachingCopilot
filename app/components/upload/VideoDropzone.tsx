'use client';

import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Video, X } from 'lucide-react';
import { cn, formatFileSize } from '@/lib/utils';

export interface VideoDropzoneProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  onClear: () => void;
  disabled?: boolean;
}

export function VideoDropzone({
  onFileSelect,
  selectedFile,
  onClear,
  disabled,
}: VideoDropzoneProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        onFileSelect(acceptedFiles[0]);
      }
    },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
    },
    multiple: false,
    disabled: disabled || !!selectedFile,
  });

  if (selectedFile) {
    return (
      <div className="rounded-xl border-2 border-[var(--border)] bg-[var(--card-bg)] p-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="rounded-lg bg-[var(--primary)] p-3">
              <Video className="h-8 w-8 text-white" />
            </div>
            <div>
              <p className="text-lg font-medium text-white">{selectedFile.name}</p>
              <p className="text-sm text-[var(--foreground-secondary)]">
                {formatFileSize(selectedFile.size)}
              </p>
            </div>
          </div>
          {!disabled && (
            <button
              onClick={onClear}
              className="rounded-lg p-2 text-[var(--foreground-secondary)] hover:bg-[var(--border)] hover:text-white transition-colors"
            >
              <X className="h-5 w-5" />
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div
      {...getRootProps()}
      className={cn(
        'cursor-pointer rounded-xl border-2 border-dashed p-12 transition-all',
        isDragActive
          ? 'border-[var(--primary)] bg-[var(--primary)]/10'
          : 'border-[var(--border)] bg-[var(--card-bg)] hover:border-[var(--primary)] hover:bg-[var(--border)]',
        disabled && 'cursor-not-allowed opacity-50'
      )}
    >
      <input {...getInputProps()} />
      <div className="flex flex-col items-center gap-4 text-center">
        <div
          className={cn(
            'rounded-full p-6 transition-colors',
            isDragActive
              ? 'bg-[var(--primary)]'
              : 'bg-[var(--border)]'
          )}
        >
          <Upload
            className={cn(
              'h-12 w-12',
              isDragActive ? 'text-white' : 'text-[var(--foreground-secondary)]'
            )}
          />
        </div>
        <div>
          <p className="text-xl font-medium text-white">
            {isDragActive ? 'Drop your video here' : 'Drag & drop your squash video'}
          </p>
          <p className="mt-2 text-sm text-[var(--foreground-secondary)]">
            or click to browse files
          </p>
          <p className="mt-4 text-xs text-[var(--foreground-secondary)]">
            Supported formats: MP4, AVI, MOV, MKV, WebM
          </p>
        </div>
      </div>
    </div>
  );
}
