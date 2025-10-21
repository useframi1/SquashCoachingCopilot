'use client';

import { useEffect, useRef } from 'react';
import { X } from 'lucide-react';

interface VideoModalProps {
  videoUrl: string;
  rallyNumber: number;
  onClose: () => void;
}

export default function VideoModal({ videoUrl, rallyNumber, onClose }: VideoModalProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  // Handle escape key to close modal
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [onClose]);

  // Prevent body scroll when modal is open
  useEffect(() => {
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, []);

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-5xl mx-4 bg-white rounded-xl shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-black/10 bg-cream">
          <h3 className="text-xl font-bold text-black">
            Rally {rallyNumber} - Video Playback
          </h3>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-black/5 transition-colors"
            aria-label="Close video"
          >
            <X className="w-6 h-6 text-black" />
          </button>
        </div>

        {/* Video Player */}
        <div className="relative bg-black">
          <video
            ref={videoRef}
            src={videoUrl}
            controls
            autoPlay
            className="w-full max-h-[70vh]"
            onError={(e) => {
              console.error('Video playback error:', e);
            }}
          >
            Your browser does not support the video tag.
          </video>
        </div>

        {/* Footer */}
        <div className="p-4 bg-cream border-t border-black/10">
          <p className="text-sm text-black/60 text-center">
            Press <kbd className="px-2 py-1 bg-white border border-black/20 rounded text-xs">ESC</kbd> or click outside to close
          </p>
        </div>
      </div>
    </div>
  );
}
