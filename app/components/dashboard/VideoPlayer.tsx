'use client';

import { useState } from 'react';
import { Card, CardContent } from '../ui/Card';
import { Button } from '../ui/Button';

interface VideoPlayerProps {
  videoUrl: string;
  rallyNumber: number;
}

export function VideoPlayer({ videoUrl, rallyNumber }: VideoPlayerProps) {
  const [playing, setPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);

  const handlePlayPause = () => {
    setPlaying(!playing);
  };

  const handlePlaybackRateChange = (rate: number) => {
    setPlaybackRate(rate);
  };

  const playbackRates = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2];

  return (
    <Card className="bg-card-bg border-border">
      <CardContent className="p-0">
        {/* Video Player */}
        <div className="relative bg-black aspect-video">
          <video
            src={videoUrl}
            className="w-full h-full"
            controls={false}
            autoPlay={playing}
            playsInline
          />
        </div>

        {/* Custom Controls */}
        <div className="p-4 space-y-4">
          {/* Control Buttons */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {/* Play/Pause */}
              <Button
                onClick={handlePlayPause}
                className="bg-primary hover:bg-primary-hover"
              >
                {playing ? (
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M5 4h3v12H5V4zm7 0h3v12h-3V4z" />
                  </svg>
                ) : (
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
                  </svg>
                )}
              </Button>

              {/* Rally Info */}
              <span className="text-foreground text-sm ml-2">
                Rally #{rallyNumber}
              </span>
            </div>

            {/* Playback Speed */}
            <div className="flex items-center gap-2">
              <span className="text-foreground-secondary text-sm">Speed:</span>
              <div className="flex gap-1">
                {playbackRates.map((rate) => (
                  <Button
                    key={rate}
                    onClick={() => handlePlaybackRateChange(rate)}
                    className={`px-2 py-1 text-xs ${
                      playbackRate === rate
                        ? 'bg-primary hover:bg-primary-hover'
                        : 'bg-card-bg hover:bg-border'
                    }`}
                  >
                    {rate}x
                  </Button>
                ))}
              </div>
            </div>
          </div>

          {/* Note about using native controls */}
          <div className="text-center text-foreground-secondary text-sm">
            <p>Use browser controls for full playback features</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
