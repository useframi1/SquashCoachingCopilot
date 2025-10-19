'use client';

import { useState } from 'react';
import { VideoPlayer } from './VideoPlayer';
import { Button } from '../ui/Button';
import type { DashboardData } from '@/lib/types';
import { api } from '@/lib/api';

interface VideoPlayerTabProps {
  data: DashboardData;
}

export function VideoPlayerTab({ data }: VideoPlayerTabProps) {
  const [currentRallyIndex, setCurrentRallyIndex] = useState(0);
  const [sequentialMode, setSequentialMode] = useState(false);

  const currentRally = data.rallies[currentRallyIndex];
  const videoUrl = api.getRallyVideoUrl(data.jobId, currentRally.rally_number);

  const handlePrevious = () => {
    if (currentRallyIndex > 0) {
      setCurrentRallyIndex(currentRallyIndex - 1);
    }
  };

  const handleNext = () => {
    if (currentRallyIndex < data.rallies.length - 1) {
      setCurrentRallyIndex(currentRallyIndex + 1);
    }
  };

  const handleRallySelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setCurrentRallyIndex(parseInt(e.target.value));
  };

  return (
    <div className="space-y-6">
      {/* Rally Selector */}
      <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
        <div className="flex items-center gap-3 flex-1">
          <label htmlFor="rally-select" className="text-foreground text-sm font-medium whitespace-nowrap">
            Select Rally:
          </label>
          <select
            id="rally-select"
            value={currentRallyIndex}
            onChange={handleRallySelect}
            className="flex-1 bg-card-bg border border-border text-foreground rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
          >
            {data.rallies.map((rally, index) => (
              <option key={rally.rally_number} value={index}>
                Rally #{rally.rally_number} - {rally.duration.toFixed(1)}s ({rally.total_strokes} strokes)
              </option>
            ))}
          </select>
        </div>

        {/* Sequential Mode Toggle */}
        <Button
          onClick={() => setSequentialMode(!sequentialMode)}
          className={`${
            sequentialMode
              ? 'bg-primary hover:bg-primary-hover'
              : 'bg-card-bg hover:bg-border'
          }`}
        >
          {sequentialMode ? 'Sequential Mode: ON' : 'Sequential Mode: OFF'}
        </Button>
      </div>

      {/* Video Player */}
      <VideoPlayer videoUrl={videoUrl} rallyNumber={currentRally.rally_number} />

      {/* Navigation Controls */}
      <div className="flex items-center justify-between">
        <Button
          onClick={handlePrevious}
          disabled={currentRallyIndex === 0}
          className="bg-primary hover:bg-primary-hover disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
            <path d="M8.445 14.832A1 1 0 0010 14v-2.798l5.445 3.63A1 1 0 0017 14V6a1 1 0 00-1.555-.832L10 8.798V6a1 1 0 00-1.555-.832l-6 4a1 1 0 000 1.664l6 4z" />
          </svg>
          Previous Rally
        </Button>

        <span className="text-foreground-secondary text-sm">
          Rally {currentRallyIndex + 1} of {data.rallies.length}
        </span>

        <Button
          onClick={handleNext}
          disabled={currentRallyIndex === data.rallies.length - 1}
          className="bg-primary hover:bg-primary-hover disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Next Rally
          <svg className="w-5 h-5 ml-2" fill="currentColor" viewBox="0 0 20 20">
            <path d="M4.555 5.168A1 1 0 003 6v8a1 1 0 001.555.832L10 11.202V14a1 1 0 001.555.832l6-4a1 1 0 000-1.664l-6-4A1 1 0 0010 6v2.798l-5.445-3.63z" />
          </svg>
        </Button>
      </div>

      {/* Rally Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 bg-card-bg border border-border rounded-lg">
        <div>
          <p className="text-foreground-secondary text-sm">Duration</p>
          <p className="text-foreground text-lg font-semibold">{currentRally.duration.toFixed(1)}s</p>
        </div>
        <div>
          <p className="text-foreground-secondary text-sm">Total Strokes</p>
          <p className="text-foreground text-lg font-semibold">{currentRally.total_strokes}</p>
        </div>
        <div>
          <p className="text-foreground-secondary text-sm">Stroke Breakdown</p>
          <p className="text-foreground text-sm">
            P1: {currentRally.player1_strokes.total} | P2: {currentRally.player2_strokes.total}
          </p>
        </div>
      </div>
    </div>
  );
}
