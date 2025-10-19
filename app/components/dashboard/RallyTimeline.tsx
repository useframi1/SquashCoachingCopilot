'use client';

import { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import { Slider } from '../ui/Slider';
import type { RallySummary } from '@/lib/types';
import { formatTimestamp } from '@/lib/utils';

interface RallyTimelineProps {
  rallies: RallySummary[];
  onTimeRangeChange?: (startTime: number, endTime: number) => void;
}

export function RallyTimeline({ rallies, onTimeRangeChange }: RallyTimelineProps) {
  const minTime = rallies.length > 0 ? rallies[0].start_time : 0;
  const maxTime = rallies.length > 0 ? rallies[rallies.length - 1].end_time : 100;

  const [timeRange, setTimeRange] = useState<[number, number]>([minTime, maxTime]);

  const handleTimeRangeChange = (value: number[]) => {
    const newRange: [number, number] = [value[0], value[1]];
    setTimeRange(newRange);
    onTimeRangeChange?.(newRange[0], newRange[1]);
  };

  const filteredRalliesCount = rallies.filter(
    (rally) => rally.start_time >= timeRange[0] && rally.end_time <= timeRange[1]
  ).length;

  return (
    <Card className="bg-card-bg border-border">
      <CardHeader>
        <CardTitle className="text-foreground">Rally Timeline</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex justify-between text-sm text-foreground-secondary">
          <span>{formatTimestamp(timeRange[0])}</span>
          <span className="text-foreground font-medium">
            {filteredRalliesCount} / {rallies.length} Rallies
          </span>
          <span>{formatTimestamp(timeRange[1])}</span>
        </div>

        <Slider
          min={minTime}
          max={maxTime}
          step={1}
          value={timeRange}
          onValueChange={handleTimeRangeChange}
          className="w-full"
        />

        <div className="relative h-12 bg-background rounded">
          {rallies.map((rally) => {
            const left = ((rally.start_time - minTime) / (maxTime - minTime)) * 100;
            const width = ((rally.end_time - rally.start_time) / (maxTime - minTime)) * 100;
            const isInRange = rally.start_time >= timeRange[0] && rally.end_time <= timeRange[1];

            return (
              <div
                key={rally.rally_number}
                className={`absolute h-full rounded transition-opacity ${
                  isInRange ? 'bg-primary opacity-100' : 'bg-foreground-secondary opacity-30'
                }`}
                style={{
                  left: `${left}%`,
                  width: `${width}%`,
                }}
                title={`Rally ${rally.rally_number} (${formatTimestamp(rally.start_time)} - ${formatTimestamp(rally.end_time)})`}
              />
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
