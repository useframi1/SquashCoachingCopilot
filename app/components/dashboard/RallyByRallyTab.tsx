'use client';

import { useState } from 'react';
import { RallyCard } from './RallyCard';
import { RallyTimeline } from './RallyTimeline';
import type { DashboardData } from '@/lib/types';
import { filterRalliesByTimeRange } from '@/lib/utils';

interface RallyByRallyTabProps {
  data: DashboardData;
  onPlayRally?: (rallyNumber: number) => void;
}

export function RallyByRallyTab({ data, onPlayRally }: RallyByRallyTabProps) {
  const [filteredRallies, setFilteredRallies] = useState(data.rallies);

  const handleTimeRangeChange = (startTime: number, endTime: number) => {
    const filtered = filterRalliesByTimeRange(data.rallies, startTime, endTime);
    setFilteredRallies(filtered);
  };

  return (
    <div className="space-y-6">
      {/* Timeline Filter */}
      <RallyTimeline rallies={data.rallies} onTimeRangeChange={handleTimeRangeChange} />

      {/* Rally List */}
      <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2">
        {filteredRallies.length > 0 ? (
          filteredRallies.map((rally) => (
            <RallyCard
              key={rally.rally_number}
              rally={rally}
              onPlayVideo={onPlayRally ? () => onPlayRally(rally.rally_number) : undefined}
            />
          ))
        ) : (
          <div className="text-center py-12 text-foreground-secondary">
            No rallies found in the selected time range.
          </div>
        )}
      </div>
    </div>
  );
}
