'use client';

import { useState } from 'react';
import { StatsCard } from './StatsCard';
import { RallyDurationChart } from './RallyDurationChart';
import { StrokeDistributionChart } from './StrokeDistributionChart';
import { RallyTimeline } from './RallyTimeline';
import type { DashboardData } from '@/lib/types';
import { formatDuration, filterRalliesByTimeRange } from '@/lib/utils';

interface OverviewTabProps {
  data: DashboardData;
}

export function OverviewTab({ data }: OverviewTabProps) {
  const [filteredRallies, setFilteredRallies] = useState(data.rallies);

  const handleTimeRangeChange = (startTime: number, endTime: number) => {
    const filtered = filterRalliesByTimeRange(data.rallies, startTime, endTime);
    setFilteredRallies(filtered);
  };

  // Calculate stats for filtered rallies
  const avgDuration = filteredRallies.length > 0
    ? filteredRallies.reduce((sum, r) => sum + r.duration, 0) / filteredRallies.length
    : 0;

  const longestFiltered = filteredRallies.length > 0
    ? filteredRallies.reduce((longest, current) =>
        current.duration > longest.duration ? current : longest
      )
    : data.longestRally;

  return (
    <div className="space-y-6">
      {/* Timeline Filter */}
      <RallyTimeline rallies={data.rallies} onTimeRangeChange={handleTimeRangeChange} />

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatsCard
          title="Total Rallies"
          value={filteredRallies.length}
          subtitle={`Out of ${data.totalRallies} total rallies`}
        />
        <StatsCard
          title="Average Rally Duration"
          value={formatDuration(avgDuration)}
          subtitle="Mean duration of all rallies"
        />
        <StatsCard
          title="Longest Rally"
          value={formatDuration(longestFiltered.duration)}
          subtitle={`Rally #${longestFiltered.rally_number}`}
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <RallyDurationChart rallies={filteredRallies} />
        <StrokeDistributionChart
          player1Stats={data.player1Stats}
          player2Stats={data.player2Stats}
        />
      </div>

      {/* Additional Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <StatsCard
          title="Player 1 Total Strokes"
          value={data.player1Stats.strokes.total}
          subtitle={`${data.player1Stats.strokes.forehand} Forehand, ${data.player1Stats.strokes.backhand} Backhand`}
        />
        <StatsCard
          title="Player 2 Total Strokes"
          value={data.player2Stats.strokes.total}
          subtitle={`${data.player2Stats.strokes.forehand} Forehand, ${data.player2Stats.strokes.backhand} Backhand`}
        />
      </div>
    </div>
  );
}
