'use client';

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { RallyTimelineItem } from '@/types/api';
import { getPlayerColor } from '@/lib/utils/chart-utils';
import { formatDecimal } from '@/lib/utils/formatters';

interface RallyDurationBarChartProps {
  data: RallyTimelineItem[];
  title?: string;
}

/**
 * Bar chart showing rally intensity (shots per second) with color-coded winners
 * Higher values = faster/more intense rallies
 */
export function RallyDurationBarChart({ data, title }: RallyDurationBarChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="h-80 flex items-center justify-center text-gray-500">
        No rally data available
      </div>
    );
  }

  // Transform data for bar chart - calculate intensity (shots per second)
  const chartData = data.map(rally => {
    const intensity = rally.rally_duration > 0 ? rally.shot_count / rally.rally_duration : 0;
    return {
      ...rally,
      rally_id: rally.rally_id,
      intensity: intensity,
    };
  });

  return (
    <div>
      {title && <h4 className="text-md font-semibold text-gray-900 mb-4">{title}</h4>}
      <ResponsiveContainer width="100%" height={350}>
        <BarChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 30 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />

          <XAxis
            dataKey="rally_id"
            label={{ value: 'Rally Number', position: 'insideBottom', offset: -10 }}
            stroke="#6b7280"
            height={50}
          />

          <YAxis
            label={{ value: 'Intensity (shots/s)', angle: -90, position: 'insideLeft' }}
            stroke="#6b7280"
            width={80}
          />

          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload || payload.length === 0) return null;

              const data = payload[0].payload as RallyTimelineItem & { intensity: number };

              return (
                <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                  <p className="text-sm font-semibold text-gray-900 mb-2">
                    Rally {data.rally_id}
                  </p>
                  <div className="space-y-1 text-sm">
                    <p className="text-gray-700">
                      Intensity: <span className="font-semibold">{formatDecimal(data.intensity, 2)} shots/s</span>
                    </p>
                    <p className="text-gray-700">
                      Duration: <span className="font-semibold">{formatDecimal(data.rally_duration, 1)}s</span>
                    </p>
                    <p className="text-gray-700">
                      Shots: <span className="font-semibold">{data.shot_count}</span>
                    </p>
                    {data.point_winner && (
                      <p className="text-gray-700">
                        Winner:{' '}
                        <span
                          className="font-semibold"
                          style={{ color: getPlayerColor(data.point_winner) }}
                        >
                          Player {data.point_winner}
                        </span>
                      </p>
                    )}
                  </div>
                </div>
              );
            }}
          />

          <Bar dataKey="intensity" radius={[4, 4, 0, 0]}>
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getPlayerColor(entry.point_winner)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 text-sm mt-4">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-sm bg-red-700" />
          <span className="text-gray-600">Player 1 won</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-sm bg-gray-500" />
          <span className="text-gray-600">Player 2 won</span>
        </div>
        <div className="text-gray-500 text-xs ml-4">Higher = faster rally</div>
      </div>
    </div>
  );
}
