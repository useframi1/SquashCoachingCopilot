'use client';

import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { RallyTimelineItem } from '@/types/api';
import { getPlayerColor } from '@/lib/utils/chart-utils';
import { formatSpeed } from '@/lib/utils/formatters';

interface RallyScatterChartProps {
  data: RallyTimelineItem[];
  title?: string;
}

/**
 * Scatter chart for rally timeline
 * X: Rally number, Y: Rally duration, Size: Shot count, Color: Winner
 */
export function RallyScatterChart({ data, title }: RallyScatterChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="h-96 flex items-center justify-center text-gray-500">
        No rally data available
      </div>
    );
  }

  // Transform data for scatter chart
  const chartData = data.map(rally => ({
    ...rally,
    x: rally.rally_id,
    y: rally.rally_duration,
    z: rally.shot_count * 3, // Size multiplier for visibility
  }));

  return (
    <div className="space-y-4">
      {title && <h4 className="text-md font-semibold text-gray-900">{title}</h4>}
      <ResponsiveContainer width="100%" height={400}>
        <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />

          <XAxis
            type="number"
            dataKey="x"
            name="Rally"
            label={{ value: 'Rally Number', position: 'insideBottom', offset: -10 }}
            stroke="#6b7280"
          />

          <YAxis
            type="number"
            dataKey="y"
            name="Duration"
            label={{ value: 'Rally Duration (s)', angle: -90, position: 'insideLeft' }}
            stroke="#6b7280"
          />

          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload || payload.length === 0) return null;

              const data = payload[0].payload as RallyTimelineItem;

              return (
                <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                  <p className="text-sm font-semibold text-gray-900 mb-2">
                    Rally {data.rally_id}
                  </p>
                  <div className="space-y-1 text-sm">
                    <p className="text-gray-700">
                      Duration: <span className="font-semibold">{data.rally_duration.toFixed(1)}s</span>
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
                    {data.avg_ball_speed !== null && (
                      <p className="text-gray-700">
                        Avg Speed: <span className="font-semibold">{formatSpeed(data.avg_ball_speed)}</span>
                      </p>
                    )}
                  </div>
                </div>
              );
            }}
          />

          <Scatter data={chartData} shape="circle">
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getPlayerColor(entry.point_winner)} />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-red-700" />
          <span className="text-gray-600">Player 1 won</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-gray-500" />
          <span className="text-gray-600">Player 2 won</span>
        </div>
        <span className="text-gray-500 text-xs">Point size = shot count</span>
      </div>
    </div>
  );
}
