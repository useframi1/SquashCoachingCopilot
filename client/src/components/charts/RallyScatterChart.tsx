'use client';

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import type { RallyTimelineItem } from '@/types/api';
import { getPlayerColor } from '@/lib/utils/chart-utils';

interface RallyScatterChartProps {
  data: RallyTimelineItem[];
  title?: string;
}

/**
 * Custom dot component with variable size based on shot count
 */
const CustomDot = (props: any) => {
  const { cx, cy, payload } = props;
  const radius = Math.max(3, Math.min(12, payload.shot_count * 0.8)); // Scale radius based on shot count
  const color = getPlayerColor(payload.point_winner);

  return (
    <circle
      cx={cx}
      cy={cy}
      r={radius}
      fill={color}
      stroke={color}
      strokeWidth={1}
    />
  );
};

/**
 * Line chart for rally timeline with variable-sized dots
 * X: Rally number, Y: Rally duration, Dot size: Shot count, Color: Winner
 */
export function RallyScatterChart({ data, title }: RallyScatterChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="h-96 flex items-center justify-center text-gray-500">
        No rally data available
      </div>
    );
  }

  // Transform data for line chart
  const chartData = data.map(rally => ({
    ...rally,
    rally_number: rally.rally_id,
    duration: rally.rally_duration,
  }));

  return (
    <div className="space-y-4">
      {title && <h4 className="text-md font-semibold text-gray-900">{title}</h4>}
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData} margin={{ top: 20, right: 30, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />

          <XAxis
            dataKey="rally_number"
            name="Rally"
            label={{ value: 'Rally Number', position: 'insideBottom', offset: -10 }}
            stroke="#6b7280"
          />

          <YAxis
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
                  </div>
                </div>
              );
            }}
          />

          <Line
            type="monotone"
            dataKey="duration"
            stroke="#9ca3af"
            strokeWidth={2}
            dot={<CustomDot />}
            activeDot={{ r: 8 }}
          />
        </LineChart>
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
