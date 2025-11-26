'use client';

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import type { RallyTimelineItem } from '@/types/api';
import { CHART_COLORS } from '@/lib/utils/chart-utils';
import { formatSpeed } from '@/lib/utils/formatters';

interface BallSpeedLineChartProps {
  data: RallyTimelineItem[];
  title?: string;
}

/**
 * Line chart showing ball speed over rallies
 * Shows fatigue/pace changes throughout the match
 */
export function BallSpeedLineChart({ data, title }: BallSpeedLineChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="h-80 flex items-center justify-center text-gray-500">
        No data available
      </div>
    );
  }

  // Filter out rallies with no ball speed data
  const chartData = data.filter(rally => rally.avg_ball_speed !== null);

  if (chartData.length === 0) {
    return (
      <div className="h-80 flex items-center justify-center text-gray-500">
        No ball speed data available
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {title && <h4 className="text-md font-semibold text-gray-900">{title}</h4>}
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />

          <XAxis
            dataKey="rally_id"
            label={{ value: 'Rally Number', position: 'insideBottom', offset: -5 }}
            stroke="#6b7280"
          />

          <YAxis
            label={{ value: 'Ball Speed (m/s)', angle: -90, position: 'insideLeft' }}
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
                      Avg Speed:{' '}
                      <span className="font-semibold">
                        {data.avg_ball_speed !== null ? formatSpeed(data.avg_ball_speed) : 'N/A'}
                      </span>
                    </p>
                    <p className="text-gray-700">
                      Shots: <span className="font-semibold">{data.shot_count}</span>
                    </p>
                    <p className="text-gray-700">
                      Duration: <span className="font-semibold">{data.rally_duration.toFixed(1)}s</span>
                    </p>
                  </div>
                </div>
              );
            }}
          />

          <Line
            type="monotone"
            dataKey="avg_ball_speed"
            stroke={CHART_COLORS.player1}
            strokeWidth={2}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
