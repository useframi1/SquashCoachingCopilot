'use client';

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import type { MovementMetricsPerRallyItem } from '@/types/api';
import { CHART_COLORS } from '@/lib/utils/chart-utils';
import { formatDistance, formatDecimal } from '@/lib/utils/formatters';

interface MovementPerRallyChartProps {
  data: MovementMetricsPerRallyItem[];
  title?: string;
  player1Name?: string;
  player2Name?: string;
}

/**
 * Line chart showing distance covered per rally for both players
 */
export function MovementPerRallyChart({
  data,
  title,
  player1Name = 'Player 1',
  player2Name = 'Player 2'
}: MovementPerRallyChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="h-80 flex items-center justify-center text-gray-500">
        No movement data available
      </div>
    );
  }

  // Transform data for recharts
  const chartData = data.map(rally => ({
    rally_id: rally.rally_id,
    player1Distance: rally.player_1?.total_distance || 0,
    player2Distance: rally.player_2?.total_distance || 0,
    rally_duration: rally.rally_duration,
    shot_count: rally.shot_count,
  }));

  return (
    <div>
      {title && <h4 className="text-md font-semibold text-gray-900 mb-4">{title}</h4>}
      <ResponsiveContainer width="100%" height={350}>
        <LineChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 30 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />

          <XAxis
            dataKey="rally_id"
            label={{ value: 'Rally Number', position: 'insideBottom', offset: -10 }}
            stroke="#6b7280"
            height={50}
          />

          <YAxis
            label={{ value: 'Distance (m)', angle: -90, position: 'insideLeft' }}
            stroke="#6b7280"
            width={80}
          />

          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload || payload.length === 0) return null;

              const data = payload[0].payload;

              return (
                <div className="bg-white p-4 border border-gray-200 rounded-lg shadow-lg">
                  <p className="text-sm font-semibold text-gray-900 mb-3">
                    Rally {data.rally_id}
                  </p>

                  <div className="space-y-2">
                    <div>
                      <p className="text-xs font-semibold mb-1" style={{ color: CHART_COLORS.player1 }}>
                        {player1Name}
                      </p>
                      <div className="space-y-1 text-xs">
                        <p className="text-gray-700">
                          Distance: <span className="font-semibold">{formatDistance(data.player1Distance)}</span>
                        </p>
                      </div>
                    </div>

                    <div>
                      <p className="text-xs font-semibold mb-1" style={{ color: CHART_COLORS.player2 }}>
                        {player2Name}
                      </p>
                      <div className="space-y-1 text-xs">
                        <p className="text-gray-700">
                          Distance: <span className="font-semibold">{formatDistance(data.player2Distance)}</span>
                        </p>
                      </div>
                    </div>

                    <div className="pt-2 mt-2 border-t border-gray-200 text-xs">
                      <p className="text-gray-700">
                        Duration: <span className="font-semibold">{formatDecimal(data.rally_duration, 1)}s</span>
                      </p>
                      <p className="text-gray-700">
                        Shots: <span className="font-semibold">{data.shot_count}</span>
                      </p>
                    </div>
                  </div>
                </div>
              );
            }}
          />

          <Legend wrapperStyle={{ paddingTop: '20px' }} />

          <Line
            type="monotone"
            dataKey="player1Distance"
            name={player1Name}
            stroke={CHART_COLORS.player1}
            strokeWidth={2}
            dot={{ r: 4 }}
            activeDot={{ r: 6 }}
          />

          <Line
            type="monotone"
            dataKey="player2Distance"
            name={player2Name}
            stroke={CHART_COLORS.player2}
            strokeWidth={2}
            dot={{ r: 4 }}
            activeDot={{ r: 6 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
