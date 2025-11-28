'use client';

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import type { BallSpeedPerRallyItem } from '@/types/api';
import { CHART_COLORS } from '@/lib/utils/chart-utils';
import { formatSpeed } from '@/lib/utils/formatters';

interface DualPlayerBallSpeedChartProps {
  data: BallSpeedPerRallyItem[];
  title?: string;
  player1Name?: string;
  player2Name?: string;
}

/**
 * Dual-player line chart showing ball speed over rallies
 * Displays both Player 1 and Player 2 ball speed trends
 */
export function DualPlayerBallSpeedChart({ data, title, player1Name = 'Player 1', player2Name = 'Player 2' }: DualPlayerBallSpeedChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="h-80 flex items-center justify-center text-gray-500">
        No data available
      </div>
    );
  }

  // Transform data for recharts
  const chartData = data.map(rally => ({
    rally_id: rally.rally_id,
    player1Speed: rally.player_1.mean_speed,
    player2Speed: rally.player_2.mean_speed,
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
            label={{ value: 'Rally Number', position: 'insideBottom', offset: -10, style: { textAnchor: 'middle' } }}
            stroke="#6b7280"
            height={50}
          />

          <YAxis
            label={{ value: 'Ball Speed (m/s)', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' } }}
            stroke="#6b7280"
            width={60}
          />

          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload || payload.length === 0) return null;

              const data = payload[0].payload;

              return (
                <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                  <p className="text-sm font-semibold text-gray-900 mb-2">
                    Rally {data.rally_id}
                  </p>
                  <div className="space-y-1 text-sm">
                    <p className="text-gray-700">
                      {player1Name}:{' '}
                      <span className="font-semibold" style={{ color: CHART_COLORS.player1 }}>
                        {data.player1Speed !== null ? formatSpeed(data.player1Speed) : 'N/A'}
                      </span>
                    </p>
                    <p className="text-gray-700">
                      {player2Name}:{' '}
                      <span className="font-semibold" style={{ color: CHART_COLORS.player2 }}>
                        {data.player2Speed !== null ? formatSpeed(data.player2Speed) : 'N/A'}
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

          <Legend
            wrapperStyle={{ paddingTop: '20px' }}
            iconType="line"
          />

          <Line
            type="monotone"
            dataKey="player1Speed"
            name={player1Name}
            stroke={CHART_COLORS.player1}
            strokeWidth={2}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
            connectNulls
          />

          <Line
            type="monotone"
            dataKey="player2Speed"
            name={player2Name}
            stroke={CHART_COLORS.player2}
            strokeWidth={2}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
            connectNulls
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
