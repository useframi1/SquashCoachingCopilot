'use client';

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import type { TZoneOccupancyPerRallyItem } from '@/types/api';
import { CHART_COLORS } from '@/lib/utils/chart-utils';
import { formatPercentage, formatDecimal } from '@/lib/utils/formatters';

interface DualPlayerTZoneOccupancyChartProps {
  data: TZoneOccupancyPerRallyItem[];
  title?: string;
  player1Name?: string;
  player2Name?: string;
}

/**
 * Dual-player bar chart showing T-zone occupancy (% time in T) over rallies
 * Displays both Player 1 and Player 2 T-zone metrics
 */
export function DualPlayerTZoneOccupancyChart({ data, title, player1Name = 'Player 1', player2Name = 'Player 2' }: DualPlayerTZoneOccupancyChartProps) {
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
    player1PctTimeInT: rally.player_1.pct_time_in_t,
    player2PctTimeInT: rally.player_2.pct_time_in_t,
    rally_duration: rally.rally_duration,
    shot_count: rally.shot_count,
    // Additional metrics for tooltip
    player1_avg_time_to_t: rally.player_1.avg_time_to_t,
    player2_avg_time_to_t: rally.player_2.avg_time_to_t,
    player1_t_zone_success_rate: rally.player_1.t_zone_success_rate,
    player2_t_zone_success_rate: rally.player_2.t_zone_success_rate,
    player1_total_shots: rally.player_1.total_shots_taken,
    player2_total_shots: rally.player_2.total_shots_taken,
    player1_successful_returns: rally.player_1.successful_returns,
    player2_successful_returns: rally.player_2.successful_returns,
  }));

  return (
    <div>
      {title && <h4 className="text-md font-semibold text-gray-900 mb-4">{title}</h4>}
      <ResponsiveContainer width="100%" height={350}>
        <BarChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 30 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />

          <XAxis
            dataKey="rally_id"
            label={{ value: 'Rally Number', position: 'insideBottom', offset: -10, style: { textAnchor: 'middle' } }}
            stroke="#6b7280"
            height={50}
          />

          <YAxis
            label={{ value: '% Time in T-Zone', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' } }}
            stroke="#6b7280"
            width={80}
          />

          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload || payload.length === 0) return null;

              const data = payload[0].payload;

              return (
                <div className="bg-white p-4 border border-gray-200 rounded-lg shadow-lg max-w-sm">
                  <p className="text-sm font-semibold text-gray-900 mb-3">
                    Rally {data.rally_id}
                  </p>

                  {/* Player 1 Metrics */}
                  <div className="mb-3">
                    <p className="text-xs font-semibold mb-1" style={{ color: CHART_COLORS.player1 }}>
                      {player1Name}
                    </p>
                    <div className="space-y-1 text-xs">
                      <p className="text-gray-700">
                        Time in T:{' '}
                        <span className="font-semibold">
                          {data.player1PctTimeInT !== null ? formatPercentage(data.player1PctTimeInT) : 'N/A'}
                        </span>
                      </p>
                      <p className="text-gray-700">
                        Avg Time to T:{' '}
                        <span className="font-semibold">
                          {data.player1_avg_time_to_t !== null ? `${formatDecimal(data.player1_avg_time_to_t, 2)}s` : 'N/A'}
                        </span>
                      </p>
                      <p className="text-gray-700">
                        T-Zone Success Rate:{' '}
                        <span className="font-semibold">
                          {data.player1_t_zone_success_rate !== null ? formatPercentage(data.player1_t_zone_success_rate) : 'N/A'}
                        </span>
                      </p>
                      <p className="text-gray-700">
                        Shots Taken:{' '}
                        <span className="font-semibold">
                          {data.player1_total_shots}
                        </span>
                      </p>
                      <p className="text-gray-700">
                        Successful Returns:{' '}
                        <span className="font-semibold">
                          {data.player1_successful_returns}
                        </span>
                      </p>
                    </div>
                  </div>

                  {/* Player 2 Metrics */}
                  <div className="mb-2">
                    <p className="text-xs font-semibold mb-1" style={{ color: CHART_COLORS.player2 }}>
                      {player2Name}
                    </p>
                    <div className="space-y-1 text-xs">
                      <p className="text-gray-700">
                        Time in T:{' '}
                        <span className="font-semibold">
                          {data.player2PctTimeInT !== null ? formatPercentage(data.player2PctTimeInT) : 'N/A'}
                        </span>
                      </p>
                      <p className="text-gray-700">
                        Avg Time to T:{' '}
                        <span className="font-semibold">
                          {data.player2_avg_time_to_t !== null ? `${formatDecimal(data.player2_avg_time_to_t, 2)}s` : 'N/A'}
                        </span>
                      </p>
                      <p className="text-gray-700">
                        T-Zone Success Rate:{' '}
                        <span className="font-semibold">
                          {data.player2_t_zone_success_rate !== null ? formatPercentage(data.player2_t_zone_success_rate) : 'N/A'}
                        </span>
                      </p>
                      <p className="text-gray-700">
                        Shots Taken:{' '}
                        <span className="font-semibold">
                          {data.player2_total_shots}
                        </span>
                      </p>
                      <p className="text-gray-700">
                        Successful Returns:{' '}
                        <span className="font-semibold">
                          {data.player2_successful_returns}
                        </span>
                      </p>
                    </div>
                  </div>

                  {/* Rally Info */}
                  <div className="pt-2 mt-2 border-t border-gray-200 space-y-1 text-xs">
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
            iconType="rect"
          />

          <Bar
            dataKey="player1PctTimeInT"
            name={player1Name}
            fill={CHART_COLORS.player1}
            radius={[4, 4, 0, 0]}
          />

          <Bar
            dataKey="player2PctTimeInT"
            name={player2Name}
            fill={CHART_COLORS.player2}
            radius={[4, 4, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
