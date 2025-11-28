'use client';

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { formatDecimal } from '@/lib/utils/formatters';
import { CHART_COLORS } from '@/lib/utils/chart-utils';

interface WinningEfficiencyData {
  rally_id: number;
  rally_duration: number;
  shot_count: number;
  player_1?: {
    shots_per_point_won: number;
    points_won: number;
    total_shots: number;
    win_rate: number;
    rallies_played: number;
  };
  player_2?: {
    shots_per_point_won: number;
    points_won: number;
    total_shots: number;
    win_rate: number;
    rallies_played: number;
  };
}

interface WinningEfficiencyBarChartProps {
  data: WinningEfficiencyData[];
  title?: string;
  player1Name?: string;
  player2Name?: string;
}

/**
 * Bar chart showing shots taken per rally for both players
 * Highlights the rally winner with brighter color
 */
export function WinningEfficiencyBarChart({
  data,
  title = 'Shots Taken per Rally',
  player1Name = 'Player 1',
  player2Name = 'Player 2',
}: WinningEfficiencyBarChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="h-80 flex items-center justify-center text-gray-500">
        No winning efficiency data available
      </div>
    );
  }

  // Transform data for recharts
  const chartData = data.map(rally => ({
    rally_id: rally.rally_id,
    player1Shots: rally.player_1?.total_shots || 0,
    player2Shots: rally.player_2?.total_shots || 0,
    player1Won: rally.player_1?.points_won === 1,
    player2Won: rally.player_2?.points_won === 1,
    player1ShotsPerPoint: rally.player_1?.shots_per_point_won || 0,
    player2ShotsPerPoint: rally.player_2?.shots_per_point_won || 0,
    rally_duration: rally.rally_duration,
    shot_count: rally.shot_count,
  }));

  return (
    <div>
      {title && <h4 className="text-md font-semibold text-gray-900 mb-4">{title}</h4>}
      <ResponsiveContainer width="100%" height={350}>
        <BarChart
          data={chartData}
          margin={{ top: 5, right: 30, left: 20, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="rally_id"
            label={{ value: 'Rally #', position: 'insideBottom', offset: -10 }}
            stroke="#6b7280"
            style={{ fontSize: '12px' }}
          />
          <YAxis
            label={{ value: 'Shots Taken', angle: -90, position: 'insideLeft' }}
            stroke="#6b7280"
            style={{ fontSize: '12px' }}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const data = payload[0].payload;
                const p1Won = data.player1Won;
                const p2Won = data.player2Won;

                return (
                  <div className="bg-white p-4 rounded-lg border border-gray-200 shadow-lg">
                    <p className="text-sm font-semibold text-gray-900 mb-2">
                      Rally #{data.rally_id}
                    </p>

                    <div>
                      <p className="text-xs font-semibold mb-1 flex items-center gap-2" style={{ color: CHART_COLORS.player1 }}>
                        {player1Name}
                        {p1Won && <span className="text-green-600 font-bold">✓ WON</span>}
                      </p>
                      <div className="space-y-1 text-xs">
                        <p className="text-gray-700">
                          Shots: <span className="font-semibold">{data.player1Shots}</span>
                        </p>
                        {p1Won && (
                          <p className="text-gray-700">
                            Efficiency: <span className="font-semibold">{formatDecimal(data.player1ShotsPerPoint, 1)} shots/point</span>
                          </p>
                        )}
                      </div>
                    </div>

                    <div className="mt-2">
                      <p className="text-xs font-semibold mb-1 flex items-center gap-2" style={{ color: CHART_COLORS.player2 }}>
                        {player2Name}
                        {p2Won && <span className="text-green-600 font-bold">✓ WON</span>}
                      </p>
                      <div className="space-y-1 text-xs">
                        <p className="text-gray-700">
                          Shots: <span className="font-semibold">{data.player2Shots}</span>
                        </p>
                        {p2Won && (
                          <p className="text-gray-700">
                            Efficiency: <span className="font-semibold">{formatDecimal(data.player2ShotsPerPoint, 1)} shots/point</span>
                          </p>
                        )}
                      </div>
                    </div>

                    <div className="pt-2 mt-2 border-t border-gray-200 text-xs">
                      <p className="text-gray-700">
                        Duration: <span className="font-semibold">{formatDecimal(data.rally_duration, 1)}s</span>
                      </p>
                      <p className="text-gray-700">
                        Total Shots: <span className="font-semibold">{data.shot_count}</span>
                      </p>
                    </div>
                  </div>
                );
              }
              return null;
            }}
          />
          <Legend
            wrapperStyle={{ paddingTop: '20px' }}
            formatter={(value) => {
              if (value === 'player1Shots') return player1Name;
              if (value === 'player2Shots') return player2Name;
              return value;
            }}
          />
          <Bar
            dataKey="player1Shots"
            name="player1Shots"
            radius={[4, 4, 0, 0]}
          >
            {chartData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.player1Won ? CHART_COLORS.player1 : `${CHART_COLORS.player1}80`}
              />
            ))}
          </Bar>
          <Bar
            dataKey="player2Shots"
            name="player2Shots"
            radius={[4, 4, 0, 0]}
          >
            {chartData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.player2Won ? CHART_COLORS.player2 : `${CHART_COLORS.player2}80`}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
