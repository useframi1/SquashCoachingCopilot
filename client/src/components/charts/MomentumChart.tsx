'use client';

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import type { MomentumTimelineItem } from '@/types/api';
import { CHART_COLORS } from '@/lib/utils/chart-utils';

interface MomentumChartProps {
  data: MomentumTimelineItem[];
}

/**
 * Momentum chart showing score differential over time
 * Positive values (red) = Player 1 ahead
 * Negative values (grey) = Player 2 ahead
 */
export function MomentumChart({ data }: MomentumChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="h-96 flex items-center justify-center text-gray-500">
        No momentum data available
      </div>
    );
  }

  return (
    <div role="img" aria-label="Momentum chart showing score differential over rallies">
      <ResponsiveContainer width="100%" height={400}>
        <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="colorDifferential" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={CHART_COLORS.player1} stopOpacity={0.8} />
              <stop offset="95%" stopColor={CHART_COLORS.player1} stopOpacity={0.1} />
            </linearGradient>
          </defs>

        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />

        <XAxis
          dataKey="rally_id"
          label={{ value: 'Rally Number', position: 'insideBottom', offset: -5 }}
          stroke="#6b7280"
        />

        <YAxis
          label={{ value: 'Score Differential', angle: -90, position: 'insideLeft' }}
          stroke="#6b7280"
        />

        <Tooltip
          content={({ active, payload }) => {
            if (!active || !payload || payload.length === 0) return null;

            const data = payload[0].payload as MomentumTimelineItem;

            return (
              <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                <p className="text-sm font-semibold text-gray-900 mb-2">
                  Rally {data.rally_id}
                </p>
                <div className="space-y-1 text-sm">
                  <p className="text-gray-700">
                    Player 1: <span className="font-semibold">{data.player_1_score}</span>
                  </p>
                  <p className="text-gray-700">
                    Player 2: <span className="font-semibold">{data.player_2_score}</span>
                  </p>
                  <p className="text-gray-700">
                    Differential:{' '}
                    <span
                      className={`font-semibold ${
                        data.score_differential > 0 ? 'text-red-700' : 'text-gray-600'
                      }`}
                    >
                      {data.score_differential > 0 ? '+' : ''}
                      {data.score_differential}
                    </span>
                  </p>
                  {data.point_winner && (
                    <p className="text-gray-700">
                      Point winner: <span className="font-semibold">Player {data.point_winner}</span>
                    </p>
                  )}
                </div>
              </div>
            );
          }}
        />

        <ReferenceLine y={0} stroke="#6b7280" strokeDasharray="3 3" />

        <Area
          type="monotone"
          dataKey="score_differential"
          stroke={CHART_COLORS.player1}
          strokeWidth={2}
          fillOpacity={1}
          fill="url(#colorDifferential)"
        />
      </AreaChart>
    </ResponsiveContainer>
    </div>
  );
}
