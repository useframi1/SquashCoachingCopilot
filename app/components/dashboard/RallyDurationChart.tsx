'use client';

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import type { RallySummary } from '@/lib/types';
import { formatDuration } from '@/lib/utils';

interface RallyDurationChartProps {
  rallies: RallySummary[];
}

export function RallyDurationChart({ rallies }: RallyDurationChartProps) {
  const chartData = rallies.map((rally) => ({
    rally: `R${rally.rally_number}`,
    duration: rally.duration,
  }));

  return (
    <Card className="bg-card-bg border-border">
      <CardHeader>
        <CardTitle className="text-foreground">Rally Duration Over Time</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2D2D2D" />
            <XAxis
              dataKey="rally"
              stroke="#9CA3AF"
              tick={{ fill: '#9CA3AF', fontSize: 12 }}
            />
            <YAxis
              stroke="#9CA3AF"
              tick={{ fill: '#9CA3AF', fontSize: 12 }}
              tickFormatter={(value) => formatDuration(value)}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1A1A1A',
                border: '1px solid #2D2D2D',
                borderRadius: '8px',
                color: '#FFFFFF',
              }}
              labelStyle={{ color: '#9CA3AF' }}
              formatter={(value: number) => [formatDuration(value), 'Duration']}
            />
            <Line
              type="monotone"
              dataKey="duration"
              stroke="#8B1538"
              strokeWidth={2}
              dot={{ fill: '#DC2626', strokeWidth: 2, r: 4 }}
              activeDot={{ r: 6, fill: '#DC2626' }}
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
