'use client';

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import type { PlayerStats } from '@/lib/types';

interface StrokeDistributionChartProps {
  player1Stats: PlayerStats;
  player2Stats: PlayerStats;
}

export function StrokeDistributionChart({ player1Stats, player2Stats }: StrokeDistributionChartProps) {
  const chartData = [
    {
      name: 'Forehand',
      Player1: player1Stats.strokes.forehand,
      Player2: player2Stats.strokes.forehand,
    },
    {
      name: 'Backhand',
      Player1: player1Stats.strokes.backhand,
      Player2: player2Stats.strokes.backhand,
    },
  ];

  return (
    <Card className="bg-card-bg border-border">
      <CardHeader>
        <CardTitle className="text-foreground">Stroke Distribution by Player</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2D2D2D" />
            <XAxis
              dataKey="name"
              stroke="#9CA3AF"
              tick={{ fill: '#9CA3AF', fontSize: 12 }}
            />
            <YAxis
              stroke="#9CA3AF"
              tick={{ fill: '#9CA3AF', fontSize: 12 }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1A1A1A',
                border: '1px solid #2D2D2D',
                borderRadius: '8px',
                color: '#FFFFFF',
              }}
              labelStyle={{ color: '#9CA3AF' }}
            />
            <Legend
              wrapperStyle={{ color: '#9CA3AF' }}
              iconType="rect"
            />
            <Bar dataKey="Player1" fill="#8B1538" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Player2" fill="#DC2626" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
