'use client';

import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import { CHART_COLORS } from '@/lib/utils/chart-utils';

interface ShotTypeRadarChartProps {
  data: Array<{
    label: string;
    count: number;
    percentage: number;
  }>;
  title: string;
}

/**
 * Radar/Spider chart for shot type distribution
 */
export function ShotTypeRadarChart({ data, title }: ShotTypeRadarChartProps) {
  // Transform data for radar chart
  const radarData = data.map(item => ({
    shotType: item.label,
    value: item.percentage,
  }));

  return (
    <div>
      {title && <h4 className="text-md font-semibold text-gray-900 mb-4">{title}</h4>}
      <ResponsiveContainer width="100%" height={350}>
        <RadarChart data={radarData}>
          <PolarGrid stroke="#e5e7eb" />
          <PolarAngleAxis
            dataKey="shotType"
            tick={{ fill: '#6b7280', fontSize: 12 }}
          />
          <PolarRadiusAxis
            angle={90}
            domain={[0, 100]}
            tick={{ fill: '#6b7280', fontSize: 11 }}
          />
          <Radar
            name="Shot Type %"
            dataKey="value"
            stroke={CHART_COLORS.player1}
            fill={CHART_COLORS.player1}
            fillOpacity={0.5}
          />
          <Tooltip
            formatter={(value: number) => `${value.toFixed(1)}%`}
            contentStyle={{
              backgroundColor: 'white',
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
            }}
          />
          <Legend />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
