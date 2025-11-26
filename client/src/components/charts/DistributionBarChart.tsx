'use client';

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import type { DistributionItem } from '@/types/api';
import { CHART_COLORS } from '@/lib/utils/chart-utils';
import { formatShotType, formatPercentage } from '@/lib/utils/formatters';

interface DistributionBarChartProps {
  data: DistributionItem[];
  title?: string;
}

/**
 * Horizontal bar chart for shot type distributions
 */
export function DistributionBarChart({ data, title }: DistributionBarChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="h-80 flex items-center justify-center text-gray-500">
        No data available
      </div>
    );
  }

  // Sort by count descending
  const sortedData = [...data].sort((a, b) => b.count - a.count);

  // Format labels for display
  const formattedData = sortedData.map(item => ({
    ...item,
    displayLabel: formatShotType(item.label),
  }));

  return (
    <div className="space-y-4">
      {title && <h4 className="text-md font-semibold text-gray-900">{title}</h4>}
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={formattedData}
          layout="vertical"
          margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />

          <XAxis type="number" stroke="#6b7280" />

          <YAxis
            dataKey="displayLabel"
            type="category"
            stroke="#6b7280"
            width={110}
            tick={{ fontSize: 12 }}
          />

          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload || payload.length === 0) return null;

              const data = payload[0].payload as DistributionItem & { displayLabel: string };

              return (
                <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                  <p className="text-sm font-semibold text-gray-900 mb-1">
                    {data.displayLabel}
                  </p>
                  <p className="text-sm text-gray-700">
                    Count: <span className="font-semibold">{data.count}</span>
                  </p>
                  <p className="text-sm text-gray-700">
                    Percentage: <span className="font-semibold">{formatPercentage(data.percentage)}</span>
                  </p>
                </div>
              );
            }}
          />

          <Bar dataKey="count" fill={CHART_COLORS.player1} radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
