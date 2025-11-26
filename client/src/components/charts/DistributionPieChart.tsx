'use client';

import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import type { DistributionItem } from '@/types/api';
import { CHART_COLORS } from '@/lib/utils/chart-utils';
import { formatPercentage } from '@/lib/utils/formatters';

interface DistributionPieChartProps {
  data: DistributionItem[];
  title?: string;
}

const COLORS = [CHART_COLORS.player1, CHART_COLORS.player2, '#f87171', '#fca5a5', '#fee2e2'];

/**
 * Pie chart for distributions (stroke types, quadrants, etc.)
 */
export function DistributionPieChart({ data, title }: DistributionPieChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="h-80 flex items-center justify-center text-gray-500">
        No data available
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {title && <h4 className="text-md font-semibold text-gray-900">{title}</h4>}
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ name, percentage }) => `${name}: ${formatPercentage(percentage)}`}
            outerRadius={100}
            fill="#8884d8"
            dataKey="count"
            nameKey="label"
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload || payload.length === 0) return null;

              const data = payload[0].payload as DistributionItem;

              return (
                <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                  <p className="text-sm font-semibold text-gray-900 mb-1 capitalize">
                    {data.label.replace(/_/g, ' ')}
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
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
