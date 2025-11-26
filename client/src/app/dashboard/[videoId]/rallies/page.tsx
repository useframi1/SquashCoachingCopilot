'use client';

import { useParams } from 'next/navigation';
import { Activity, Clock, Zap, Trophy } from 'lucide-react';
import { KPICard } from '@/components/dashboard/KPICard';
import { RallyScatterChart } from '@/components/charts/RallyScatterChart';
import {
  useRallyTimeline,
  useRallyIntensity,
  useWinningStats,
} from '@/lib/hooks/useAnalytics';
import { formatDecimal, formatPercentage } from '@/lib/utils/formatters';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { CHART_COLORS } from '@/lib/utils/chart-utils';


/**
 * Rally Analysis tab
 * Shows rally-by-rally breakdown, intensity, and winning patterns
 */
export default function RalliesPage() {
  const params = useParams();
  const videoId = params.videoId as string;

  const { data: rallyTimeline, isLoading: timelineLoading } = useRallyTimeline(videoId);
  const { data: intensity, isLoading: intensityLoading } = useRallyIntensity(videoId);
  const { data: winningP1, isLoading: winningP1Loading } = useWinningStats(videoId, 1);
  const { data: winningP2, isLoading: winningP2Loading } = useWinningStats(videoId, 2);

  const isLoading = timelineLoading || intensityLoading || winningP1Loading || winningP2Loading;

  if (isLoading) {
    return (
      <div className="p-8 space-y-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="h-32 bg-gray-200 rounded-lg animate-pulse" />
          ))}
        </div>
        <div className="h-96 bg-gray-200 rounded-lg animate-pulse" />
      </div>
    );
  }

  // Calculate average rally duration from timeline
  const avgRallyDuration =
    (rallyTimeline?.data.reduce((sum, rally) => sum + rally.rally_duration, 0) ?? 0) /
      (rallyTimeline?.data.length || 1);

  // Prepare data for winning stats comparison
  const winningComparisonData = [
    {
      name: 'Efficiency',
      player1: winningP1?.data.efficiency || 0,
      player2: winningP2?.data.efficiency || 0,
    },
    {
      name: 'Points Won',
      player1: winningP1?.data.points_won || 0,
      player2: winningP2?.data.points_won || 0,
    },
  ];

  return (
    <div className="p-8 space-y-8">
      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        <KPICard
          title="Total Rallies"
          value={rallyTimeline?.total_rallies || 0}
          icon={Activity}
          subtitle="Match rallies"
        />

        <KPICard
          title="Avg Rally Duration"
          value={formatDecimal(avgRallyDuration, 1) + 's'}
          icon={Clock}
          subtitle={
            rallyTimeline?.data
              ? `${rallyTimeline.data.length} rallies analyzed`
              : undefined
          }
        />

        <KPICard
          title="Rally Intensity"
          value={
            intensity?.data.avg_seconds_per_shot
              ? formatDecimal(intensity.data.avg_seconds_per_shot, 2) + 's/shot'
              : 'N/A'
          }
          icon={Zap}
          subtitle="Average pace"
        />
      </div>

      {/* Rally Timeline Visualization */}
      <div className="bg-white p-6 rounded-lg border border-gray-200">
        <RallyScatterChart
          data={rallyTimeline?.data || []}
          title="Rally Timeline"
        />
      </div>

      {/* Winning Stats Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Grouped Bar Chart */}
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Winning Stats Comparison
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={winningComparisonData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="name" stroke="#6b7280" />
              <YAxis stroke="#6b7280" />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'white',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                }}
              />
              <Legend />
              <Bar dataKey="player1" name="Player 1" fill={CHART_COLORS.player1} />
              <Bar dataKey="player2" name="Player 2" fill={CHART_COLORS.player2} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Detailed Stats */}
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Efficiency Breakdown</h3>
          <div className="space-y-6">
            {/* Player 1 */}
            <div>
              <p className="text-sm font-medium text-red-700 mb-2">Player 1</p>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-gray-600">Efficiency</p>
                  <p className="text-xl font-bold text-gray-900">
                    {winningP1?.data.efficiency
                      ? formatPercentage(winningP1.data.efficiency * 100)
                      : 'N/A'}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-600">Points Won</p>
                  <p className="text-xl font-bold text-gray-900">
                    {winningP1?.data.points_won || 0}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-600">Total Shots</p>
                  <p className="text-xl font-bold text-gray-900">
                    {winningP1?.data.total_shots || 0}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-600">Points/Rally</p>
                  <p className="text-xl font-bold text-gray-900">
                    {winningP1?.data.points_per_rally
                      ? formatDecimal(winningP1.data.points_per_rally, 2)
                      : 'N/A'}
                  </p>
                </div>
              </div>
            </div>

            {/* Player 2 */}
            <div>
              <p className="text-sm font-medium text-gray-600 mb-2">Player 2</p>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-gray-600">Efficiency</p>
                  <p className="text-xl font-bold text-gray-900">
                    {winningP2?.data.efficiency
                      ? formatPercentage(winningP2.data.efficiency * 100)
                      : 'N/A'}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-600">Points Won</p>
                  <p className="text-xl font-bold text-gray-900">
                    {winningP2?.data.points_won || 0}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-600">Total Shots</p>
                  <p className="text-xl font-bold text-gray-900">
                    {winningP2?.data.total_shots || 0}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-600">Points/Rally</p>
                  <p className="text-xl font-bold text-gray-900">
                    {winningP2?.data.points_per_rally
                      ? formatDecimal(winningP2.data.points_per_rally, 2)
                      : 'N/A'}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Rally Intensity Details */}
      {intensity && (
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Rally Intensity Metrics</h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-6">
            <div>
              <p className="text-sm text-gray-600">Avg Seconds/Shot</p>
              <p className="text-2xl font-bold text-gray-900">
                {formatDecimal(intensity.data.avg_seconds_per_shot, 2)}s
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Fastest Rally</p>
              <p className="text-2xl font-bold text-gray-900">
                {formatDecimal(intensity.data.min_seconds_per_shot, 2)}s/shot
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Slowest Rally</p>
              <p className="text-2xl font-bold text-gray-900">
                {formatDecimal(intensity.data.max_seconds_per_shot, 2)}s/shot
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Rallies Analyzed</p>
              <p className="text-2xl font-bold text-gray-900">
                {intensity.data.rally_count}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
