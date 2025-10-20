'use client';

import { Activity, Clock, TrendingUp, Zap } from 'lucide-react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { AnalysisResults } from '@/lib/types';
import StatCard from './StatCard';

interface OverviewTabProps {
  results: AnalysisResults;
}

export default function OverviewTab({ results }: OverviewTabProps) {
  // Calculate rally durations
  const rallyDurationData = results.rallies.map((rally, idx) => ({
    name: `${idx + 1}`,
    duration: parseFloat(((rally.end_frame - rally.start_frame) / 30).toFixed(1)),
    frames: rally.end_frame - rally.start_frame,
  }));

  // Find longest rally
  const longestRally = Math.max(...rallyDurationData.map((r) => r.duration));

  // Count stroke types across all rallies (for demo purposes)
  const strokeCounts: Record<string, number> = {};
  let totalStrokes = 0;

  results.rallies.forEach((rally) => {
    rally.rally_frames.forEach((frame) => {
      if (frame.player1.stroke_type) {
        strokeCounts[frame.player1.stroke_type] =
          (strokeCounts[frame.player1.stroke_type] || 0) + 1;
        totalStrokes++;
      }
      if (frame.player2.stroke_type) {
        strokeCounts[frame.player2.stroke_type] =
          (strokeCounts[frame.player2.stroke_type] || 0) + 1;
        totalStrokes++;
      }
    });
  });

  const strokeData = Object.entries(strokeCounts).map(([type, count]) => ({
    type,
    count,
  }));

  // Calculate wall hits per rally
  const wallHitData = results.rallies.map((rally, idx) => {
    const wallHits = rally.rally_frames.filter((frame) => frame.ball.is_wall_hit).length;
    return {
      name: `${idx + 1}`,
      hits: wallHits,
    };
  });

  // Categorize rallies by length
  const rallyLengthCategories = {
    short: rallyDurationData.filter((r) => r.duration < 5).length,
    medium: rallyDurationData.filter((r) => r.duration >= 5 && r.duration < 15).length,
    long: rallyDurationData.filter((r) => r.duration >= 15 && r.duration < 30).length,
    veryLong: rallyDurationData.filter((r) => r.duration >= 30).length,
  };

  const categoryData = [
    { category: 'Short (0-5s)', count: rallyLengthCategories.short },
    { category: 'Medium (5-15s)', count: rallyLengthCategories.medium },
    { category: 'Long (15-30s)', count: rallyLengthCategories.long },
    { category: 'Very Long (30s+)', count: rallyLengthCategories.veryLong },
  ];

  return (
    <div className="space-y-8">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Rallies"
          value={results.total_rallies}
          icon={<Activity className="w-6 h-6" />}
          primary
        />
        <StatCard
          title="Avg Rally Duration"
          value={results.avg_rally_duration.toFixed(1)}
          unit="sec"
          icon={<Clock className="w-6 h-6" />}
        />
        <StatCard
          title="Longest Rally"
          value={longestRally.toFixed(1)}
          unit="sec"
          icon={<Zap className="w-6 h-6" />}
        />
        <StatCard
          title="Total Strokes"
          value={totalStrokes}
          icon={<TrendingUp className="w-6 h-6" />}
          subtitle="Demo only - may be inaccurate"
        />
      </div>

      {/* Charts Row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Rally Duration Timeline */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-2xl font-bold text-black mb-6">Rally Duration Timeline</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={rallyDurationData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#00000010" />
              <XAxis
                dataKey="name"
                stroke="#000000"
                tick={{ fill: '#000000' }}
                label={{ value: 'Rally Number', position: 'insideBottom', offset: -5 }}
              />
              <YAxis
                stroke="#000000"
                tick={{ fill: '#000000' }}
                label={{ value: 'Duration (s)', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#FFFFFF',
                  border: '1px solid #00000020',
                  borderRadius: '8px',
                }}
              />
              <Line
                type="monotone"
                dataKey="duration"
                stroke="#8B0000"
                strokeWidth={3}
                dot={{ fill: '#8B0000', r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Rally Length Distribution */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-2xl font-bold text-black mb-6">Rally Length Distribution</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={categoryData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#00000010" />
              <XAxis
                dataKey="category"
                stroke="#000000"
                tick={{ fill: '#000000', fontSize: 12 }}
              />
              <YAxis stroke="#000000" tick={{ fill: '#000000' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#FFFFFF',
                  border: '1px solid #00000020',
                  borderRadius: '8px',
                }}
              />
              <Bar dataKey="count" fill="#8B0000" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Charts Row 2 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Wall Hit Frequency */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-2xl font-bold text-black mb-6">
            Wall Hit Frequency per Rally
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={wallHitData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#00000010" />
              <XAxis
                dataKey="name"
                stroke="#000000"
                tick={{ fill: '#000000' }}
                label={{ value: 'Rally Number', position: 'insideBottom', offset: -5 }}
              />
              <YAxis
                stroke="#000000"
                tick={{ fill: '#000000' }}
                label={{ value: 'Wall Hits', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#FFFFFF',
                  border: '1px solid #00000020',
                  borderRadius: '8px',
                }}
              />
              <Line
                type="monotone"
                dataKey="hits"
                stroke="#000000"
                strokeWidth={3}
                dot={{ fill: '#000000', r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Stroke Type Distribution (Demo) */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-2xl font-bold text-black mb-6">
            Stroke Type Distribution
            <span className="text-sm font-normal text-black/50 ml-2">(Demo - Inaccurate)</span>
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={strokeData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#00000010" />
              <XAxis dataKey="type" stroke="#000000" tick={{ fill: '#000000' }} />
              <YAxis stroke="#000000" tick={{ fill: '#000000' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#FFFFFF',
                  border: '1px solid #00000020',
                  borderRadius: '8px',
                }}
              />
              <Bar dataKey="count" fill="#000000" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
