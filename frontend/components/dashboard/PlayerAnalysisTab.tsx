'use client';

import { useState, useMemo } from 'react';
import { Target, TrendingUp, Zap, MapPin } from 'lucide-react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { AnalysisResults } from '@/lib/types';
import StatCard from './StatCard';
import CourtHeatmap from './CourtHeatmap';
import {
  extractAllPlayerPositions,
  extractAllPlayerPositionsWithTime,
  calculateTDominance,
  calculateAggressiveness,
  calculateCourtCoverage,
  getZoneDistribution,
  createHeatmapData,
  getTDominanceTimelineWithTime,
  getAggressivenessTimelineWithTime,
} from '@/lib/courtMetrics';

interface PlayerAnalysisTabProps {
  results: AnalysisResults;
}

export default function PlayerAnalysisTab({ results }: PlayerAnalysisTabProps) {
  const [selectedPlayer, setSelectedPlayer] = useState<1 | 2>(1);

  // Extract positions for both players
  const player1Positions = useMemo(
    () => extractAllPlayerPositions(results.rallies, 1),
    [results.rallies]
  );
  const player2Positions = useMemo(
    () => extractAllPlayerPositions(results.rallies, 2),
    [results.rallies]
  );

  // Extract positions with timestamps for timeline charts
  const player1PositionsWithTime = useMemo(
    () => extractAllPlayerPositionsWithTime(results.rallies, 1),
    [results.rallies]
  );
  const player2PositionsWithTime = useMemo(
    () => extractAllPlayerPositionsWithTime(results.rallies, 2),
    [results.rallies]
  );

  const currentPlayerPositions = selectedPlayer === 1 ? player1Positions : player2Positions;
  const currentPlayerPositionsWithTime = selectedPlayer === 1 ? player1PositionsWithTime : player2PositionsWithTime;

  // Calculate metrics for selected player
  const tDominance = useMemo(
    () => calculateTDominance(currentPlayerPositions),
    [currentPlayerPositions]
  );

  const aggressiveness = useMemo(
    () => calculateAggressiveness(currentPlayerPositions),
    [currentPlayerPositions]
  );

  const courtCoverage = useMemo(
    () => calculateCourtCoverage(currentPlayerPositions),
    [currentPlayerPositions]
  );

  const zoneDistribution = useMemo(
    () => getZoneDistribution(currentPlayerPositions),
    [currentPlayerPositions]
  );

  // Create heatmap
  const heatmapData = useMemo(
    () => createHeatmapData(currentPlayerPositions),
    [currentPlayerPositions]
  );

  // Prepare chart data
  const zoneData = [
    { zone: 'Front Court', percentage: zoneDistribution.front },
    { zone: 'Mid Court', percentage: zoneDistribution.mid },
    { zone: 'Back Court', percentage: zoneDistribution.back },
  ];

  // Get timeline data with time in seconds
  const tDominanceTimelineData = useMemo(
    () => getTDominanceTimelineWithTime(currentPlayerPositionsWithTime, 2.0, 1.0),
    [currentPlayerPositionsWithTime]
  );

  const aggressivenessTimelineData = useMemo(
    () => getAggressivenessTimelineWithTime(currentPlayerPositionsWithTime, 1.0),
    [currentPlayerPositionsWithTime]
  );

  // Combine timeline data for charts
  const timelineData = useMemo(() => {
    // Merge T-dominance and aggressiveness data by time
    const combined = tDominanceTimelineData.map((item, index) => ({
      time: item.time,
      tDominance: item.tDominance,
      aggressiveness: aggressivenessTimelineData[index]?.aggressiveness || 0,
    }));

    return combined;
  }, [tDominanceTimelineData, aggressivenessTimelineData]);

  return (
    <div className="space-y-8">
      {/* Player Selector */}
      <div className="bg-white rounded-xl shadow-lg p-4">
        <div className="flex gap-4">
          <button
            onClick={() => setSelectedPlayer(1)}
            className={`flex-1 px-6 py-4 rounded-lg font-bold text-lg transition-all ${
              selectedPlayer === 1
                ? 'bg-dark-red text-white shadow-md'
                : 'bg-black/5 text-black hover:bg-black/10'
            }`}
          >
            Player 1
          </button>
          <button
            onClick={() => setSelectedPlayer(2)}
            className={`flex-1 px-6 py-4 rounded-lg font-bold text-lg transition-all ${
              selectedPlayer === 2
                ? 'bg-dark-red text-white shadow-md'
                : 'bg-black/5 text-black hover:bg-black/10'
            }`}
          >
            Player 2
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Court Coverage"
          value={courtCoverage.toFixed(1)}
          unit="%"
          icon={<MapPin className="w-6 h-6" />}
          primary
        />
        <StatCard
          title="T-Dominance"
          value={tDominance.toFixed(1)}
          unit="%"
          icon={<Target className="w-6 h-6" />}
          subtitle="Time near T-position"
        />
        <StatCard
          title="Aggressiveness"
          value={aggressiveness.toFixed(1)}
          unit="%"
          icon={<Zap className="w-6 h-6" />}
          subtitle="Time in front half"
        />
        <StatCard
          title="Front Court Time"
          value={zoneDistribution.front.toFixed(1)}
          unit="%"
          icon={<TrendingUp className="w-6 h-6" />}
        />
      </div>

      {/* Heatmap and Zone Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Court Position Heatmap */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-2xl font-bold text-black mb-6">Court Position Heatmap</h2>
          <div className="flex justify-center">
            <CourtHeatmap heatmapData={heatmapData} width={300} height={450} />
          </div>
        </div>

        {/* Zone Distribution */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-2xl font-bold text-black mb-6">Court Zone Distribution</h2>
          <ResponsiveContainer width="100%" height={450}>
            <BarChart data={zoneData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#00000010" />
              <XAxis type="number" stroke="#000000" tick={{ fill: '#000000' }} />
              <YAxis
                type="category"
                dataKey="zone"
                stroke="#000000"
                tick={{ fill: '#000000' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#FFFFFF',
                  border: '1px solid #00000020',
                  borderRadius: '8px',
                }}
                formatter={(value: number) => `${value.toFixed(1)}%`}
              />
              <Bar dataKey="percentage" fill="#8B0000" radius={[0, 8, 8, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Timeline Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* T-Dominance Timeline */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-2xl font-bold text-black mb-6">T-Dominance Over Time</h2>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={timelineData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#00000010" />
              <XAxis
                dataKey="time"
                stroke="#000000"
                tick={{ fill: '#000000' }}
                label={{ value: 'Time (seconds)', position: 'insideBottom', offset: -5 }}
              />
              <YAxis
                stroke="#000000"
                tick={{ fill: '#000000' }}
                label={{ value: 'T-Dominance (%)', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#FFFFFF',
                  border: '1px solid #00000020',
                  borderRadius: '8px',
                }}
                formatter={(value: number) => `${value.toFixed(1)}%`}
                labelFormatter={(value: number) => `${value.toFixed(1)}s`}
              />
              <Area
                type="monotone"
                dataKey="tDominance"
                stroke="#8B0000"
                fill="#8B0000"
                fillOpacity={0.3}
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Aggressiveness Timeline */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-2xl font-bold text-black mb-6">Aggressiveness Over Time</h2>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={timelineData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#00000010" />
              <XAxis
                dataKey="time"
                stroke="#000000"
                tick={{ fill: '#000000' }}
                label={{ value: 'Time (seconds)', position: 'insideBottom', offset: -5 }}
              />
              <YAxis
                stroke="#000000"
                tick={{ fill: '#000000' }}
                label={{
                  value: 'Front Court Time (%)',
                  angle: -90,
                  position: 'insideLeft',
                }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#FFFFFF',
                  border: '1px solid #00000020',
                  borderRadius: '8px',
                }}
                formatter={(value: number) => `${value.toFixed(1)}%`}
                labelFormatter={(value: number) => `${value.toFixed(1)}s`}
              />
              <Area
                type="monotone"
                dataKey="aggressiveness"
                stroke="#000000"
                fill="#000000"
                fillOpacity={0.3}
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Comparison View */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-bold text-black mb-6">Player Comparison</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="flex flex-col items-center">
            <h3 className="text-lg font-semibold text-black mb-4">Player 1</h3>
            <CourtHeatmap
              heatmapData={createHeatmapData(player1Positions)}
              width={250}
              height={375}
            />
          </div>
          <div className="flex flex-col items-center">
            <h3 className="text-lg font-semibold text-black mb-4">Player 2</h3>
            <CourtHeatmap
              heatmapData={createHeatmapData(player2Positions)}
              width={250}
              height={375}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
