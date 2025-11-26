'use client';

import { useParams } from 'next/navigation';
import { Zap, TrendingUp, Target, Layers } from 'lucide-react';
import { KPICard } from '@/components/dashboard/KPICard';
import { DistributionPieChart } from '@/components/charts/DistributionPieChart';
import { DistributionBarChart } from '@/components/charts/DistributionBarChart';
import { BallSpeedLineChart } from '@/components/charts/BallSpeedLineChart';
import {
  useStrokeDistribution,
  useShotTypeDistribution,
  useBallSpeed,
  useRhythmDisruption,
  useWallQuadrants,
  useRallyTimeline,
  useShotEffectiveness,
} from '@/lib/hooks/useAnalytics';
import { useFilterStore } from '@/lib/stores/filterStore';
import { formatSpeed, formatDecimal, formatPercentage } from '@/lib/utils/formatters';

/**
 * Performance & Shot Analysis tab
 * Shows stroke patterns, shot types, ball speed, and placement quality
 */
export default function PerformancePage() {
  const params = useParams();
  const videoId = params.videoId as string;
  const { playerId } = useFilterStore();

  const { data: strokeDist, isLoading: strokeLoading } = useStrokeDistribution(videoId);
  const { data: shotTypeDist, isLoading: shotTypeLoading } = useShotTypeDistribution(videoId);
  const { data: ballSpeed, isLoading: ballSpeedLoading } = useBallSpeed(videoId);
  const { data: rhythm, isLoading: rhythmLoading } = useRhythmDisruption(videoId);
  const { data: wallQuad, isLoading: wallQuadLoading } = useWallQuadrants(videoId);
  const { data: rallyTimeline, isLoading: timelineLoading } = useRallyTimeline(videoId);

  // Shot effectiveness (only if player filter is active)
  const { data: shotEff } = useShotEffectiveness(
    videoId,
    playerId || 1 // Default to player 1 if no filter
  );

  const isLoading =
    strokeLoading ||
    shotTypeLoading ||
    ballSpeedLoading ||
    rhythmLoading ||
    wallQuadLoading ||
    timelineLoading;

  if (isLoading) {
    return (
      <div className="p-8 space-y-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-32 bg-gray-200 rounded-lg animate-pulse" />
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-80 bg-gray-200 rounded-lg animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="p-8 space-y-8">
      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        <KPICard
          title="Avg Ball Speed"
          value={ballSpeed?.data.mean_speed ? formatSpeed(ballSpeed.data.mean_speed) : 'N/A'}
          icon={Zap}
          subtitle={
            ballSpeed?.data.shot_count
              ? `${ballSpeed.data.shot_count} shots`
              : undefined
          }
        />

        <KPICard
          title="Rhythm Disruption"
          value={rhythm?.data.ball_speed_cv ? formatDecimal(rhythm.data.ball_speed_cv, 2) : 'N/A'}
          icon={TrendingUp}
          subtitle="Coefficient of Variation"
        />

        <KPICard
          title="Shot Effectiveness"
          value={
            shotEff?.data.avg_displacement_from_t
              ? formatDecimal(shotEff.data.avg_displacement_from_t, 1) + 'm'
              : 'N/A'
          }
          icon={Target}
          subtitle="Avg opponent displacement"
        />

        <KPICard
          title="Depth Dominance"
          value={
            shotEff?.data.depth_dominance_pct
              ? formatPercentage(shotEff.data.depth_dominance_pct)
              : 'N/A'
          }
          icon={Layers}
          subtitle="% keeping opponent deep"
        />
      </div>

      {/* Row 2: Distribution Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Stroke Distribution */}
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <DistributionPieChart
            data={strokeDist?.data.distribution || []}
            title="Stroke Distribution"
          />
        </div>

        {/* Shot Types Distribution */}
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <DistributionBarChart
            data={shotTypeDist?.data.distribution || []}
            title="Shot Types"
          />
        </div>
      </div>

      {/* Row 3: Wall Quadrants and Ball Speed Over Time */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Wall Quadrants */}
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <DistributionPieChart
            data={wallQuad?.data.distribution || []}
            title="Wall Hit Distribution"
          />
        </div>

        {/* Ball Speed Over Time */}
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <BallSpeedLineChart
            data={rallyTimeline?.data || []}
            title="Ball Speed Over Rallies"
          />
        </div>
      </div>

      {/* Detailed Ball Speed Stats */}
      {ballSpeed && (
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Ball Speed Statistics</h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-6">
            <div>
              <p className="text-sm text-gray-600">Mean Speed</p>
              <p className="text-2xl font-bold text-gray-900">
                {formatSpeed(ballSpeed.data.mean_speed)}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Min Speed</p>
              <p className="text-2xl font-bold text-gray-900">
                {formatSpeed(ballSpeed.data.min_speed)}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Max Speed</p>
              <p className="text-2xl font-bold text-gray-900">
                {formatSpeed(ballSpeed.data.max_speed)}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Std Deviation</p>
              <p className="text-2xl font-bold text-gray-900">
                {formatDecimal(ballSpeed.data.std_dev, 1)}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
