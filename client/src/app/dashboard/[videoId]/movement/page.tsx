'use client';

import { useParams } from 'next/navigation';
import { Move, Target, Clock, TrendingUp } from 'lucide-react';
import { KPICard } from '@/components/dashboard/KPICard';
import { HeatmapChart } from '@/components/charts/HeatmapChart';
import { DistributionPieChart } from '@/components/charts/DistributionPieChart';
import {
  useMovementMetrics,
  useTZoneOccupancy,
  usePlayerPositionHeatmap,
  useCourtQuadrants,
} from '@/lib/hooks/useAnalytics';
import { formatDistance, formatDecimal, formatPercentage } from '@/lib/utils/formatters';


/**
 * Movement & Positioning tab
 * Shows court coverage, T-zone control, and positioning analysis
 */
export default function MovementPage() {
  const params = useParams();
  const videoId = params.videoId as string;

  const { data: movement, isLoading: movementLoading } = useMovementMetrics(videoId);
  const { data: tZone, isLoading: tZoneLoading } = useTZoneOccupancy(videoId);
  const { data: heatmap, isLoading: heatmapLoading } = usePlayerPositionHeatmap(videoId);
  const { data: quadrants, isLoading: quadrantsLoading } = useCourtQuadrants(videoId);

  const isLoading = movementLoading || tZoneLoading || heatmapLoading || quadrantsLoading;

  if (isLoading) {
    return (
      <div className="p-8 space-y-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-32 bg-gray-200 rounded-lg animate-pulse" />
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {[...Array(2)].map((_, i) => (
            <div key={i} className="h-96 bg-gray-200 rounded-lg animate-pulse" />
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
          title="Total Distance"
          value={movement?.data.total_distance ? formatDistance(movement.data.total_distance) : 'N/A'}
          icon={Move}
          subtitle={
            movement?.data.shot_count
              ? `${movement.data.shot_count} shots`
              : undefined
          }
        />

        <KPICard
          title="% Time in T-Zone"
          value={tZone?.data.pct_time_in_t ? formatPercentage(tZone.data.pct_time_in_t) : 'N/A'}
          icon={Target}
          subtitle="Court positioning"
        />

        <KPICard
          title="Avg Time to T"
          value={
            tZone?.data.avg_time_to_t
              ? formatDecimal(tZone.data.avg_time_to_t, 2) + 's'
              : 'N/A'
          }
          icon={Clock}
          subtitle="T-zone recovery"
        />

        <KPICard
          title="T-Zone Success Rate"
          value={
            tZone?.data.t_zone_success_rate
              ? formatPercentage(tZone.data.t_zone_success_rate)
              : 'N/A'
          }
          icon={TrendingUp}
          subtitle={
            tZone?.data.successful_returns
              ? `${tZone.data.successful_returns} successful`
              : undefined
          }
        />
      </div>

      {/* Row 2: Heatmap and Court Quadrants */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Player Position Heatmap */}
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          {heatmap?.data.heatmap_grid ? (
            <HeatmapChart
              grid={heatmap.data.heatmap_grid}
              title="Player Position Heatmap"
              showCourtLines={true}
            />
          ) : (
            <div className="h-96 flex items-center justify-center text-gray-500">
              No heatmap data available
            </div>
          )}
        </div>

        {/* Court Quadrants Distribution */}
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <DistributionPieChart
            data={quadrants?.data.distribution || []}
            title="Court Quadrant Distribution"
          />
        </div>
      </div>

      {/* Movement Metrics Details */}
      {movement && (
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Movement Statistics</h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-6">
            <div>
              <p className="text-sm text-gray-600">Avg Distance per Rally</p>
              <p className="text-2xl font-bold text-gray-900">
                {formatDistance(movement.data.avg_distance_per_rally)}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Avg Distance to Ball</p>
              <p className="text-2xl font-bold text-gray-900">
                {formatDistance(movement.data.avg_distance_to_ball)}
              </p>
            </div>
            {movement.data.max_distance_to_ball !== null && (
              <div>
                <p className="text-sm text-gray-600">Max Distance to Ball</p>
                <p className="text-2xl font-bold text-gray-900">
                  {formatDistance(movement.data.max_distance_to_ball)}
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* T-Zone Metrics Details */}
      {tZone && (
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">T-Zone Control Metrics</h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-6">
            {tZone.data.min_time_to_t !== null && (
              <div>
                <p className="text-sm text-gray-600">Min Time to T</p>
                <p className="text-2xl font-bold text-gray-900">
                  {formatDecimal(tZone.data.min_time_to_t, 2)}s
                </p>
              </div>
            )}
            {tZone.data.max_time_to_t !== null && (
              <div>
                <p className="text-sm text-gray-600">Max Time to T</p>
                <p className="text-2xl font-bold text-gray-900">
                  {formatDecimal(tZone.data.max_time_to_t, 2)}s
                </p>
              </div>
            )}
            <div>
              <p className="text-sm text-gray-600">Total Shots</p>
              <p className="text-2xl font-bold text-gray-900">
                {tZone.data.total_shots_taken}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Successful Returns</p>
              <p className="text-2xl font-bold text-gray-900">
                {tZone.data.successful_returns}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
