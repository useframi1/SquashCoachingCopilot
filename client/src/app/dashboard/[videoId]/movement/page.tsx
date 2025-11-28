'use client';

import { useParams } from 'next/navigation';
import { Move, Target, Clock, TrendingUp, Footprints } from 'lucide-react';
import { DualPlayerKPICard } from '@/components/dashboard/DualPlayerKPICard';
import { ComparisonStatCard } from '@/components/dashboard/ComparisonStatCard';
import { DualPlayerSquashCourtHeatmap } from '@/components/charts/DualPlayerSquashCourtHeatmap';
import { MovementPerRallyChart } from '@/components/charts/MovementPerRallyChart';
import { DualPlayerTZoneOccupancyChart } from '@/components/charts/DualPlayerTZoneOccupancyChart';
import {
  useMovementMetricsPerRally,
  useTZoneOccupancyPerRally,
  useMovementMetricsForPlayer,
  useTZoneOccupancyForPlayer,
  usePlayerPositionHeatmapForPlayer,
  useCourtQuadrantsForPlayer,
} from '@/lib/hooks/useAnalytics';
import { usePlayerNames } from '@/lib/hooks/usePlayerNames';
import { formatDistance, formatDecimal, formatPercentage } from '@/lib/utils/formatters';

/**
 * Movement & Positioning Analysis Tab
 *
 * Focuses on:
 * - Distance covered and movement efficiency
 * - T-zone control and recovery metrics
 * - Court positioning and coverage patterns
 * - Per-rally movement trends
 */
export default function MovementPage() {
  const params = useParams();
  const videoId = params.videoId as string;

  const { player1Name, player2Name } = usePlayerNames(videoId);

  // Call aggregate APIs separately for each player (ignores global player filter)
  const { data: player1Movement, isLoading: p1MovementLoading } = useMovementMetricsForPlayer(videoId, 1);
  const { data: player2Movement, isLoading: p2MovementLoading } = useMovementMetricsForPlayer(videoId, 2);
  const { data: player1TZone, isLoading: p1TZoneLoading } = useTZoneOccupancyForPlayer(videoId, 1);
  const { data: player2TZone, isLoading: p2TZoneLoading } = useTZoneOccupancyForPlayer(videoId, 2);

  // Heatmaps and quadrants for each player separately
  const { data: player1Heatmap, isLoading: p1HeatmapLoading } = usePlayerPositionHeatmapForPlayer(videoId, 1);
  const { data: player2Heatmap, isLoading: p2HeatmapLoading } = usePlayerPositionHeatmapForPlayer(videoId, 2);
  const { data: player1Quadrants, isLoading: p1QuadrantsLoading } = useCourtQuadrantsForPlayer(videoId, 1);
  const { data: player2Quadrants, isLoading: p2QuadrantsLoading } = useCourtQuadrantsForPlayer(videoId, 2);

  // Per-rally endpoints for charts (respects global filters)
  const { data: movementPerRally, isLoading: movementPerRallyLoading } = useMovementMetricsPerRally(videoId);
  const { data: tZonePerRally, isLoading: tZonePerRallyLoading } = useTZoneOccupancyPerRally(videoId);

  const isLoading =
    p1MovementLoading || p2MovementLoading ||
    p1TZoneLoading || p2TZoneLoading ||
    p1HeatmapLoading || p2HeatmapLoading ||
    p1QuadrantsLoading || p2QuadrantsLoading ||
    movementPerRallyLoading || tZonePerRallyLoading;

  // Calculate max distance in a single rally for each player
  const player1MaxRallyDistance = movementPerRally?.data && movementPerRally.data.length > 0
    ? Math.max(...movementPerRally.data.map((r: any) => r.player_1?.total_distance || 0))
    : 0;
  const player2MaxRallyDistance = movementPerRally?.data && movementPerRally.data.length > 0
    ? Math.max(...movementPerRally.data.map((r: any) => r.player_2?.total_distance || 0))
    : 0;

  if (isLoading) {
    return (
      <div className="p-8 space-y-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="h-32 bg-gray-200 rounded-lg animate-pulse" />
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-96 bg-gray-200 rounded-lg animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="p-8 space-y-8">
      {/* Section 1: Key Movement Metrics - Dual Player Cards */}
      <div>
        <h2 className="text-xl font-bold text-gray-900 mb-4">Movement Metrics</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          <DualPlayerKPICard
            title="Total Distance Covered"
            player1Value={player1Movement?.data ? formatDistance(player1Movement.data.total_distance) : 'N/A'}
            player2Value={player2Movement?.data ? formatDistance(player2Movement.data.total_distance) : 'N/A'}
            player1Name={player1Name}
            player2Name={player2Name}
            icon={Move}
            subtitle="Match total"
          />

          <DualPlayerKPICard
            title="Avg Distance per Rally"
            player1Value={player1Movement?.data ? formatDistance(player1Movement.data.avg_distance_per_rally) : 'N/A'}
            player2Value={player2Movement?.data ? formatDistance(player2Movement.data.avg_distance_per_rally) : 'N/A'}
            player1Name={player1Name}
            player2Name={player2Name}
            icon={Footprints}
            subtitle="Rally efficiency"
          />

          <ComparisonStatCard
            title="Max Distance in Rally"
            player1Label="Longest rally"
            player2Label="Longest rally"
            player1Value={player1MaxRallyDistance > 0 ? formatDistance(player1MaxRallyDistance) : 'N/A'}
            player2Value={player2MaxRallyDistance > 0 ? formatDistance(player2MaxRallyDistance) : 'N/A'}
            player1Name={player1Name}
            player2Name={player2Name}
            icon={TrendingUp}
            highlightBetter={true}
          />
        </div>
      </div>

      {/* Section 2: T-Zone Control Metrics */}
      <div>
        <h2 className="text-xl font-bold text-gray-900 mb-4">T-Zone Control</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          <ComparisonStatCard
            title="% Time in T-Zone"
            player1Label="T-zone occupancy"
            player2Label="T-zone occupancy"
            player1Value={player1TZone?.data ? formatPercentage(player1TZone.data.pct_time_in_t) : 'N/A'}
            player2Value={player2TZone?.data ? formatPercentage(player2TZone.data.pct_time_in_t) : 'N/A'}
            player1Name={player1Name}
            player2Name={player2Name}
            icon={Target}
            highlightBetter={true}
          />

          <ComparisonStatCard
            title="Avg Time to T-Zone"
            player1Label="Recovery speed"
            player2Label="Recovery speed"
            player1Value={
              player1TZone?.data.avg_time_to_t !== null && player1TZone?.data.avg_time_to_t !== undefined
                ? formatDecimal(player1TZone.data.avg_time_to_t, 2) + 's'
                : 'N/A'
            }
            player2Value={
              player2TZone?.data.avg_time_to_t !== null && player2TZone?.data.avg_time_to_t !== undefined
                ? formatDecimal(player2TZone.data.avg_time_to_t, 2) + 's'
                : 'N/A'
            }
            player1Name={player1Name}
            player2Name={player2Name}
            icon={Clock}
            highlightLower={true}
          />

          <ComparisonStatCard
            title="T-Zone Success Rate"
            player1Label="Successful returns"
            player2Label="Successful returns"
            player1Value={
              player1TZone?.data.t_zone_success_rate !== null && player1TZone?.data.t_zone_success_rate !== undefined
                ? formatPercentage(player1TZone.data.t_zone_success_rate)
                : 'N/A'
            }
            player2Value={
              player2TZone?.data.t_zone_success_rate !== null && player2TZone?.data.t_zone_success_rate !== undefined
                ? formatPercentage(player2TZone.data.t_zone_success_rate)
                : 'N/A'
            }
            player1Name={player1Name}
            player2Name={player2Name}
            icon={TrendingUp}
            highlightBetter={true}
          />
        </div>
      </div>

      {/* Section 3: Spatial Analysis - Dual Court Heatmap */}
      <div>
        <h2 className="text-xl font-bold text-gray-900 mb-4">Court Coverage & Positioning</h2>
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          {player1Heatmap?.data.heatmap_grid && player2Heatmap?.data.heatmap_grid ? (
            <DualPlayerSquashCourtHeatmap
              player1HeatmapData={player1Heatmap.data.heatmap_grid}
              player2HeatmapData={player2Heatmap.data.heatmap_grid}
              player1QuadrantData={player1Quadrants?.data.distribution}
              player2QuadrantData={player2Quadrants?.data.distribution}
              quadrantBoundaries={player1Quadrants?.quadrant_boundaries}
              player1Name={player1Name}
              player2Name={player2Name}
              title="Player Position Analysis (Top-down View)"
            />
          ) : (
            <div className="h-96 flex items-center justify-center text-gray-500">
              No court positioning data available
            </div>
          )}
        </div>
      </div>

      {/* Section 4: Movement Per Rally Line Chart */}
      <div className="bg-white p-6 rounded-lg border border-gray-200">
        <MovementPerRallyChart
          data={movementPerRally?.data || []}
          title="Distance Covered per Rally"
          player1Name={player1Name}
          player2Name={player2Name}
        />
      </div>

      {/* Section 5: T-Zone Occupancy Per Rally */}
      <div className="bg-white p-6 rounded-lg border border-gray-200">
        <DualPlayerTZoneOccupancyChart
          data={tZonePerRally?.data || []}
          title="T-Zone Occupancy per Rally"
          player1Name={player1Name}
          player2Name={player2Name}
        />
      </div>

      {/* Section 6: Detailed Statistics */}
      {player1Movement?.data && player2Movement?.data && player1TZone?.data && player2TZone?.data && (
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Detailed Statistics</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Player 1 */}
            <div>
              <p className="text-sm font-medium text-red-700 mb-3">{player1Name}</p>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Total Distance</span>
                  <span className="text-lg font-bold text-gray-900">
                    {formatDistance(player1Movement.data.total_distance)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Avg per Rally</span>
                  <span className="text-lg font-bold text-gray-900">
                    {formatDistance(player1Movement.data.avg_distance_per_rally)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">% Time in T-Zone</span>
                  <span className="text-lg font-bold text-gray-900">
                    {formatPercentage(player1TZone.data.pct_time_in_t)}
                  </span>
                </div>
                {player1TZone.data.avg_time_to_t !== null && (
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Avg Time to T</span>
                    <span className="text-lg font-bold text-gray-900">
                      {formatDecimal(player1TZone.data.avg_time_to_t, 2)}s
                    </span>
                  </div>
                )}
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Total Shots</span>
                  <span className="text-lg font-bold text-gray-900">
                    {player1TZone.data.total_shots_taken}
                  </span>
                </div>
              </div>
            </div>

            {/* Player 2 */}
            <div>
              <p className="text-sm font-medium text-gray-600 mb-3">{player2Name}</p>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Total Distance</span>
                  <span className="text-lg font-bold text-gray-900">
                    {formatDistance(player2Movement.data.total_distance)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Avg per Rally</span>
                  <span className="text-lg font-bold text-gray-900">
                    {formatDistance(player2Movement.data.avg_distance_per_rally)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">% Time in T-Zone</span>
                  <span className="text-lg font-bold text-gray-900">
                    {formatPercentage(player2TZone.data.pct_time_in_t)}
                  </span>
                </div>
                {player2TZone.data.avg_time_to_t !== null && (
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Avg Time to T</span>
                    <span className="text-lg font-bold text-gray-900">
                      {formatDecimal(player2TZone.data.avg_time_to_t, 2)}s
                    </span>
                  </div>
                )}
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Total Shots</span>
                  <span className="text-lg font-bold text-gray-900">
                    {player2TZone.data.total_shots_taken}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
