/**
 * Chart utility functions for data transformation
 */

import type { RallyTimelineItem } from '@/types/api';

/**
 * Brand colors for consistent charting
 */
export const CHART_COLORS = {
  player1: '#b91c1c', // Red-700 for Player 1
  player2: '#6b7280', // Gray-500 for Player 2
  neutral: '#d1d5db', // Gray-300
  positive: '#10b981', // Green-500 for positive trends
  negative: '#ef4444', // Red-500 for negative trends
  background: '#ffffff',
  text: '#1f2937', // Gray-800
};

/**
 * Shot type colors (for consistent visualization across charts)
 */
export const SHOT_TYPE_COLORS: Record<string, string> = {
  forehand: '#b91c1c',
  backhand: '#6b7280',
  straight_drive: '#dc2626',
  cross_court_drive: '#f87171',
  drop: '#fca5a5',
  lob: '#fecaca',
  volley: '#991b1b',
  boast: '#7f1d1d',
};

/**
 * Get color for a specific player
 */
export function getPlayerColor(playerId: 1 | 2 | null): string {
  if (playerId === 1) return CHART_COLORS.player1;
  if (playerId === 2) return CHART_COLORS.player2;
  return CHART_COLORS.neutral;
}

/**
 * Normalize heatmap values to 0-1 range
 */
export function normalizeHeatmapGrid(grid: number[][]): number[][] {
  const flatValues = grid.flat();
  const maxValue = Math.max(...flatValues);

  if (maxValue === 0) return grid;

  return grid.map((row) => row.map((value) => value / maxValue));
}

/**
 * Get color from value (0-1) using a color scale
 */
export function getHeatmapColor(normalizedValue: number): string {
  // Color scale: white → light red → dark red
  const colors = ['#ffffff', '#fee2e2', '#fca5a5', '#dc2626', '#991b1b'];

  const index = Math.min(Math.floor(normalizedValue * (colors.length - 1)), colors.length - 1);
  return colors[index];
}

/**
 * Transform rally timeline data for scatter chart
 * Adds jitter to x-axis for better visualization when rallies overlap
 */
export function transformRallyTimelineForScatter(
  rallies: RallyTimelineItem[]
): Array<{
  x: number;
  y: number;
  shotCount: number;
  winner: 1 | 2 | null;
  avgBallSpeed: number | null;
  rallyId: number;
}> {
  return rallies.map((rally) => ({
    x: rally.rally_id,
    y: rally.rally_duration,
    shotCount: rally.shot_count,
    winner: rally.point_winner,
    avgBallSpeed: rally.avg_ball_speed,
    rallyId: rally.rally_id,
  }));
}

/**
 * Calculate histogram buckets for intensity distribution
 */
export function calculateIntensityBuckets(
  intensityData: number[],
  bucketCount: number = 10
): Array<{ range: string; count: number }> {
  if (intensityData.length === 0) return [];

  const min = Math.min(...intensityData);
  const max = Math.max(...intensityData);
  const bucketSize = (max - min) / bucketCount;

  const buckets = Array.from({ length: bucketCount }, (_, i) => ({
    range: `${(min + i * bucketSize).toFixed(1)}-${(min + (i + 1) * bucketSize).toFixed(1)}`,
    count: 0,
  }));

  intensityData.forEach((value) => {
    const bucketIndex = Math.min(
      Math.floor((value - min) / bucketSize),
      bucketCount - 1
    );
    buckets[bucketIndex].count++;
  });

  return buckets;
}

/**
 * Generate gradient for momentum chart
 */
export function getMomentumGradient(differential: number): string {
  if (differential > 0) return CHART_COLORS.player1;
  if (differential < 0) return CHART_COLORS.player2;
  return CHART_COLORS.neutral;
}
