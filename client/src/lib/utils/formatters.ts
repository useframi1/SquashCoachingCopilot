/**
 * Format utilities for displaying data in the dashboard
 */

/**
 * Format duration in seconds to MM:SS format
 */
export function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Format number to 1 decimal place
 */
export function formatDecimal(value: number, decimals: number = 1): string {
  return value.toFixed(decimals);
}

/**
 * Format percentage (0-100) with 1 decimal place
 */
export function formatPercentage(value: number): string {
  return `${value.toFixed(1)}%`;
}

/**
 * Format distance in meters
 */
export function formatDistance(meters: number): string {
  return `${meters.toFixed(1)}m`;
}

/**
 * Format speed in m/s
 */
export function formatSpeed(metersPerSecond: number): string {
  return `${metersPerSecond.toFixed(1)} m/s`;
}

/**
 * Format score (e.g., "3-2")
 */
export function formatScore(p1Score: number, p2Score: number): string {
  return `${p1Score}-${p2Score}`;
}

/**
 * Format player name
 */
export function formatPlayerName(playerId: 1 | 2): string {
  return `Player ${playerId}`;
}

/**
 * Format shot type label (convert snake_case to Title Case)
 */
export function formatShotType(shotType: string): string {
  return shotType
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/**
 * Format large numbers with commas
 */
export function formatNumber(value: number): string {
  return value.toLocaleString();
}

/**
 * Format timestamp to human-readable format
 */
export function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleString();
}
