import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import type {
  AnalysisResults,
  DashboardData,
  RallySummary,
  StrokeStats,
  PlayerStats,
  RallyData,
} from './types';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Format seconds to MM:SS format
 */
export function formatDuration(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

/**
 * Format timestamp to HH:MM:SS
 */
export function formatTimestamp(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${minutes}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Count strokes by type from rally frames
 */
function countStrokes(rally: RallyData, playerId: number): StrokeStats {
  const stats: StrokeStats = { forehand: 0, backhand: 0, total: 0 };

  rally.rally_frames.forEach((frame) => {
    const player = playerId === 1 ? frame.player1 : frame.player2;
    if (player?.stroke_type === 'forehand') {
      stats.forehand++;
      stats.total++;
    } else if (player?.stroke_type === 'backhand') {
      stats.backhand++;
      stats.total++;
    }
  });

  return stats;
}

/**
 * Calculate player statistics from all rallies
 */
function calculatePlayerStats(rallies: RallyData[], playerId: number): PlayerStats {
  let totalForehand = 0;
  let totalBackhand = 0;
  let totalPositionX = 0;
  let totalPositionY = 0;
  let positionCount = 0;

  rallies.forEach((rally) => {
    rally.rally_frames.forEach((frame) => {
      const player = playerId === 1 ? frame.player1 : frame.player2;
      if (player) {
        if (player.stroke_type === 'forehand') totalForehand++;
        if (player.stroke_type === 'backhand') totalBackhand++;

        if (player.real_position) {
          totalPositionX += player.real_position[0];
          totalPositionY += player.real_position[1];
          positionCount++;
        }
      }
    });
  });

  return {
    player_id: playerId,
    strokes: {
      forehand: totalForehand,
      backhand: totalBackhand,
      total: totalForehand + totalBackhand,
    },
    averagePosition: [
      positionCount > 0 ? totalPositionX / positionCount : 0,
      positionCount > 0 ? totalPositionY / positionCount : 0,
    ],
    movementDistance: 0, // TODO: Calculate based on position changes
  };
}

/**
 * Calculate rally duration and timestamps from frame data
 */
function calculateRallyTiming(rally: RallyData): { duration: number; start_time: number; end_time: number } {
  // If the API provides these values, use them
  if (rally.duration !== undefined && rally.start_timestamp !== undefined && rally.end_timestamp !== undefined) {
    return {
      duration: rally.duration,
      start_time: rally.start_timestamp,
      end_time: rally.end_timestamp,
    };
  }

  // Otherwise calculate from frames
  const startFrame = rally.rally_frames[0];
  const endFrame = rally.rally_frames[rally.rally_frames.length - 1];

  const start_time = startFrame?.timestamp || 0;
  const end_time = endFrame?.timestamp || 0;
  const duration = end_time - start_time;

  return { duration, start_time, end_time };
}

/**
 * Transform API results into dashboard data
 */
export function transformToDashboardData(
  jobId: string,
  filename: string,
  results: AnalysisResults
): DashboardData {
  const rallies: RallySummary[] = results.rallies.map((rally, index) => {
    const timing = calculateRallyTiming(rally);
    return {
      rally_number: index + 1,
      duration: timing.duration,
      start_time: timing.start_time,
      end_time: timing.end_time,
      player1_strokes: countStrokes(rally, 1),
      player2_strokes: countStrokes(rally, 2),
      total_strokes: countStrokes(rally, 1).total + countStrokes(rally, 2).total,
    };
  });

  // Handle empty rallies case
  const longestRally = rallies.length > 0
    ? rallies.reduce((longest, current) =>
        current.duration > longest.duration ? current : longest
      )
    : {
        rally_number: 0,
        duration: 0,
        start_time: 0,
        end_time: 0,
        player1_strokes: { forehand: 0, backhand: 0, total: 0 },
        player2_strokes: { forehand: 0, backhand: 0, total: 0 },
        total_strokes: 0,
      };

  return {
    jobId,
    filename,
    totalRallies: results.total_rallies,
    avgRallyDuration: results.avg_rally_duration,
    longestRally,
    rallies,
    player1Stats: calculatePlayerStats(results.rallies, 1),
    player2Stats: calculatePlayerStats(results.rallies, 2),
  };
}

/**
 * Filter rallies by time range
 */
export function filterRalliesByTimeRange(
  rallies: RallySummary[],
  startTime: number,
  endTime: number
): RallySummary[] {
  return rallies.filter(
    (rally) => rally.start_time >= startTime && rally.end_time <= endTime
  );
}

/**
 * Get file size in human readable format
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}
