import {Rally, RallyFrame, PlayerData} from "./types";

// Squash court dimensions (in meters)
export const COURT_LENGTH = 9.75;
export const COURT_WIDTH = 6.4;
export const T_POSITION: [number, number] = [COURT_WIDTH / 2, COURT_LENGTH / 2]; // Center of court

// Court zones (thirds of the court)
export const FRONT_COURT_BOUNDARY = COURT_LENGTH / 3;
export const MID_COURT_BOUNDARY = (2 * COURT_LENGTH) / 3;

/**
 * Calculate distance from a position to the T
 */
export function distanceFromT(position: [number, number]): number {
    const [x, y] = position;
    const [tx, ty] = T_POSITION;
    return Math.sqrt(Math.pow(x - tx, 2) + Math.pow(y - ty, 2));
}

/**
 * Classify position into court zone
 */
export function getCourtZone(
    position: [number, number]
): "front" | "mid" | "back" {
    const [, y] = position;
    if (y < FRONT_COURT_BOUNDARY) return "front";
    if (y < MID_COURT_BOUNDARY) return "mid";
    return "back";
}

/**
 * Check if position is in front half (aggressive positioning)
 */
export function isInFrontHalf(position: [number, number]): boolean {
    const [, y] = position;
    return y < COURT_LENGTH / 2;
}

/**
 * Calculate T-Dominance percentage for a player
 * T-Dominance: percentage of time spent within a threshold distance from T
 */
export function calculateTDominance(
    positions: [number, number][],
    threshold: number = 2.0 // meters from T
): number {
    if (positions.length === 0) return 0;

    const nearTCount = positions.filter(
        (pos) => distanceFromT(pos) <= threshold
    ).length;
    return (nearTCount / positions.length) * 100;
}

/**
 * Calculate Aggressiveness score (% time in front half)
 */
export function calculateAggressiveness(positions: [number, number][]): number {
    if (positions.length === 0) return 0;

    const frontHalfCount = positions.filter((pos) => isInFrontHalf(pos)).length;
    return (frontHalfCount / positions.length) * 100;
}

/**
 * Calculate court coverage (simplified as max range covered)
 */
export function calculateCourtCoverage(positions: [number, number][]): number {
    if (positions.length === 0) return 0;

    const xValues = positions.map((p) => p[0]);
    const yValues = positions.map((p) => p[1]);

    const xRange = Math.max(...xValues) - Math.min(...xValues);
    const yRange = Math.max(...yValues) - Math.min(...yValues);

    // Calculate as percentage of total court area
    const coveredArea = xRange * yRange;
    const totalArea = COURT_WIDTH * COURT_LENGTH;

    return Math.min((coveredArea / totalArea) * 100, 100);
}

/**
 * Get court zone distribution
 */
export function getZoneDistribution(positions: [number, number][]): {
    front: number;
    mid: number;
    back: number;
} {
    if (positions.length === 0) return {front: 0, mid: 0, back: 0};

    const zones = positions.map((pos) => getCourtZone(pos));
    const total = zones.length;

    return {
        front: (zones.filter((z) => z === "front").length / total) * 100,
        mid: (zones.filter((z) => z === "mid").length / total) * 100,
        back: (zones.filter((z) => z === "back").length / total) * 100,
    };
}

/**
 * Extract player positions from rally frames
 */
export function extractPlayerPositions(
    rallyFrames: RallyFrame[],
    playerId: 1 | 2
): [number, number][] {
    return rallyFrames.map((frame) => {
        const player = playerId === 1 ? frame.player1 : frame.player2;
        return player.real_position;
    });
}

/**
 * Extract player positions with timestamps from rally frames
 */
export function extractPlayerPositionsWithTime(
    rallyFrames: RallyFrame[],
    playerId: 1 | 2
): Array<{position: [number, number]; timestamp: number}> {
    return rallyFrames.map((frame) => {
        const player = playerId === 1 ? frame.player1 : frame.player2;
        return {
            position: player.real_position,
            timestamp: frame.timestamp,
        };
    });
}

/**
 * Extract all player positions from multiple rallies
 */
export function extractAllPlayerPositions(
    rallies: Rally[],
    playerId: 1 | 2
): [number, number][] {
    const allPositions: [number, number][] = [];

    rallies.forEach((rally) => {
        const positions = extractPlayerPositions(rally.rally_frames, playerId);
        allPositions.push(...positions);
    });

    return allPositions;
}

/**
 * Extract all player positions with timestamps from multiple rallies
 */
export function extractAllPlayerPositionsWithTime(
    rallies: Rally[],
    playerId: 1 | 2
): Array<{position: [number, number]; timestamp: number}> {
    const allPositions: Array<{position: [number, number]; timestamp: number}> = [];

    rallies.forEach((rally) => {
        const positions = extractPlayerPositionsWithTime(rally.rally_frames, playerId);
        allPositions.push(...positions);
    });

    return allPositions;
}

/**
 * Create heatmap data from positions
 * Returns a 2D grid with density values
 */
export function createHeatmapData(
    positions: [number, number][],
    gridSize: {rows: number; cols: number} = {rows: 20, cols: 13}
): number[][] {
    const {rows, cols} = gridSize;
    const grid: number[][] = Array(rows)
        .fill(0)
        .map(() => Array(cols).fill(0));

    if (positions.length === 0) return grid;

    const cellWidth = COURT_WIDTH / cols;
    const cellHeight = COURT_LENGTH / rows;

    positions.forEach(([x, y]) => {
        // Clamp positions to court boundaries
        const clampedX = Math.max(0, Math.min(x, COURT_WIDTH));
        const clampedY = Math.max(0, Math.min(y, COURT_LENGTH));

        const colIndex = Math.min(Math.floor(clampedX / cellWidth), cols - 1);
        const rowIndex = Math.min(Math.floor(clampedY / cellHeight), rows - 1);

        grid[rowIndex][colIndex]++;
    });

    return grid;
}

/**
 * Create wall hit heatmap from ball data
 * Returns data for visualizing where the ball hits the front wall
 */
export function createWallHitHeatmap(
    rallyFrames: RallyFrame[],
    gridSize: {rows: number; cols: number} = {rows: 15, cols: 10}
): number[][] {
    const {rows, cols} = gridSize;
    const grid: number[][] = Array(rows)
        .fill(0)
        .map(() => Array(cols).fill(0));

    // Front wall dimensions (approximate)
    const WALL_WIDTH = COURT_WIDTH; // 6.4m
    const WALL_HEIGHT = 4.57; // Front wall height in meters

    const cellWidth = WALL_WIDTH / cols;
    const cellHeight = WALL_HEIGHT / rows;

    rallyFrames.forEach((frame) => {
        if (frame.ball.is_wall_hit && frame.ball.ball_hit_real_position) {
            const [x, y] = frame.ball.ball_hit_real_position;

            // Assuming x is horizontal position, y is height on wall
            const clampedX = Math.max(0, Math.min(x, WALL_WIDTH));
            const clampedY = Math.max(0, Math.min(y, WALL_HEIGHT));

            const colIndex = Math.min(
                Math.floor(clampedX / cellWidth),
                cols - 1
            );
            const rowIndex = Math.min(
                Math.floor(clampedY / cellHeight),
                rows - 1
            );

            grid[rowIndex][colIndex]++;
        }
    });

    return grid;
}

/**
 * Calculate average position
 */
export function calculateAveragePosition(
    positions: [number, number][]
): [number, number] {
    if (positions.length === 0) return [0, 0];

    const sumX = positions.reduce((sum, pos) => sum + pos[0], 0);
    const sumY = positions.reduce((sum, pos) => sum + pos[1], 0);

    return [sumX / positions.length, sumY / positions.length];
}

/**
 * Calculate movement intensity (total distance covered)
 */
export function calculateMovementIntensity(
    positions: [number, number][]
): number {
    if (positions.length < 2) return 0;

    let totalDistance = 0;

    for (let i = 1; i < positions.length; i++) {
        const [x1, y1] = positions[i - 1];
        const [x2, y2] = positions[i];
        const distance = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
        totalDistance += distance;
    }

    return totalDistance;
}

/**
 * Get T-dominance over time for timeline visualization
 */
export function getTDominanceTimeline(
    positions: [number, number][],
    threshold: number = 2.0
): number[] {
    return positions.map((pos) => {
        const distance = distanceFromT(pos);
        return distance <= threshold ? 1 : 0;
    });
}

/**
 * Get aggressiveness over time
 */
export function getAggressivenessTimeline(
    positions: [number, number][]
): number[] {
    return positions.map((pos) => (isInFrontHalf(pos) ? 1 : 0));
}

/**
 * Get T-dominance timeline with timestamps
 * Returns data points with time in seconds and T-dominance percentage
 */
export function getTDominanceTimelineWithTime(
    positionsWithTime: Array<{position: [number, number]; timestamp: number}>,
    threshold: number = 2.0,
    intervalSeconds: number = 1.0
): Array<{time: number; tDominance: number}> {
    if (positionsWithTime.length === 0) return [];

    const startTime = positionsWithTime[0].timestamp;
    const endTime = positionsWithTime[positionsWithTime.length - 1].timestamp;
    const duration = endTime - startTime;

    const timeline: Array<{time: number; tDominance: number}> = [];

    // Use integer-based loop to avoid floating-point precision issues
    const numIntervals = Math.ceil(duration / intervalSeconds);

    for (let i = 0; i <= numIntervals; i++) {
        const t = startTime + i * intervalSeconds;

        // Get all positions within this time interval
        const frameInInterval = positionsWithTime.filter(
            (item) => item.timestamp >= t && item.timestamp < t + intervalSeconds
        );

        if (frameInInterval.length > 0) {
            // Calculate T-dominance percentage for this interval
            const nearTCount = frameInInterval.filter(
                (item) => distanceFromT(item.position) <= threshold
            ).length;
            const tDominance = (nearTCount / frameInInterval.length) * 100;

            // Round time to 2 decimal places to avoid floating-point precision issues
            const relativeTime = Math.round((t - startTime) * 100) / 100;

            timeline.push({
                time: relativeTime,
                tDominance,
            });
        }
    }

    return timeline;
}

/**
 * Get aggressiveness timeline with timestamps
 * Returns data points with time in seconds and aggressiveness percentage
 */
export function getAggressivenessTimelineWithTime(
    positionsWithTime: Array<{position: [number, number]; timestamp: number}>,
    intervalSeconds: number = 1.0
): Array<{time: number; aggressiveness: number}> {
    if (positionsWithTime.length === 0) return [];

    const startTime = positionsWithTime[0].timestamp;
    const endTime = positionsWithTime[positionsWithTime.length - 1].timestamp;
    const duration = endTime - startTime;

    const timeline: Array<{time: number; aggressiveness: number}> = [];

    // Use integer-based loop to avoid floating-point precision issues
    const numIntervals = Math.ceil(duration / intervalSeconds);

    for (let i = 0; i <= numIntervals; i++) {
        const t = startTime + i * intervalSeconds;

        // Get all positions within this time interval
        const frameInInterval = positionsWithTime.filter(
            (item) => item.timestamp >= t && item.timestamp < t + intervalSeconds
        );

        if (frameInInterval.length > 0) {
            // Calculate aggressiveness percentage for this interval
            const frontHalfCount = frameInInterval.filter(
                (item) => isInFrontHalf(item.position)
            ).length;
            const aggressiveness = (frontHalfCount / frameInInterval.length) * 100;

            // Round time to 2 decimal places to avoid floating-point precision issues
            const relativeTime = Math.round((t - startTime) * 100) / 100;

            timeline.push({
                time: relativeTime,
                aggressiveness,
            });
        }
    }

    return timeline;
}
