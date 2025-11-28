"use client";

import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Legend,
} from "recharts";
import type {ShotEffectivenessPerRallyItem} from "@/types/api";
import {CHART_COLORS} from "@/lib/utils/chart-utils";
import {formatDecimal, formatPercentage} from "@/lib/utils/formatters";

interface DualPlayerShotEffectivenessChartProps {
    data: ShotEffectivenessPerRallyItem[];
    title?: string;
    player1Name?: string;
    player2Name?: string;
}

/**
 * Dual-player line chart showing shot effectiveness (opponent distance moved) over rallies
 * Displays both Player 1 and Player 2 shot effectiveness trends
 */
export function DualPlayerShotEffectivenessChart({
    data,
    title,
    player1Name = 'Player 1',
    player2Name = 'Player 2',
}: DualPlayerShotEffectivenessChartProps) {
    if (!data || data.length === 0) {
        return (
            <div className="h-80 flex items-center justify-center text-gray-500">
                No data available
            </div>
        );
    }

    // Transform data for recharts
    const chartData = data.map((rally) => ({
        rally_id: rally.rally_id,
        player1Distance: rally.player_1.avg_opponent_distance_moved,
        player2Distance: rally.player_2.avg_opponent_distance_moved,
        rally_duration: rally.rally_duration,
        shot_count: rally.shot_count,
        // Additional metrics for tooltip
        player1_depth_dominance: rally.player_1.depth_dominance_pct,
        player2_depth_dominance: rally.player_2.depth_dominance_pct,
        player1_displacement_from_t: rally.player_1.avg_displacement_from_t,
        player2_displacement_from_t: rally.player_2.avg_displacement_from_t,
        player1_straight_quality: rally.player_1.straight_shot_quality_pct,
        player2_straight_quality: rally.player_2.straight_shot_quality_pct,
    }));

    return (
        <div>
            {title && (
                <h4 className="text-md font-semibold text-gray-900 mb-4">{title}</h4>
            )}
            <ResponsiveContainer width="100%" height={350}>
                <LineChart
                    data={chartData}
                    margin={{top: 10, right: 30, left: 20, bottom: 30}}
                >
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />

                    <XAxis
                        dataKey="rally_id"
                        label={{
                            value: "Rally Number",
                            position: "insideBottom",
                            offset: -10,
                            style: {textAnchor: "middle"},
                        }}
                        stroke="#6b7280"
                        height={50}
                    />

                    <YAxis
                        label={{
                            value: "Avg Opp. Distance Moved (m)",
                            angle: -90,
                            position: "insideLeft",
                            style: {textAnchor: "middle"},
                        }}
                        stroke="#6b7280"
                        width={80}
                    />

                    <Tooltip
                        content={({active, payload}) => {
                            if (!active || !payload || payload.length === 0)
                                return null;

                            const data = payload[0].payload;

                            return (
                                <div className="bg-white p-4 border border-gray-200 rounded-lg shadow-lg max-w-sm">
                                    <p className="text-sm font-semibold text-gray-900 mb-3">
                                        Rally {data.rally_id}
                                    </p>

                                    {/* Player 1 Metrics */}
                                    <div className="mb-3">
                                        <p
                                            className="text-xs font-semibold mb-1"
                                            style={{
                                                color: CHART_COLORS.player1,
                                            }}
                                        >
                                            {player1Name}
                                        </p>
                                        <div className="space-y-1 text-xs">
                                            <p className="text-gray-700">
                                                Opp Distance:{" "}
                                                <span className="font-semibold">
                                                    {data.player1Distance !==
                                                    null
                                                        ? `${formatDecimal(
                                                              data.player1Distance,
                                                              2
                                                          )} m`
                                                        : "N/A"}
                                                </span>
                                            </p>
                                            <p className="text-gray-700">
                                                Depth Dominance:{" "}
                                                <span className="font-semibold">
                                                    {data.player1_depth_dominance !==
                                                    null
                                                        ? formatPercentage(
                                                              data.player1_depth_dominance
                                                          )
                                                        : "N/A"}
                                                </span>
                                            </p>
                                            <p className="text-gray-700">
                                                Displacement from T:{" "}
                                                <span className="font-semibold">
                                                    {data.player1_displacement_from_t !==
                                                    null
                                                        ? `${formatDecimal(
                                                              data.player1_displacement_from_t,
                                                              2
                                                          )} m`
                                                        : "N/A"}
                                                </span>
                                            </p>
                                            <p className="text-gray-700">
                                                Straight Shot Quality:{" "}
                                                <span className="font-semibold">
                                                    {data.player1_straight_quality !==
                                                    null
                                                        ? formatPercentage(
                                                              data.player1_straight_quality
                                                          )
                                                        : "N/A"}
                                                </span>
                                            </p>
                                        </div>
                                    </div>

                                    {/* Player 2 Metrics */}
                                    <div className="mb-2">
                                        <p
                                            className="text-xs font-semibold mb-1"
                                            style={{
                                                color: CHART_COLORS.player2,
                                            }}
                                        >
                                            {player2Name}
                                        </p>
                                        <div className="space-y-1 text-xs">
                                            <p className="text-gray-700">
                                                Opp Distance:{" "}
                                                <span className="font-semibold">
                                                    {data.player2Distance !==
                                                    null
                                                        ? `${formatDecimal(
                                                              data.player2Distance,
                                                              2
                                                          )} m`
                                                        : "N/A"}
                                                </span>
                                            </p>
                                            <p className="text-gray-700">
                                                Depth Dominance:{" "}
                                                <span className="font-semibold">
                                                    {data.player2_depth_dominance !==
                                                    null
                                                        ? formatPercentage(
                                                              data.player2_depth_dominance
                                                          )
                                                        : "N/A"}
                                                </span>
                                            </p>
                                            <p className="text-gray-700">
                                                Displacement from T:{" "}
                                                <span className="font-semibold">
                                                    {data.player2_displacement_from_t !==
                                                    null
                                                        ? `${formatDecimal(
                                                              data.player2_displacement_from_t,
                                                              2
                                                          )} m`
                                                        : "N/A"}
                                                </span>
                                            </p>
                                            <p className="text-gray-700">
                                                Straight Shot Quality:{" "}
                                                <span className="font-semibold">
                                                    {data.player2_straight_quality !==
                                                    null
                                                        ? formatPercentage(
                                                              data.player2_straight_quality
                                                          )
                                                        : "N/A"}
                                                </span>
                                            </p>
                                        </div>
                                    </div>

                                    {/* Rally Info */}
                                    <div className="pt-2 mt-2 border-t border-gray-200 space-y-1 text-xs">
                                        <p className="text-gray-700">
                                            Shots:{" "}
                                            <span className="font-semibold">
                                                {data.shot_count}
                                            </span>
                                        </p>
                                        <p className="text-gray-700">
                                            Duration:{" "}
                                            <span className="font-semibold">
                                                {data.rally_duration.toFixed(1)}
                                                s
                                            </span>
                                        </p>
                                    </div>
                                </div>
                            );
                        }}
                    />

                    <Legend
                        wrapperStyle={{paddingTop: "20px"}}
                        iconType="line"
                    />

                    <Line
                        type="monotone"
                        dataKey="player1Distance"
                        name={player1Name}
                        stroke={CHART_COLORS.player1}
                        strokeWidth={2}
                        dot={{r: 3}}
                        activeDot={{r: 5}}
                        connectNulls
                    />

                    <Line
                        type="monotone"
                        dataKey="player2Distance"
                        name={player2Name}
                        stroke={CHART_COLORS.player2}
                        strokeWidth={2}
                        dot={{r: 3}}
                        activeDot={{r: 5}}
                        connectNulls
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}
