"use client";

import {useParams} from "next/navigation";
import {Zap, TrendingUp, Target, Layers} from "lucide-react";
import {DualPlayerKPICard} from "@/components/dashboard/DualPlayerKPICard";
import {usePlayerNames} from "@/lib/hooks/usePlayerNames";
import {DistributionPieChart} from "@/components/charts/DistributionPieChart";
import {DualPlayerBallSpeedChart} from "@/components/charts/DualPlayerBallSpeedChart";
import {DualPlayerShotEffectivenessChart} from "@/components/charts/DualPlayerShotEffectivenessChart";
import {DualPlayerTZoneOccupancyChart} from "@/components/charts/DualPlayerTZoneOccupancyChart";
import {ShotTypeRadarChart} from "@/components/charts/ShotTypeRadarChart";
import {SquashWallVisualization} from "@/components/charts/SquashWallVisualization";
import {
    useStrokeDistribution,
    useShotTypeDistribution,
    useWallQuadrants,
    useWallHitsHeatmap,
    useShotEffectiveness,
} from "@/lib/hooks/useAnalytics";
import {useQuery} from "@tanstack/react-query";
import {
    getRhythmDisruption,
    getBallSpeed,
    getBallSpeedPerRally,
    getShotEffectivenessPerRally,
    getTZoneOccupancyPerRally,
} from "@/lib/api/analytics";
import {useFilterStore} from "@/lib/stores/filterStore";
import {
    formatSpeed,
    formatDecimal,
    formatPercentage,
} from "@/lib/utils/formatters";

/**
 * Performance & Shot Analysis tab
 * Shows stroke patterns, shot types, ball speed, and placement quality
 */
export default function PerformancePage() {
    const params = useParams();
    const videoId = params.videoId as string;
    const {gameNumber, startTime, endTime} = useFilterStore();
    const {player1Name, player2Name} = usePlayerNames(videoId);

    const {data: strokeDist, isLoading: strokeLoading} =
        useStrokeDistribution(videoId);
    const {data: shotTypeDist, isLoading: shotTypeLoading} =
        useShotTypeDistribution(videoId);
    const {data: wallQuad, isLoading: wallQuadLoading} =
        useWallQuadrants(videoId);
    const {data: wallHeatmap, isLoading: wallHeatmapLoading} =
        useWallHitsHeatmap(videoId);

    // Per-rally ball speed for both players
    const {data: ballSpeedPerRally, isLoading: ballSpeedPerRallyLoading} =
        useQuery({
            queryKey: [
                "ball-speed-per-rally",
                videoId,
                gameNumber,
                startTime,
                endTime,
            ],
            queryFn: () =>
                getBallSpeedPerRally(videoId, {
                    game_number: gameNumber ?? undefined,
                    start_time: startTime ?? undefined,
                    end_time: endTime ?? undefined,
                }),
            enabled: !!videoId,
        });

    // Per-rally shot effectiveness for both players
    const {data: shotEffPerRally, isLoading: shotEffPerRallyLoading} = useQuery(
        {
            queryKey: [
                "shot-effectiveness-per-rally",
                videoId,
                gameNumber,
                startTime,
                endTime,
            ],
            queryFn: () =>
                getShotEffectivenessPerRally(videoId, {
                    game_number: gameNumber ?? undefined,
                    start_time: startTime ?? undefined,
                    end_time: endTime ?? undefined,
                }),
            enabled: !!videoId,
        }
    );

    // Per-rally T-zone occupancy for both players
    const {data: tZonePerRally, isLoading: tZonePerRallyLoading} = useQuery({
        queryKey: [
            "t-zone-occupancy-per-rally",
            videoId,
            gameNumber,
            startTime,
            endTime,
        ],
        queryFn: () =>
            getTZoneOccupancyPerRally(videoId, {
                game_number: gameNumber ?? undefined,
                start_time: startTime ?? undefined,
                end_time: endTime ?? undefined,
            }),
        enabled: !!videoId,
    });

    // Ball speed for both players
    const {data: ballSpeedP1, isLoading: ballSpeedP1Loading} = useQuery({
        queryKey: ["ball-speed-p1", videoId, gameNumber, startTime, endTime],
        queryFn: () =>
            getBallSpeed(videoId, {
                player_id: 1,
                game_number: gameNumber ?? undefined,
                start_time: startTime ?? undefined,
                end_time: endTime ?? undefined,
            }),
        enabled: !!videoId,
    });

    const {data: ballSpeedP2, isLoading: ballSpeedP2Loading} = useQuery({
        queryKey: ["ball-speed-p2", videoId, gameNumber, startTime, endTime],
        queryFn: () =>
            getBallSpeed(videoId, {
                player_id: 2,
                game_number: gameNumber ?? undefined,
                start_time: startTime ?? undefined,
                end_time: endTime ?? undefined,
            }),
        enabled: !!videoId,
    });

    // Shot effectiveness for both players
    const {data: shotEffP1} = useShotEffectiveness(videoId, 1);
    const {data: shotEffP2} = useShotEffectiveness(videoId, 2);

    // Rhythm disruption for both players
    const {data: rhythmP1, isLoading: rhythmP1Loading} = useQuery({
        queryKey: [
            "rhythm-disruption-p1",
            videoId,
            gameNumber,
            startTime,
            endTime,
        ],
        queryFn: () =>
            getRhythmDisruption(videoId, {
                player_id: 1,
                game_number: gameNumber ?? undefined,
                start_time: startTime ?? undefined,
                end_time: endTime ?? undefined,
            }),
        enabled: !!videoId,
    });

    const {data: rhythmP2, isLoading: rhythmP2Loading} = useQuery({
        queryKey: [
            "rhythm-disruption-p2",
            videoId,
            gameNumber,
            startTime,
            endTime,
        ],
        queryFn: () =>
            getRhythmDisruption(videoId, {
                player_id: 2,
                game_number: gameNumber ?? undefined,
                start_time: startTime ?? undefined,
                end_time: endTime ?? undefined,
            }),
        enabled: !!videoId,
    });

    const isLoading =
        strokeLoading ||
        shotTypeLoading ||
        ballSpeedP1Loading ||
        ballSpeedP2Loading ||
        rhythmP1Loading ||
        rhythmP2Loading ||
        wallQuadLoading ||
        wallHeatmapLoading ||
        ballSpeedPerRallyLoading ||
        shotEffPerRallyLoading ||
        tZonePerRallyLoading;

    if (isLoading) {
        return (
            <div className="p-8 space-y-6">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    {[...Array(4)].map((_, i) => (
                        <div
                            key={i}
                            className="h-32 bg-gray-200 rounded-lg animate-pulse"
                        />
                    ))}
                </div>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {[...Array(4)].map((_, i) => (
                        <div
                            key={i}
                            className="h-80 bg-gray-200 rounded-lg animate-pulse"
                        />
                    ))}
                </div>
            </div>
        );
    }

    return (
        <div className="p-8 space-y-8">
            {/* KPI Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                {/* Depth Dominance - Both Players */}
                <DualPlayerKPICard
                    title="Depth Dominance"
                    player1Value={
                        shotEffP1?.data.depth_dominance_pct
                            ? formatPercentage(
                                  shotEffP1.data.depth_dominance_pct
                              )
                            : "N/A"
                    }
                    player2Value={
                        shotEffP2?.data.depth_dominance_pct
                            ? formatPercentage(
                                  shotEffP2.data.depth_dominance_pct
                              )
                            : "N/A"
                    }
                    player1Name={player1Name}
                    player2Name={player2Name}
                    icon={Layers}
                    subtitle="% keeping opponent deep"
                />

                {/* Rhythm Disruption - Both Players */}
                <DualPlayerKPICard
                    title="Rhythm Disruption"
                    player1Value={
                        rhythmP1?.data.ball_speed_cv
                            ? formatDecimal(rhythmP1.data.ball_speed_cv, 2)
                            : "N/A"
                    }
                    player2Value={
                        rhythmP2?.data.ball_speed_cv
                            ? formatDecimal(rhythmP2.data.ball_speed_cv, 2)
                            : "N/A"
                    }
                    player1Name={player1Name}
                    player2Name={player2Name}
                    icon={TrendingUp}
                    subtitle="Coefficient of Variation"
                />

                {/* Avg Ball Speed - Both Players */}
                <DualPlayerKPICard
                    title="Avg Ball Speed"
                    player1Value={
                        ballSpeedP1?.data.mean_speed
                            ? formatSpeed(ballSpeedP1.data.mean_speed)
                            : "N/A"
                    }
                    player2Value={
                        ballSpeedP2?.data.mean_speed
                            ? formatSpeed(ballSpeedP2.data.mean_speed)
                            : "N/A"
                    }
                    player1Name={player1Name}
                    player2Name={player2Name}
                    icon={Zap}
                    subtitle="Average ball speed"
                />

                {/* Straight Shot Quality - Both Players */}
                <DualPlayerKPICard
                    title="Straight Shot Quality"
                    player1Value={
                        shotEffP1?.data.straight_shot_quality_pct
                            ? formatPercentage(
                                  shotEffP1.data.straight_shot_quality_pct
                              )
                            : "N/A"
                    }
                    player2Value={
                        shotEffP2?.data.straight_shot_quality_pct
                            ? formatPercentage(
                                  shotEffP2.data.straight_shot_quality_pct
                              )
                            : "N/A"
                    }
                    player1Name={player1Name}
                    player2Name={player2Name}
                    icon={Target}
                    subtitle="% of straight shots close to wall"
                />
            </div>

            {/* Main Content Grid: Left Column (narrow) and Right Column (wide) */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Column - Stroke, Shot Types, and Wall Visualization */}
                <div className="lg:col-span-1 space-y-6">
                    {/* Squash Wall Visualization */}
                    <div className="bg-white p-6 rounded-lg border border-gray-200">
                        <SquashWallVisualization
                            heatmapData={wallHeatmap?.data.heatmap_grid}
                            quadrantData={wallQuad?.data.distribution}
                            quadrantBoundaries={wallQuad?.quadrant_boundaries}
                        />
                    </div>

                    {/* Stroke Distribution Pie Chart */}
                    <div className="bg-white p-6 rounded-lg border border-gray-200">
                        <DistributionPieChart
                            data={strokeDist?.data.distribution || []}
                            title="Stroke Distribution"
                        />
                    </div>

                    {/* Shot Types Radar Chart */}
                    <div className="bg-white p-6 rounded-lg border border-gray-200">
                        <ShotTypeRadarChart
                            data={shotTypeDist?.data.distribution || []}
                            title="Shot Types"
                        />
                    </div>
                </div>

                {/* Right Column - Time Series Charts */}
                <div className="lg:col-span-2 space-y-6">
                    {/* Shot Effectiveness Over Time - Both Players */}
                    <div className="bg-white p-6 rounded-lg border border-gray-200">
                        <DualPlayerShotEffectivenessChart
                            data={shotEffPerRally?.data || []}
                            title="Shot Effectiveness Over Rallies"
                            player1Name={player1Name}
                            player2Name={player2Name}
                        />
                    </div>

                    {/* T-Zone Occupancy Over Time - Both Players */}
                    <div className="bg-white p-6 rounded-lg border border-gray-200">
                        <DualPlayerTZoneOccupancyChart
                            data={tZonePerRally?.data || []}
                            title="T-Zone Occupancy Over Rallies"
                            player1Name={player1Name}
                            player2Name={player2Name}
                        />
                    </div>

                    {/* Ball Speed Over Time - Both Players */}
                    <div className="bg-white p-6 rounded-lg border border-gray-200">
                        <DualPlayerBallSpeedChart
                            data={ballSpeedPerRally?.data || []}
                            title="Ball Speed Over Rallies"
                            player1Name={player1Name}
                            player2Name={player2Name}
                        />
                    </div>
                </div>
            </div>
        </div>
    );
}
