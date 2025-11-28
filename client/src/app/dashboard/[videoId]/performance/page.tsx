"use client";

import {useParams} from "next/navigation";
import {Zap, TrendingUp, Target, Layers} from "lucide-react";
import {DualPlayerKPICard} from "@/components/dashboard/DualPlayerKPICard";
import {PlayerFilterCard} from "@/components/dashboard/PlayerFilterCard";
import {usePlayerNames} from "@/lib/hooks/usePlayerNames";
import {DistributionPieChart} from "@/components/charts/DistributionPieChart";
import {DualPlayerBallSpeedChart} from "@/components/charts/DualPlayerBallSpeedChart";
import {DualPlayerShotEffectivenessChart} from "@/components/charts/DualPlayerShotEffectivenessChart";
import {WinningEfficiencyBarChart} from "@/components/charts/WinningEfficiencyBarChart";
import {ShotTypeRadarChart} from "@/components/charts/ShotTypeRadarChart";
import {SquashWallVisualization} from "@/components/charts/SquashWallVisualization";
import {
    useStrokeDistributionWithPlayer,
    useShotTypeDistributionWithPlayer,
    useWallQuadrantsWithPlayer,
    useWallHitsHeatmapWithPlayer,
    useShotEffectiveness,
    useWinningEfficiencyPerRally,
} from "@/lib/hooks/useAnalytics";
import {useQuery} from "@tanstack/react-query";
import {
    getRhythmDisruption,
    getBallSpeed,
    getBallSpeedPerRally,
    getShotEffectivenessPerRally,
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

    // Per-rally winning efficiency for both players
    const {data: winningEffPerRally, isLoading: winningEffPerRallyLoading} = useWinningEfficiencyPerRally(videoId);

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
        ballSpeedP1Loading ||
        ballSpeedP2Loading ||
        rhythmP1Loading ||
        rhythmP2Loading ||
        ballSpeedPerRallyLoading ||
        shotEffPerRallyLoading ||
        winningEffPerRallyLoading;

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
                    <PlayerFilterCard
                        videoId={videoId}
                        title="Wall Hits Heatmap"
                    >
                        {(playerId) => {
                            const {data: wallHeatmap} = useWallHitsHeatmapWithPlayer(videoId, playerId);
                            const {data: wallQuad} = useWallQuadrantsWithPlayer(videoId, playerId);

                            return (
                                <SquashWallVisualization
                                    heatmapData={wallHeatmap?.data.heatmap_grid}
                                    quadrantData={wallQuad?.data.distribution}
                                    quadrantBoundaries={wallQuad?.quadrant_boundaries}
                                />
                            );
                        }}
                    </PlayerFilterCard>

                    {/* Stroke Distribution Pie Chart */}
                    <PlayerFilterCard
                        videoId={videoId}
                        title="Stroke Distribution"
                    >
                        {(playerId) => {
                            const {data: strokeDist} = useStrokeDistributionWithPlayer(videoId, playerId);

                            return (
                                <DistributionPieChart
                                    data={strokeDist?.data.distribution || []}
                                    title=""
                                />
                            );
                        }}
                    </PlayerFilterCard>

                    {/* Shot Types Radar Chart */}
                    <PlayerFilterCard
                        videoId={videoId}
                        title="Shot Types"
                    >
                        {(playerId) => {
                            const {data: shotTypeDist} = useShotTypeDistributionWithPlayer(videoId, playerId);

                            return (
                                <ShotTypeRadarChart
                                    data={shotTypeDist?.data.distribution || []}
                                    title=""
                                />
                            );
                        }}
                    </PlayerFilterCard>
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

                    {/* Winning Efficiency Over Rallies - Both Players */}
                    <div className="bg-white p-6 rounded-lg border border-gray-200">
                        <WinningEfficiencyBarChart
                            data={winningEffPerRally?.data || []}
                            title="Winning Efficiency Over Rallies"
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
