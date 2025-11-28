"use client";

import {useParams} from "next/navigation";
import {Activity, Clock, Zap, BarChart3, Timer} from "lucide-react";
import {KPICard} from "@/components/dashboard/KPICard";
import {RallyScatterChart} from "@/components/charts/RallyScatterChart";
import {RallyDurationBarChart} from "@/components/charts/RallyDurationBarChart";
import {
    useRallyTimeline,
    useRallyIntensity,
    useLongestRally,
} from "@/lib/hooks/useAnalytics";
import {formatDecimal} from "@/lib/utils/formatters";

/**
 * Rally Analysis Tab
 *
 * Focuses on:
 * - Rally patterns and timeline
 * - Rally intensity and pace
 * - Shot count and duration analysis
 * - Point winning efficiency
 */
export default function RalliesPage() {
    const params = useParams();
    const videoId = params.videoId as string;

    const {data: rallyTimeline, isLoading: timelineLoading} =
        useRallyTimeline(videoId);
    const {data: intensity, isLoading: intensityLoading} =
        useRallyIntensity(videoId);
    const {data: longestRally, isLoading: longestLoading} =
        useLongestRally(videoId);

    const isLoading = timelineLoading || intensityLoading || longestLoading;

    if (isLoading) {
        return (
            <div className="p-8 space-y-6">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
                    {[...Array(5)].map((_, i) => (
                        <div
                            key={i}
                            className="h-32 bg-gray-200 rounded-lg animate-pulse"
                        />
                    ))}
                </div>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {[...Array(2)].map((_, i) => (
                        <div
                            key={i}
                            className="h-96 bg-gray-200 rounded-lg animate-pulse"
                        />
                    ))}
                </div>
            </div>
        );
    }

    // Calculate statistics
    const avgRallyDuration =
        rallyTimeline?.data && rallyTimeline.data.length > 0
            ? rallyTimeline.data.reduce(
                  (sum, rally) => sum + rally.rally_duration,
                  0
              ) / rallyTimeline.data.length
            : 0;

    const avgShotCount =
        rallyTimeline?.data && rallyTimeline.data.length > 0
            ? rallyTimeline.data.reduce(
                  (sum, rally) => sum + rally.shot_count,
                  0
              ) / rallyTimeline.data.length
            : 0;

    return (
        <div className="p-8 space-y-8">
            {/* Section 1: Overview KPIs */}
            <div>
                <h2 className="text-xl font-bold text-gray-900 mb-4">
                    Rally Overview
                </h2>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-6">
                    <KPICard
                        title="Total Rallies"
                        value={rallyTimeline?.total_rallies || 0}
                        icon={Activity}
                        subtitle="Match total"
                        infoTooltip="The total number of rallies played in the match. Each rally ends when a player wins a point."
                    />

                    <KPICard
                        title="Avg Rally Duration"
                        value={formatDecimal(avgRallyDuration, 1) + "s"}
                        icon={Clock}
                        subtitle={
                            rallyTimeline?.data
                                ? `${rallyTimeline.data.length} rallies`
                                : undefined
                        }
                        infoTooltip="The average time length of rallies across the match, measured in seconds from the first shot to the point being won."
                    />

                    <KPICard
                        title="Avg Shot Count"
                        value={formatDecimal(avgShotCount, 1)}
                        icon={BarChart3}
                        subtitle="Shots per rally"
                        infoTooltip="The average number of shots per rally. This includes all racket hits by both players until the rally ends."
                    />

                    <KPICard
                        title="Rally Intensity"
                        value={
                            intensity?.data.avg_seconds_per_shot
                                ? formatDecimal(
                                      1 / intensity.data.avg_seconds_per_shot,
                                      2
                                  ) + " shots/s"
                                : "N/A"
                        }
                        icon={Zap}
                        subtitle="Match pace"
                        infoTooltip="The average number of shots per second across all rallies. Higher values indicate faster, more intense rallies with quicker exchanges."
                    />

                    {longestRally?.data && (
                        <KPICard
                            title="Longest Rally"
                            value={`${longestRally.data.shot_count} shots`}
                            icon={Timer}
                            subtitle={`${formatDecimal(
                                longestRally.data.rally_duration,
                                1
                            )}s duration`}
                            infoTooltip="The rally with the most shots in the match, showing both the total shot count and how long it lasted."
                        />
                    )}
                </div>
            </div>

            {/* Section 2: Rally Timeline Visualizations */}
            <div>
                <h2 className="text-xl font-bold text-gray-900 mb-4">
                    Rally Patterns
                </h2>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Scatter Chart */}
                    <div className="bg-white p-6 rounded-lg border border-gray-200">
                        <RallyScatterChart
                            data={rallyTimeline?.data || []}
                            title="Rally Duration vs Shot Count"
                        />
                    </div>

                    {/* Intensity Bar Chart */}
                    <div className="bg-white p-6 rounded-lg border border-gray-200">
                        <RallyDurationBarChart
                            data={rallyTimeline?.data || []}
                            title="Rally Intensity per Rally"
                        />
                    </div>
                </div>
            </div>
        </div>
    );
}
