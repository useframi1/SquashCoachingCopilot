"use client";

import {useState, useMemo} from "react";
import {Video, Clock, Activity, MapPin, List} from "lucide-react";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
} from "recharts";
import {Rally, AnalysisResults} from "@/lib/types";
import StatCard from "./StatCard";
import CourtHeatmap from "./CourtHeatmap";
import WallHeatmap from "./WallHeatmap";
import VideoModal from "./VideoModal";
import {getRallyVideoUrl} from "@/lib/api";
import {
    extractPlayerPositions,
    createHeatmapData,
    createWallHitHeatmap,
    calculateCourtCoverage,
} from "@/lib/courtMetrics";

interface RallyByRallyTabProps {
    results: AnalysisResults;
    jobId: string;
}

export default function RallyByRallyTab({results, jobId}: RallyByRallyTabProps) {
    const [selectedRallyIndex, setSelectedRallyIndex] = useState<number | null>(
        null
    );
    const [showVideoModal, setShowVideoModal] = useState(false);

    const selectedRally =
        selectedRallyIndex !== null
            ? results.rallies[selectedRallyIndex]
            : null;

    return (
        <>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Rally List - Left Sidebar */}
                <div className="lg:col-span-1">
                    <div className="bg-white rounded-xl shadow-lg p-6">
                        <h2 className="text-2xl font-bold text-black mb-6 flex items-center gap-2">
                            <List className="w-6 h-6 text-dark-red" />
                            Rally List
                        </h2>
                        <div className="space-y-3 max-h-[800px] overflow-y-auto">
                            {results.rallies.map((rally, index) => (
                                <RallyListItem
                                    key={index}
                                    fps={results.fps}
                                    rally={rally}
                                    rallyNumber={index + 1}
                                    isSelected={selectedRallyIndex === index}
                                    onClick={() => setSelectedRallyIndex(index)}
                                />
                            ))}
                        </div>
                    </div>
                </div>

                {/* Rally Details - Main Panel */}
                <div className="lg:col-span-2">
                    {selectedRally === null ? (
                        <div className="bg-white rounded-xl shadow-lg p-12 flex items-center justify-center">
                            <div className="text-center text-black/40">
                                <Activity className="w-16 h-16 mx-auto mb-4" />
                                <p className="text-xl">
                                    Select a rally to view detailed analysis
                                </p>
                            </div>
                        </div>
                    ) : (
                        <RallyDetails
                            fps={results.fps}
                            rally={selectedRally}
                            rallyNumber={(selectedRallyIndex ?? 0) + 1}
                            onPlayVideo={() => setShowVideoModal(true)}
                        />
                    )}
                </div>
            </div>

            {/* Video Modal */}
            {showVideoModal && selectedRallyIndex !== null && (
                <VideoModal
                    videoUrl={getRallyVideoUrl(jobId, selectedRallyIndex + 1)}
                    rallyNumber={selectedRallyIndex + 1}
                    onClose={() => setShowVideoModal(false)}
                />
            )}
        </>
    );
}

interface RallyListItemProps {
    fps: number;
    rally: Rally;
    rallyNumber: number;
    isSelected: boolean;
    onClick: () => void;
}

function RallyListItem({
    fps,
    rally,
    rallyNumber,
    isSelected,
    onClick,
}: RallyListItemProps) {
    const duration = ((rally.end_frame - rally.start_frame) / fps).toFixed(1);
    const totalFrames = rally.rally_frames.length;

    // Count shots in this rally (ball wall hits)
    const shots = rally.rally_frames.filter(
        (f) => f.ball.is_wall_hit
    ).length;

    // Calculate intensity (shots per second)
    const intensity = (shots / parseFloat(duration)).toFixed(1);

    return (
        <button
            onClick={onClick}
            className={`w-full p-4 rounded-lg border-2 transition-all text-left ${
                isSelected
                    ? "border-dark-red bg-dark-red/5 shadow-md"
                    : "border-black/10 hover:border-dark-red/50 bg-white"
            }`}
        >
            <div className="flex items-center justify-between mb-2">
                <h3 className="text-lg font-bold text-black">
                    Rally {rallyNumber}
                </h3>
                <Activity
                    className={`w-5 h-5 ${
                        isSelected ? "text-dark-red" : "text-black/40"
                    }`}
                />
            </div>
            <div className="space-y-1 text-sm text-black/60">
                <p>Duration: {duration}s</p>
                <p>Frames: {totalFrames}</p>
                <p>Shots: {shots}</p>
                <div className="mt-2 pt-2 border-t border-black/10">
                    <div className="flex items-center gap-2">
                        <span className="text-xs">Intensity:</span>
                        <div className="flex-1 bg-black/5 rounded-full h-2">
                            <div
                                className="bg-dark-red rounded-full h-2 transition-all"
                                style={{
                                    width: `${Math.min(
                                        (parseFloat(intensity) / 2) * 100,
                                        100
                                    )}%`,
                                }}
                            />
                        </div>
                    </div>
                </div>
            </div>
        </button>
    );
}

interface RallyDetailsProps {
    fps: number;
    rally: Rally;
    rallyNumber: number;
    onPlayVideo: () => void;
}

function RallyDetails({fps, rally, rallyNumber, onPlayVideo}: RallyDetailsProps) {
    const duration = parseFloat(
        ((rally.end_frame - rally.start_frame) / fps).toFixed(1)
    );

    // Count shots (ball wall hits)
    const shots = rally.rally_frames.filter(
        (f) => f.ball.is_wall_hit
    ).length;

    const avgShotFrequency =
        shots > 0 ? (duration / shots).toFixed(1) : "0";

    // Calculate court coverage for this rally
    const player1Positions = extractPlayerPositions(rally.rally_frames, 1);
    const player2Positions = extractPlayerPositions(rally.rally_frames, 2);

    const p1Coverage = calculateCourtCoverage(player1Positions);
    const p2Coverage = calculateCourtCoverage(player2Positions);

    // Create heatmaps
    const p1Heatmap = useMemo(
        () => createHeatmapData(player1Positions),
        [player1Positions]
    );
    const p2Heatmap = useMemo(
        () => createHeatmapData(player2Positions),
        [player2Positions]
    );

    // Create wall hit heatmap
    const wallHeatmap = useMemo(
        () => createWallHitHeatmap(rally.rally_frames),
        [rally.rally_frames]
    );

    // Position timeline data
    const positionTimelineData = useMemo(() => {
        const sampleInterval = Math.max(
            1,
            Math.floor(rally.rally_frames.length / 100)
        );
        return rally.rally_frames
            .filter((_, index) => index % sampleInterval === 0)
            .map((frame) => ({
                time: ((frame.frame_number - rally.start_frame) / fps).toFixed(2),
                player1Y: frame.player1.real_position[1],
                player2Y: frame.player2.real_position[1],
            }));
    }, [rally.rally_frames, rally.start_frame, fps]);

    return (
        <div className="space-y-6">
            {/* Play Video Button */}
            <div className="bg-white rounded-xl shadow-lg p-6">
                <button
                    onClick={onPlayVideo}
                    className="w-full flex items-center justify-center gap-3 px-6 py-4 bg-dark-red text-white rounded-lg font-bold text-lg hover:bg-dark-red/90 transition-all shadow-md hover:shadow-lg"
                >
                    <Video className="w-6 h-6" />
                    <span>Play Rally {rallyNumber} Video</span>
                </button>
            </div>

            {/* Stats Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard
                    title="Duration"
                    value={duration}
                    unit="sec"
                    icon={<Clock className="w-5 h-5" />}
                    primary
                />
                <StatCard
                    title="Total Shots"
                    value={shots}
                    icon={<Activity className="w-5 h-5" />}
                />
                <StatCard
                    title="Avg Shot Freq"
                    value={avgShotFrequency}
                    unit="s"
                    icon={<Activity className="w-5 h-5" />}
                />
                <StatCard
                    title="Rally Intensity"
                    value={shots > 0 ? (shots / duration).toFixed(1) : "0"}
                    unit="s/s"
                    icon={<MapPin className="w-5 h-5" />}
                />
            </div>

            {/* Player Movement on Court */}
            <div className="bg-white rounded-xl shadow-lg p-6">
                <h2 className="text-2xl font-bold text-black mb-6">
                    Player Movement on Court
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div className="flex flex-col items-center">
                        <h3 className="text-lg font-semibold text-black mb-2">
                            Player 1
                        </h3>
                        <p className="text-sm text-black/60 mb-4">
                            Coverage: {p1Coverage.toFixed(1)}%
                        </p>
                        <CourtHeatmap
                            heatmapData={p1Heatmap}
                            width={250}
                            height={375}
                        />
                    </div>
                    <div className="flex flex-col items-center">
                        <h3 className="text-lg font-semibold text-black mb-2">
                            Player 2
                        </h3>
                        <p className="text-sm text-black/60 mb-4">
                            Coverage: {p2Coverage.toFixed(1)}%
                        </p>
                        <CourtHeatmap
                            heatmapData={p2Heatmap}
                            width={250}
                            height={375}
                        />
                    </div>
                </div>
            </div>

            {/* Wall Hit Heatmap */}
            <div className="bg-white rounded-xl shadow-lg p-6">
                <h2 className="text-2xl font-bold text-black mb-6">
                    Front Wall Hit Distribution
                </h2>
                <div className="flex justify-center">
                    <WallHeatmap
                        heatmapData={wallHeatmap}
                        width={500}
                        height={350}
                    />
                </div>
            </div>

            {/* Position Timeline */}
            <div className="bg-white rounded-xl shadow-lg p-6">
                <h2 className="text-2xl font-bold text-black mb-6">
                    Front-to-Back Position Timeline
                </h2>
                <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={positionTimelineData}>
                        <CartesianGrid
                            strokeDasharray="3 3"
                            stroke="#00000010"
                        />
                        <XAxis
                            dataKey="time"
                            stroke="#000000"
                            tick={{fill: "#000000"}}
                            label={{
                                value: "Time (seconds)",
                                position: "insideBottom",
                                offset: -5,
                            }}
                            type="number"
                            domain={['dataMin', 'dataMax']}
                        />
                        <YAxis
                            stroke="#000000"
                            tick={{fill: "#000000"}}
                            label={{
                                value: "Court Position (m)",
                                angle: -90,
                                position: "insideLeft",
                            }}
                            domain={[0, 10]}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: "#FFFFFF",
                                border: "1px solid #00000020",
                                borderRadius: "8px",
                            }}
                            formatter={(value: number) =>
                                `${value.toFixed(2)}m`
                            }
                            labelFormatter={(value: number) =>
                                `Time: ${value}s`
                            }
                        />
                        <Line
                            type="monotone"
                            dataKey="player1Y"
                            stroke="#8B0000"
                            strokeWidth={2}
                            dot={false}
                            name="Player 1"
                        />
                        <Line
                            type="monotone"
                            dataKey="player2Y"
                            stroke="#000000"
                            strokeWidth={2}
                            dot={false}
                            name="Player 2"
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}
