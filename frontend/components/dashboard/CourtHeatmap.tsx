"use client";

import {useMemo} from "react";

interface CourtHeatmapProps {
    heatmapData: number[][];
    title?: string;
    width?: number;
    height?: number;
}

export default function CourtHeatmap({
    heatmapData,
    title,
    width = 400,
    height = 600,
}: CourtHeatmapProps) {
    // Find max value for normalization
    const maxValue = useMemo(() => {
        let max = 0;
        heatmapData.forEach((row) => {
            row.forEach((val) => {
                if (val > max) max = val;
            });
        });
        return max || 1;
    }, [heatmapData]);

    const rows = heatmapData.length;
    const cols = heatmapData[0]?.length || 0;

    const cellWidth = width / cols;
    const cellHeight = height / rows;

    // Color scale: from transparent to dark red
    const getColor = (value: number): string => {
        if (value === 0) return "rgba(0, 0, 0, 0.02)";
        const intensity = value / maxValue;
        // Use dark red (#8B0000) with varying opacity
        return `rgba(139, 0, 0, ${0.2 + intensity * 0.8})`;
    };

    return (
        <div className="flex flex-col items-center">
            {title && (
                <h3 className="text-lg font-semibold text-black mb-4">
                    {title}
                </h3>
            )}
            <div className="relative border-2 border-black/20 rounded-lg overflow-hidden bg-cream/50">
                <svg width={width} height={height}>
                    {/* Draw heatmap cells */}
                    {heatmapData.map((row, rowIndex) =>
                        row.map((value, colIndex) => (
                            <rect
                                key={`${rowIndex}-${colIndex}`}
                                x={colIndex * cellWidth}
                                y={rowIndex * cellHeight}
                                width={cellWidth}
                                height={cellHeight}
                                fill={getColor(value)}
                                stroke="rgba(0, 0, 0, 0.05)"
                                strokeWidth="0.5"
                            />
                        ))
                    )}

                    {/* Draw court lines - Top-down view of squash court */}

                    {/* Short line (divides front and back court) */}
                    {/* In squash, this is at 5.49m from front wall, roughly 55% of court length */}
                    <line
                        x1={0}
                        y1={height * 0.55}
                        x2={width}
                        y2={height * 0.55}
                        stroke="rgba(0, 0, 0, 0.5)"
                        strokeWidth="2"
                    />

                    {/* Half-court line (from back wall to short line) */}
                    <line
                        x1={width / 2}
                        y1={height * 0.55}
                        x2={width / 2}
                        y2={height}
                        stroke="rgba(0, 0, 0, 0.4)"
                        strokeWidth="2"
                    />

                    {/* Service boxes - 1.6m x 1.6m squares along the side walls */}
                    {/* Court is 9.75m long, 6.4m wide */}
                    {/* Boxes start at short line (5.49m) and extend 1.6m back to 7.09m */}
                    {/* Width: 1.6m out of 6.4m = 25% of court width */}
                    {/* Height: 1.6m out of 9.75m = 16.4% of court length */}

                    {/* Service box - Left side (touches short line) */}
                    <rect
                        x={0}
                        y={height * 0.55}
                        width={width * 0.25}
                        height={height * 0.164}
                        fill="none"
                        stroke="rgba(0, 0, 0, 0.4)"
                        strokeWidth="2"
                    />

                    {/* Service box - Right side (touches short line) */}
                    <rect
                        x={width * 0.75}
                        y={height * 0.55}
                        width={width * 0.25}
                        height={height * 0.164}
                        fill="none"
                        stroke="rgba(0, 0, 0, 0.4)"
                        strokeWidth="2"
                    />

                    {/* Front wall label */}
                    <text
                        x={width / 2}
                        y={20}
                        textAnchor="middle"
                        className="text-xs font-semibold"
                        fill="rgba(0, 0, 0, 0.6)"
                    >
                        Front Wall
                    </text>

                    {/* Back wall label */}
                    <text
                        x={width / 2}
                        y={height - 10}
                        textAnchor="middle"
                        className="text-xs font-semibold"
                        fill="rgba(0, 0, 0, 0.6)"
                    >
                        Back Wall
                    </text>
                </svg>
            </div>
            <div className="mt-4 flex items-center gap-2 text-sm text-black/60">
                <span>Low Activity</span>
                <div className="flex gap-1">
                    {[0.2, 0.4, 0.6, 0.8, 1.0].map((intensity) => (
                        <div
                            key={intensity}
                            className="w-8 h-4 rounded"
                            style={{
                                backgroundColor: `rgba(139, 0, 0, ${
                                    0.2 + intensity * 0.8
                                })`,
                            }}
                        />
                    ))}
                </div>
                <span>High Activity</span>
            </div>
        </div>
    );
}
