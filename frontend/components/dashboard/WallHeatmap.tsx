'use client';

import { useMemo } from 'react';

interface WallHeatmapProps {
  heatmapData: number[][];
  title?: string;
  width?: number;
  height?: number;
}

export default function WallHeatmap({
  heatmapData,
  title,
  width = 400,
  height = 300,
}: WallHeatmapProps) {
  // Find max value for normalization
  const maxValue = useMemo(() => {
    let max = 0;
    heatmapData.forEach(row => {
      row.forEach(val => {
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
    if (value === 0) return 'rgba(0, 0, 0, 0.02)';
    const intensity = value / maxValue;
    return `rgba(139, 0, 0, ${0.2 + intensity * 0.8})`;
  };

  return (
    <div className="flex flex-col items-center">
      {title && <h3 className="text-lg font-semibold text-black mb-4">{title}</h3>}
      <div className="relative border-2 border-black/20 rounded-lg overflow-hidden bg-white">
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

          {/* Draw wall markings */}
          {/* Service line (approximately 1.83m high) */}
          <line
            x1={0}
            y1={height * 0.6}
            x2={width}
            y2={height * 0.6}
            stroke="rgba(0, 0, 0, 0.3)"
            strokeWidth="2"
            strokeDasharray="4,4"
          />

          {/* Tin line (approximately 0.48m high) */}
          <line
            x1={0}
            y1={height * 0.9}
            x2={width}
            y2={height * 0.9}
            stroke="rgba(139, 0, 0, 0.5)"
            strokeWidth="2"
          />

          {/* Out line (top) */}
          <line
            x1={0}
            y1={10}
            x2={width}
            y2={10}
            stroke="rgba(0, 0, 0, 0.3)"
            strokeWidth="2"
            strokeDasharray="4,4"
          />
        </svg>
      </div>
      <div className="mt-4 flex items-center gap-2 text-sm text-black/60">
        <span>Fewer Hits</span>
        <div className="flex gap-1">
          {[0.2, 0.4, 0.6, 0.8, 1.0].map((intensity) => (
            <div
              key={intensity}
              className="w-8 h-4 rounded"
              style={{ backgroundColor: `rgba(139, 0, 0, ${0.2 + intensity * 0.8})` }}
            />
          ))}
        </div>
        <span>More Hits</span>
      </div>
    </div>
  );
}
