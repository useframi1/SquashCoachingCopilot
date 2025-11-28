'use client';

import { useState } from 'react';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import type { HeatmapGrid } from '@/types/api';

interface QuadrantData {
  label: string;
  count: number;
  percentage: number;
}

interface SquashWallVisualizationProps {
  heatmapData?: HeatmapGrid;
  quadrantData?: QuadrantData[];
  quadrantBoundaries?: {
    x_cut: number;
    y_cut: number;
  };
}

/**
 * Squash wall visualization with toggle between heatmap and quadrants
 * Displays the front wall of a squash court with wall hit data
 */
export function SquashWallVisualization({
  heatmapData,
  quadrantData,
  quadrantBoundaries,
}: SquashWallVisualizationProps) {
  const [viewMode, setViewMode] = useState<'heatmap' | 'quadrants'>('quadrants');

  // Wall dimensions (in meters, standard squash court)
  const wallWidth = 6.4; // meters
  const wallHeight = 4.57; // meters
  const tinHeight = 0.48; // tin height
  const serviceLineHeight = 1.78; // service line height

  // Canvas dimensions
  const canvasWidth = 600;
  const canvasHeight = 430;

  // Scale factors
  const scaleX = canvasWidth / wallWidth;
  const scaleY = canvasHeight / wallHeight;

  // Downsample heatmap grid to reduce cell count
  const downsampleHeatmap = (grid: number[][], targetSize: number = 20) => {
    if (!grid || grid.length === 0) return { grid: [], width: 0, height: 0 };

    const originalHeight = grid.length;
    const originalWidth = grid[0].length;

    // Calculate downsample factors
    const rowFactor = Math.ceil(originalHeight / targetSize);
    const colFactor = Math.ceil(originalWidth / targetSize);

    const newHeight = Math.ceil(originalHeight / rowFactor);
    const newWidth = Math.ceil(originalWidth / colFactor);

    const downsampled: number[][] = [];

    for (let i = 0; i < newHeight; i++) {
      const row: number[] = [];
      for (let j = 0; j < newWidth; j++) {
        // Average the values in the block
        let sum = 0;
        let count = 0;

        for (let di = 0; di < rowFactor; di++) {
          for (let dj = 0; dj < colFactor; dj++) {
            const srcRow = i * rowFactor + di;
            const srcCol = j * colFactor + dj;

            if (srcRow < originalHeight && srcCol < originalWidth) {
              sum += grid[srcRow][srcCol];
              count++;
            }
          }
        }

        row.push(count > 0 ? sum / count : 0);
      }
      downsampled.push(row);
    }

    return { grid: downsampled, width: newWidth, height: newHeight };
  };

  // Get color for heatmap intensity using theme colors
  const getHeatmapColor = (intensity: number, maxIntensity: number) => {
    if (intensity === 0) return 'rgba(0, 0, 0, 0)';

    // Normalize based on actual max intensity in the data
    const normalized = maxIntensity > 0 ? intensity / maxIntensity : 0;

    // Map to red color scale from theme with better contrast
    if (normalized < 0.2) {
      return '#fee2e2'; // Very light red
    } else if (normalized < 0.4) {
      return '#fca5a5'; // Light red
    } else if (normalized < 0.6) {
      return '#f87171'; // Medium red
    } else if (normalized < 0.8) {
      return '#ef4444'; // Red
    } else {
      return '#dc2626'; // Dark red
    }
  };

  // Get quadrant position and label
  const getQuadrantInfo = (label: string) => {
    const upper = label.toLowerCase().includes('upper') || label.toLowerCase().includes('top');
    const left = label.toLowerCase().includes('left');

    return {
      upper,
      left,
      label: label,
    };
  };

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-md font-semibold text-gray-900">Wall Hit Distribution</h4>
        <Tabs value={viewMode} onValueChange={(v) => setViewMode(v as 'heatmap' | 'quadrants')}>
          <TabsList>
            <TabsTrigger value="quadrants">Quadrants</TabsTrigger>
            <TabsTrigger value="heatmap">Heatmap</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      <div className="relative bg-white border border-gray-200 rounded-lg p-4">
        <svg
          width={canvasWidth}
          height={canvasHeight}
          viewBox={`0 0 ${canvasWidth} ${canvasHeight}`}
          className="mx-auto"
        >
          {/* Wall background */}
          <rect
            x={0}
            y={0}
            width={canvasWidth}
            height={canvasHeight}
            fill="#f9fafb"
            stroke="#d1d5db"
            strokeWidth={2}
          />

          {/* Render heatmap if in heatmap mode */}
          {viewMode === 'heatmap' && heatmapData && (() => {
            // Downsample to 20x20 grid for cleaner visualization
            const { grid: downsampledGrid, width: dsWidth, height: dsHeight } =
              downsampleHeatmap(heatmapData.grid, 20);

            // Find max intensity for normalization
            const maxIntensity = Math.max(
              ...downsampledGrid.flat().filter(v => v > 0)
            );

            return (
              <>
                {downsampledGrid.map((row, rowIndex) =>
                  row.map((intensity, colIndex) => {
                    const cellWidth = canvasWidth / dsWidth;
                    const cellHeight = canvasHeight / dsHeight;
                    const x = colIndex * cellWidth;
                    // Flip y-axis so origin is at bottom-left
                    const y = canvasHeight - (rowIndex + 1) * cellHeight;

                    return (
                      <rect
                        key={`heat-${rowIndex}-${colIndex}`}
                        x={x}
                        y={y}
                        width={cellWidth}
                        height={cellHeight}
                        fill={getHeatmapColor(intensity, maxIntensity)}
                      />
                    );
                  })
                )}
              </>
            );
          })()}

          {/* Render quadrants if in quadrants mode */}
          {viewMode === 'quadrants' && quadrantData && quadrantBoundaries && (
            <>
              {quadrantData.map((quad, index) => {
                const info = getQuadrantInfo(quad.label);
                const x = info.left ? 0 : canvasWidth / 2;
                const y = info.upper ? 0 : canvasHeight / 2;
                const w = canvasWidth / 2;
                const h = canvasHeight / 2;

                // Color based on percentage using theme red (lighter for lower, darker for higher)
                const opacity = 0.2 + (quad.percentage / 100) * 0.5;
                const color = `rgba(185, 28, 28, ${opacity})`; // #b91c1c (player1 red)

                return (
                  <g key={`quad-${index}`}>
                    <rect
                      x={x}
                      y={y}
                      width={w}
                      height={h}
                      fill={color}
                      stroke="#b91c1c"
                      strokeWidth={1}
                    />
                    <text
                      x={x + w / 2}
                      y={y + h / 2 - 10}
                      textAnchor="middle"
                      className="text-sm font-semibold"
                      fill="#1f2937"
                    >
                      {quad.percentage.toFixed(1)}%
                    </text>
                    <text
                      x={x + w / 2}
                      y={y + h / 2 + 10}
                      textAnchor="middle"
                      className="text-xs"
                      fill="#6b7280"
                    >
                      ({quad.count} shots)
                    </text>
                  </g>
                );
              })}
            </>
          )}

          {/* Tin line (red) */}
          <line
            x1={0}
            y1={canvasHeight - (tinHeight * scaleY)}
            x2={canvasWidth}
            y2={canvasHeight - (tinHeight * scaleY)}
            stroke="#ef4444"
            strokeWidth={3}
          />

          {/* Service line */}
          <line
            x1={0}
            y1={canvasHeight - (serviceLineHeight * scaleY)}
            x2={canvasWidth}
            y2={canvasHeight - (serviceLineHeight * scaleY)}
            stroke="#9ca3af"
            strokeWidth={2}
            strokeDasharray="5,5"
          />

          {/* Center dividing line */}
          <line
            x1={canvasWidth / 2}
            y1={0}
            x2={canvasWidth / 2}
            y2={canvasHeight}
            stroke="#9ca3af"
            strokeWidth={2}
            strokeDasharray="5,5"
          />

          {/* Labels */}
          <text
            x={10}
            y={canvasHeight - 5}
            className="text-xs font-medium"
            fill="#ef4444"
          >
            Tin
          </text>
          <text
            x={10}
            y={canvasHeight - (serviceLineHeight * scaleY) - 5}
            className="text-xs"
            fill="#6b7280"
          >
            Service Line
          </text>
        </svg>

        {/* Legend for heatmap */}
        {viewMode === 'heatmap' && (
          <div className="mt-4 flex items-center justify-center gap-2">
            <span className="text-xs text-gray-600">Low</span>
            <div className="flex h-4 w-48">
              {[0.1, 0.3, 0.5, 0.7, 0.9].map((intensity, i) => (
                <div
                  key={i}
                  style={{ backgroundColor: getHeatmapColor(intensity, 1) }}
                  className="flex-1"
                />
              ))}
            </div>
            <span className="text-xs text-gray-600">High</span>
          </div>
        )}
      </div>
    </div>
  );
}
