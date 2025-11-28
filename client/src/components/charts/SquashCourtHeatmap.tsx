'use client';

import { useState } from 'react';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import type { HeatmapGrid } from '@/types/api';

interface QuadrantData {
  label: string;
  count: number;
  percentage: number;
}

interface SquashCourtHeatmapProps {
  heatmapData?: HeatmapGrid;
  quadrantData?: QuadrantData[];
  quadrantBoundaries?: {
    x_cut: number;
    y_cut: number;
  };
  title?: string;
}

/**
 * Squash court visualization with toggle between heatmap and quadrants
 * Displays a top-down view of the squash court with player position data
 */
export function SquashCourtHeatmap({
  heatmapData,
  quadrantData,
  quadrantBoundaries,
  title = 'Player Position Heatmap',
}: SquashCourtHeatmapProps) {
  const [viewMode, setViewMode] = useState<'heatmap' | 'quadrants'>('heatmap');

  // Squash court dimensions (in meters, standard squash court)
  const courtLength = 9.75; // meters (front to back)
  const courtWidth = 6.4;   // meters (side to side)

  // Court markings (from front wall)
  const serviceLine = 5.49;
  const halfCourtLine = courtLength / 2;
  const shortLine = 4.26;
  const serviceBoxWidth = 1.6;

  // Canvas dimensions - portrait orientation (length is vertical)
  const canvasWidth = 400;
  const canvasHeight = 610;

  // Scale factors
  const scaleX = canvasWidth / courtWidth;
  const scaleY = canvasHeight / courtLength;

  // Downsample heatmap grid to reduce cell count
  const downsampleHeatmap = (grid: number[][], targetSize: number = 25) => {
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

    // Map to red color scale from theme
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

  // Get quadrant position based on label
  const getQuadrantInfo = (label: string) => {
    const isFront = label.toLowerCase().includes('front');
    const isLeft = label.toLowerCase().includes('left');

    return {
      isFront,
      isLeft,
      label,
    };
  };

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-md font-semibold text-gray-900">{title}</h4>
        {quadrantData && (
          <Tabs value={viewMode} onValueChange={(v) => setViewMode(v as 'heatmap' | 'quadrants')}>
            <TabsList>
              <TabsTrigger value="heatmap">Heatmap</TabsTrigger>
              <TabsTrigger value="quadrants">Quadrants</TabsTrigger>
            </TabsList>
          </Tabs>
        )}
      </div>

      <div className="relative bg-white border border-gray-200 rounded-lg p-4">
        <svg
          width={canvasWidth}
          height={canvasHeight}
          viewBox={`0 0 ${canvasWidth} ${canvasHeight}`}
          className="mx-auto"
        >
          {/* Court background */}
          <rect
            x={0}
            y={0}
            width={canvasWidth}
            height={canvasHeight}
            fill="#f9fafb"
            stroke="#1f2937"
            strokeWidth={3}
          />

          {/* Render heatmap if in heatmap mode */}
          {viewMode === 'heatmap' && heatmapData && (() => {
            // Downsample to ~25x25 grid for cleaner visualization
            const { grid: downsampledGrid, width: dsWidth, height: dsHeight } =
              downsampleHeatmap(heatmapData.grid, 25);

            // Find max intensity for normalization
            const maxIntensity = Math.max(
              ...downsampledGrid.flat().filter(v => v > 0),
              1 // Prevent division by zero
            );

            return (
              <>
                {downsampledGrid.map((row, rowIndex) =>
                  row.map((intensity, colIndex) => {
                    const cellWidth = canvasWidth / dsWidth;
                    const cellHeight = canvasHeight / dsHeight;
                    const x = colIndex * cellWidth;
                    const y = rowIndex * cellHeight;

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
          {viewMode === 'quadrants' && quadrantData && (
            <>
              {quadrantData.map((quad, index) => {
                const info = getQuadrantInfo(quad.label);
                const x = info.isLeft ? 0 : canvasWidth / 2;
                const y = info.isFront ? 0 : canvasHeight / 2;
                const w = canvasWidth / 2;
                const h = canvasHeight / 2;

                // Color based on percentage
                const opacity = 0.2 + (quad.percentage / 100) * 0.5;
                const color = `rgba(185, 28, 28, ${opacity})`;

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
                      ({quad.count} hits)
                    </text>
                  </g>
                );
              })}
            </>
          )}

          {/* Court lines */}

          {/* Short line (dotted) */}
          <line
            x1={0}
            y1={shortLine * scaleY}
            x2={canvasWidth}
            y2={shortLine * scaleY}
            stroke="#6b7280"
            strokeWidth={2}
            strokeDasharray="5,5"
          />

          {/* Half court line (center vertical line) */}
          <line
            x1={canvasWidth / 2}
            y1={0}
            x2={canvasWidth / 2}
            y2={canvasHeight}
            stroke="#6b7280"
            strokeWidth={2}
            strokeDasharray="5,5"
          />

          {/* Service line */}
          <line
            x1={0}
            y1={serviceLine * scaleY}
            x2={canvasWidth}
            y2={serviceLine * scaleY}
            stroke="#6b7280"
            strokeWidth={2}
            strokeDasharray="5,5"
          />

          {/* T-Zone (center circle at intersection) */}
          <circle
            cx={canvasWidth / 2}
            cy={halfCourtLine * scaleY}
            r={25}
            fill="none"
            stroke="#3b82f6"
            strokeWidth={2}
            strokeDasharray="3,3"
          />

          {/* Service boxes */}
          <rect
            x={(courtWidth / 2 - serviceBoxWidth) * scaleX}
            y={shortLine * scaleY}
            width={serviceBoxWidth * scaleX}
            height={(serviceLine - shortLine) * scaleY}
            fill="none"
            stroke="#9ca3af"
            strokeWidth={1.5}
          />
          <rect
            x={(courtWidth / 2) * scaleX}
            y={shortLine * scaleY}
            width={serviceBoxWidth * scaleX}
            height={(serviceLine - shortLine) * scaleY}
            fill="none"
            stroke="#9ca3af"
            strokeWidth={1.5}
          />

          {/* Labels */}
          <text
            x={canvasWidth / 2}
            y={20}
            textAnchor="middle"
            className="text-xs font-semibold"
            fill="#1f2937"
          >
            FRONT WALL
          </text>

          <text
            x={canvasWidth / 2}
            y={canvasHeight - 10}
            textAnchor="middle"
            className="text-xs font-semibold"
            fill="#1f2937"
          >
            BACK WALL
          </text>

          <text
            x={10}
            y={shortLine * scaleY - 5}
            className="text-xs"
            fill="#6b7280"
          >
            Short Line
          </text>

          <text
            x={10}
            y={serviceLine * scaleY - 5}
            className="text-xs"
            fill="#6b7280"
          >
            Service Line
          </text>

          <text
            x={canvasWidth / 2 + 30}
            y={halfCourtLine * scaleY + 5}
            className="text-xs font-medium"
            fill="#3b82f6"
          >
            T
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

        {/* Court dimensions */}
        <div className="mt-2 text-center text-xs text-gray-500">
          {courtWidth}m × {courtLength}m (Top-down view)
        </div>
      </div>
    </div>
  );
}
