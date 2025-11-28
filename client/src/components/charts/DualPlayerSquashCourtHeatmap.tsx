'use client';

import { useState } from 'react';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import type { HeatmapGrid } from '@/types/api';

interface QuadrantData {
  label: string;
  count: number;
  percentage: number;
}

interface DualPlayerSquashCourtHeatmapProps {
  player1HeatmapData?: HeatmapGrid;
  player2HeatmapData?: HeatmapGrid;
  player1QuadrantData?: QuadrantData[];
  player2QuadrantData?: QuadrantData[];
  quadrantBoundaries?: {
    x_cut: number;
    y_cut: number;
  };
  player1Name?: string;
  player2Name?: string;
  title?: string;
}

/**
 * Dual squash court visualization showing side-by-side heatmaps for both players
 * Displays a top-down view of the squash court with player position data
 */
export function DualPlayerSquashCourtHeatmap({
  player1HeatmapData,
  player2HeatmapData,
  player1QuadrantData,
  player2QuadrantData,
  quadrantBoundaries,
  player1Name = 'Player 1',
  player2Name = 'Player 2',
  title = 'Player Position Analysis',
}: DualPlayerSquashCourtHeatmapProps) {
  const [viewMode, setViewMode] = useState<'heatmap' | 'quadrants'>('heatmap');

  // Squash court dimensions (in meters, standard squash court)
  const courtLength = 9.75; // meters (front to back)
  const courtWidth = 6.4;   // meters (side to side)

  // Court markings (from front wall)
  const serviceLine = 5.49;
  const halfCourtLine = courtLength / 2;
  const shortLine = 4.26;

  // Canvas dimensions - portrait orientation (length is vertical)
  const canvasWidth = 350;
  const canvasHeight = 550;

  // Scale factors
  const scaleX = canvasWidth / courtWidth;
  const scaleY = canvasHeight / courtLength;

  // Downsample heatmap grid to reduce cell count
  const downsampleHeatmap = (grid: number[][], targetSize: number = 25) => {
    if (!grid || grid.length === 0) return { grid: [], width: 0, height: 0 };

    const originalHeight = grid.length;
    const originalWidth = grid[0].length;

    const rowFactor = Math.ceil(originalHeight / targetSize);
    const colFactor = Math.ceil(originalWidth / targetSize);

    const newHeight = Math.ceil(originalHeight / rowFactor);
    const newWidth = Math.ceil(originalWidth / colFactor);

    const downsampled: number[][] = [];

    for (let i = 0; i < newHeight; i++) {
      const row: number[] = [];
      for (let j = 0; j < newWidth; j++) {
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

  // Get color for heatmap intensity (both players use red scale)
  const getHeatmapColor = (intensity: number, maxIntensity: number) => {
    if (intensity === 0) return 'rgba(0, 0, 0, 0)';

    const normalized = maxIntensity > 0 ? intensity / maxIntensity : 0;

    // Red scale for both players
    if (normalized < 0.2) return '#fee2e2';
    else if (normalized < 0.4) return '#fca5a5';
    else if (normalized < 0.6) return '#f87171';
    else if (normalized < 0.8) return '#ef4444';
    else return '#dc2626';
  };

  // Get quadrant position based on label
  const getQuadrantInfo = (label: string) => {
    const isFront = label.toLowerCase().includes('front');
    const isLeft = label.toLowerCase().includes('left');

    return { isFront, isLeft, label };
  };

  // Render a single court
  const renderCourt = (
    heatmapData: HeatmapGrid | undefined,
    quadrantData: QuadrantData[] | undefined,
    playerName: string
  ) => {
    return (
      <div className="flex flex-col items-center">
        <h5 className="text-sm font-semibold text-gray-900 mb-2">{playerName}</h5>
        <svg
          width={canvasWidth}
          height={canvasHeight}
          viewBox={`0 0 ${canvasWidth} ${canvasHeight}`}
          className="border-2 border-gray-300 rounded"
        >
          {/* Court background */}
          <rect
            x={0}
            y={0}
            width={canvasWidth}
            height={canvasHeight}
            fill="#f9fafb"
          />

          {/* Render heatmap if in heatmap mode */}
          {viewMode === 'heatmap' && heatmapData && (() => {
            const { grid: downsampledGrid, width: dsWidth, height: dsHeight } =
              downsampleHeatmap(heatmapData.grid, 25);

            const maxIntensity = Math.max(
              ...downsampledGrid.flat().filter(v => v > 0),
              1
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

                const opacity = 0.2 + (quad.percentage / 100) * 0.5;
                const color = `rgba(185, 28, 28, ${opacity})`; // Red for both players

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
                      ({quad.count})
                    </text>
                  </g>
                );
              })}
            </>
          )}

          {/* Court lines */}
          <line
            x1={0}
            y1={shortLine * scaleY}
            x2={canvasWidth}
            y2={shortLine * scaleY}
            stroke="#6b7280"
            strokeWidth={1.5}
            strokeDasharray="5,5"
          />

          <line
            x1={canvasWidth / 2}
            y1={0}
            x2={canvasWidth / 2}
            y2={canvasHeight}
            stroke="#6b7280"
            strokeWidth={1.5}
            strokeDasharray="5,5"
          />

          <line
            x1={0}
            y1={serviceLine * scaleY}
            x2={canvasWidth}
            y2={serviceLine * scaleY}
            stroke="#6b7280"
            strokeWidth={1.5}
            strokeDasharray="5,5"
          />

          <circle
            cx={canvasWidth / 2}
            cy={halfCourtLine * scaleY}
            r={20}
            fill="none"
            stroke="#3b82f6"
            strokeWidth={1.5}
            strokeDasharray="3,3"
          />

          {/* Court border */}
          <rect
            x={0}
            y={0}
            width={canvasWidth}
            height={canvasHeight}
            fill="none"
            stroke="#1f2937"
            strokeWidth={2}
          />
        </svg>
      </div>
    );
  };

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-md font-semibold text-gray-900">{title}</h4>
        {player1QuadrantData && player2QuadrantData && (
          <Tabs value={viewMode} onValueChange={(v) => setViewMode(v as 'heatmap' | 'quadrants')}>
            <TabsList>
              <TabsTrigger value="heatmap">Heatmap</TabsTrigger>
              <TabsTrigger value="quadrants">Quadrants</TabsTrigger>
            </TabsList>
          </Tabs>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {renderCourt(player1HeatmapData, player1QuadrantData, player1Name)}
        {renderCourt(player2HeatmapData, player2QuadrantData, player2Name)}
      </div>

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
  );
}
