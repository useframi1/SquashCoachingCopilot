'use client';

import { useEffect, useRef } from 'react';
import type { HeatmapGrid } from '@/types/api';
import { normalizeHeatmapGrid, getHeatmapColor } from '@/lib/utils/chart-utils';

interface HeatmapChartProps {
  grid: HeatmapGrid;
  title?: string;
  showCourtLines?: boolean;
}

/**
 * Canvas-based heatmap for player positions or wall hits
 * Renders a 32x50 grid with color intensity based on density
 */
export function HeatmapChart({ grid, title, showCourtLines = true }: HeatmapChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Normalize grid values
    const normalizedGrid = normalizeHeatmapGrid(grid.grid);

    // Calculate cell dimensions
    const cellWidth = canvas.width / grid.width;
    const cellHeight = canvas.height / grid.height;

    // Draw heatmap
    normalizedGrid.forEach((row, y) => {
      row.forEach((value, x) => {
        const color = getHeatmapColor(value);
        ctx.fillStyle = color;
        ctx.fillRect(x * cellWidth, y * cellHeight, cellWidth, cellHeight);
      });
    });

    // Draw court lines if enabled
    if (showCourtLines) {
      ctx.strokeStyle = '#9ca3af';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);

      // Court outline
      ctx.strokeRect(0, 0, canvas.width, canvas.height);

      // T-zone (center of court)
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const tZoneRadius = 20;

      ctx.beginPath();
      ctx.arc(centerX, centerY, tZoneRadius, 0, 2 * Math.PI);
      ctx.stroke();

      // Service boxes (approximate)
      ctx.beginPath();
      ctx.moveTo(canvas.width / 2, 0);
      ctx.lineTo(canvas.width / 2, canvas.height);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(0, canvas.height * 0.5);
      ctx.lineTo(canvas.width, canvas.height * 0.5);
      ctx.stroke();

      ctx.setLineDash([]);
    }
  }, [grid, showCourtLines]);

  return (
    <div className="space-y-4">
      {title && <h4 className="text-md font-semibold text-gray-900">{title}</h4>}
      <div className="relative bg-white border border-gray-200 rounded-lg p-4">
        <canvas
          ref={canvasRef}
          width={320}
          height={500}
          className="w-full h-auto"
          style={{ maxHeight: '500px' }}
        />

        {/* Legend */}
        <div className="mt-4 flex items-center justify-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-white border border-gray-300 rounded" />
            <span className="text-xs text-gray-600">Low</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: '#fca5a5' }} />
            <span className="text-xs text-gray-600">Medium</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: '#991b1b' }} />
            <span className="text-xs text-gray-600">High</span>
          </div>
        </div>

        {/* Dimensions label */}
        <div className="mt-2 text-center text-xs text-gray-500">
          {grid.bounds.x_max}m × {grid.bounds.y_max}m
        </div>
      </div>
    </div>
  );
}
