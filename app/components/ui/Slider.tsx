'use client';

import * as React from 'react';
import { cn } from '@/lib/utils';

export interface SliderProps {
  min: number;
  max: number;
  step?: number;
  value: [number, number];
  onValueChange: (value: [number, number]) => void;
  className?: string;
}

export function Slider({
  min,
  max,
  step = 1,
  value,
  onValueChange,
  className,
}: SliderProps) {
  const [localValue, setLocalValue] = React.useState(value);

  React.useEffect(() => {
    setLocalValue(value);
  }, [value]);

  const handleChange = (index: 0 | 1, newValue: number) => {
    const updatedValue: [number, number] = [...localValue] as [number, number];
    updatedValue[index] = newValue;

    // Ensure min is always less than max
    if (index === 0 && newValue > localValue[1]) {
      updatedValue[1] = newValue;
    } else if (index === 1 && newValue < localValue[0]) {
      updatedValue[0] = newValue;
    }

    setLocalValue(updatedValue);
    onValueChange(updatedValue);
  };

  const percentage = (val: number) => ((val - min) / (max - min)) * 100;

  return (
    <div className={cn('relative w-full', className)}>
      <div className="relative h-2 w-full rounded-full bg-[var(--border)]">
        {/* Active range */}
        <div
          className="absolute h-2 rounded-full bg-[var(--primary)]"
          style={{
            left: `${percentage(localValue[0])}%`,
            width: `${percentage(localValue[1]) - percentage(localValue[0])}%`,
          }}
        />

        {/* Min handle */}
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={localValue[0]}
          onChange={(e) => handleChange(0, Number(e.target.value))}
          className="slider-thumb"
          style={{
            position: 'absolute',
            width: '100%',
            pointerEvents: 'all',
          }}
        />

        {/* Max handle */}
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={localValue[1]}
          onChange={(e) => handleChange(1, Number(e.target.value))}
          className="slider-thumb"
          style={{
            position: 'absolute',
            width: '100%',
            pointerEvents: 'all',
          }}
        />
      </div>

      {/* Value labels */}
      <div className="mt-2 flex justify-between text-sm text-[var(--foreground-secondary)]">
        <span>{localValue[0].toFixed(1)}s</span>
        <span>{localValue[1].toFixed(1)}s</span>
      </div>

      <style jsx>{`
        .slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          background: transparent;
          cursor: pointer;
          margin: 0;
          padding: 0;
          height: 0.5rem;
        }

        .slider-thumb::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 1rem;
          height: 1rem;
          border-radius: 50%;
          background: var(--primary);
          cursor: pointer;
          border: 2px solid white;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }

        .slider-thumb::-moz-range-thumb {
          width: 1rem;
          height: 1rem;
          border-radius: 50%;
          background: var(--primary);
          cursor: pointer;
          border: 2px solid white;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }

        .slider-thumb:focus {
          outline: none;
        }

        .slider-thumb:focus::-webkit-slider-thumb {
          box-shadow: 0 0 0 3px rgba(139, 21, 56, 0.3);
        }
      `}</style>
    </div>
  );
}
