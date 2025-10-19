import * as React from 'react';
import { cn } from '@/lib/utils';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(
          'inline-flex items-center justify-center rounded-lg font-medium transition-all',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2',
          'disabled:pointer-events-none disabled:opacity-50',
          {
            'bg-[var(--primary)] text-white hover:bg-[var(--primary-hover)] focus-visible:ring-[var(--primary)]':
              variant === 'primary',
            'bg-[var(--card-bg)] text-white hover:bg-[var(--border-light)] focus-visible:ring-[var(--border-light)]':
              variant === 'secondary',
            'border-2 border-[var(--border)] bg-transparent text-white hover:bg-[var(--card-bg)] focus-visible:ring-[var(--border)]':
              variant === 'outline',
            'bg-transparent text-white hover:bg-[var(--card-bg)] focus-visible:ring-[var(--border-light)]':
              variant === 'ghost',
          },
          {
            'h-9 px-3 text-sm': size === 'sm',
            'h-11 px-6 text-base': size === 'md',
            'h-14 px-8 text-lg': size === 'lg',
          },
          className
        )}
        {...props}
      />
    );
  }
);

Button.displayName = 'Button';

export { Button };
