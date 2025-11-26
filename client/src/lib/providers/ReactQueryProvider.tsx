'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactNode, useState } from 'react';

interface ReactQueryProviderProps {
  children: ReactNode;
}

/**
 * React Query provider component
 * Wraps the app to provide data fetching and caching functionality
 */
export function ReactQueryProvider({ children }: ReactQueryProviderProps) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            // Analytics data rarely changes once processed
            staleTime: 5 * 60 * 1000, // 5 minutes
            // Retry failed requests
            retry: 2,
            // Refetch on window focus for real-time updates
            refetchOnWindowFocus: false,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}
