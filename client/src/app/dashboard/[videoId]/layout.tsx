'use client';

import { useState, useEffect } from 'react';
import { useParams, usePathname, useSearchParams, useRouter } from 'next/navigation';
import { useQuery } from '@tanstack/react-query';
import { Sidebar } from '@/components/dashboard/Sidebar';
import { FilterBar } from '@/components/dashboard/FilterBar';
import { PlayerNamingModal } from '@/components/dashboard/PlayerNamingModal';
import { getVideoMetadata } from '@/lib/api/videos';

interface DashboardLayoutProps {
  children: React.ReactNode;
}

/**
 * Dashboard layout with sidebar and filter bar
 * Wraps all dashboard tab pages
 */
export default function DashboardLayout({ children }: DashboardLayoutProps) {
  const params = useParams();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const router = useRouter();
  const videoId = params.videoId as string;
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [isNamingModalOpen, setIsNamingModalOpen] = useState(false);

  // Fetch video metadata for player names
  const { data: videoMetadata } = useQuery({
    queryKey: ['video', videoId],
    queryFn: () => getVideoMetadata(videoId),
    staleTime: 5 * 60 * 1000,
  });

  // Check if we should auto-show naming modal (after processing)
  useEffect(() => {
    const showNaming = searchParams.get('showNaming');
    if (showNaming === 'true' && videoMetadata) {
      // Only show if player names are not already set
      if (!videoMetadata.player_1_name && !videoMetadata.player_2_name) {
        setIsNamingModalOpen(true);
      }
      // Clean up URL parameter
      const newUrl = pathname;
      router.replace(newUrl);
    }
  }, [searchParams, pathname, router, videoMetadata]);

  // Check if current page is chat
  const isChatPage = pathname.endsWith('/chat');

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <Sidebar
        videoId={videoId}
        mobileOpen={mobileMenuOpen}
        onMobileToggle={() => setMobileMenuOpen(!mobileMenuOpen)}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Filter Bar - Hidden on chat page */}
        {!isChatPage && (
          <FilterBar
            videoId={videoId}
            onMenuClick={() => setMobileMenuOpen(true)}
          />
        )}

        {/* Tab Content */}
        <main className="flex-1 overflow-y-auto">
          {children}
        </main>
      </div>

      {/* Player Naming Modal - Auto-show after processing */}
      <PlayerNamingModal
        videoId={videoId}
        isOpen={isNamingModalOpen}
        onClose={() => setIsNamingModalOpen(false)}
        currentPlayer1Name={videoMetadata?.player_1_name}
        currentPlayer2Name={videoMetadata?.player_2_name}
      />
    </div>
  );
}
