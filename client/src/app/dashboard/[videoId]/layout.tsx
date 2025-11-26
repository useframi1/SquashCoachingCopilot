'use client';

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import { Sidebar } from '@/components/dashboard/Sidebar';
import { FilterBar } from '@/components/dashboard/FilterBar';

interface DashboardLayoutProps {
  children: React.ReactNode;
}

/**
 * Dashboard layout with sidebar and filter bar
 * Wraps all dashboard tab pages
 */
export default function DashboardLayout({ children }: DashboardLayoutProps) {
  const params = useParams();
  const videoId = params.videoId as string;
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

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
        {/* Filter Bar */}
        <FilterBar
          videoId={videoId}
          onMenuClick={() => setMobileMenuOpen(true)}
        />

        {/* Tab Content */}
        <main className="flex-1 overflow-y-auto">
          {children}
        </main>
      </div>
    </div>
  );
}
