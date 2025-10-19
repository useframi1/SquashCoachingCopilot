'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { ArrowLeft } from 'lucide-react';
import { api } from '@/lib/api';
import { transformToDashboardData } from '@/lib/utils';
import { Button } from '@/components/ui/Button';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs';
import { LoadingScreen } from '@/components/ui/LoadingSpinner';
import { OverviewTab } from '@/components/dashboard/OverviewTab';
import { RallyByRallyTab } from '@/components/dashboard/RallyByRallyTab';
import { VideoPlayerTab } from '@/components/dashboard/VideoPlayerTab';
import type { DashboardData } from '@/lib/types';

export default function DashboardPage() {
  const params = useParams();
  const router = useRouter();
  const jobId = params.jobId as string;

  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    async function fetchResults() {
      try {
        const results = await api.getResults(jobId);
        const dashboardData = transformToDashboardData(
          jobId,
          results.video_filename,
          results.results
        );
        setData(dashboardData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load results');
      } finally {
        setLoading(false);
      }
    }

    fetchResults();
  }, [jobId]);

  if (loading) {
    return <LoadingScreen message="Loading analysis results..." />;
  }

  if (error || !data) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-4 px-6">
        <p className="text-xl text-red-400">{error || 'No data available'}</p>
        <Button onClick={() => router.push('/')} variant="outline">
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Upload
        </Button>
      </div>
    );
  }

  return (
    <main className="min-h-screen bg-[var(--background)] px-6 py-8">
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold text-white">
              Analysis Dashboard
            </h1>
            <p className="mt-2 text-[var(--foreground-secondary)]">
              {data.filename}
            </p>
          </div>
          <Button onClick={() => router.push('/')} variant="outline">
            <ArrowLeft className="mr-2 h-4 w-4" />
            New Analysis
          </Button>
        </div>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="rallies">Rally-by-Rally</TabsTrigger>
            <TabsTrigger value="video">Video Player</TabsTrigger>
          </TabsList>

          <TabsContent value="overview">
            <OverviewTab data={data} />
          </TabsContent>

          <TabsContent value="rallies">
            <RallyByRallyTab
              data={data}
              onPlayRally={() => {
                setActiveTab('video');
                // Note: VideoPlayerTab will need to handle rally selection
              }}
            />
          </TabsContent>

          <TabsContent value="video">
            <VideoPlayerTab data={data} />
          </TabsContent>
        </Tabs>
      </div>
    </main>
  );
}
