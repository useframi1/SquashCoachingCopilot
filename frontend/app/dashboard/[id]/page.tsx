'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { ArrowLeft } from 'lucide-react';
import { getAnalysisResults } from '@/lib/api';
import { ResultsResponse } from '@/lib/types';
import TabNavigation from '@/components/dashboard/TabNavigation';
import OverviewTab from '@/components/dashboard/OverviewTab';
import PlayerAnalysisTab from '@/components/dashboard/PlayerAnalysisTab';
import RallyByRallyTab from '@/components/dashboard/RallyByRallyTab';

export default function Dashboard() {
  const params = useParams();
  const router = useRouter();
  const jobId = params.id as string;
  const [results, setResults] = useState<ResultsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'player' | 'rally'>('overview');

  useEffect(() => {
    async function fetchResults() {
      try {
        const data = await getAnalysisResults(jobId);
        setResults(data);
      } catch (error) {
        console.error('Failed to fetch results:', error);
        alert('Failed to load analysis results');
      } finally {
        setLoading(false);
      }
    }

    if (jobId) {
      fetchResults();
    }
  }, [jobId]);

  if (loading) {
    return (
      <div className="min-h-screen bg-cream flex items-center justify-center">
        <div className="text-2xl text-black">Loading results...</div>
      </div>
    );
  }

  if (!results || !results.results) {
    return (
      <div className="min-h-screen bg-cream flex items-center justify-center">
        <div className="text-2xl text-black">No results found</div>
      </div>
    );
  }

  const { results: analysisData } = results;

  return (
    <div className="min-h-screen bg-cream p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <button
            onClick={() => router.push('/')}
            className="flex items-center gap-2 text-black/60 hover:text-black transition-colors mb-6"
          >
            <ArrowLeft className="w-5 h-5" />
            <span>Back to Home</span>
          </button>
          <h1 className="text-5xl font-bold text-black mb-2">
            Match Analysis Dashboard
          </h1>
          <p className="text-xl text-black/70">
            {results.video_filename}
          </p>
        </div>

        {/* Tab Navigation */}
        <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />

        {/* Tab Content */}
        {activeTab === 'overview' && <OverviewTab results={analysisData} />}
        {activeTab === 'player' && <PlayerAnalysisTab results={analysisData} />}
        {activeTab === 'rally' && <RallyByRallyTab results={analysisData} />}
      </div>
    </div>
  );
}
