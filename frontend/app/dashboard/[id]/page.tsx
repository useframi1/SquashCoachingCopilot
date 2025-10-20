'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { ArrowLeft, Activity, Video, Clock, TrendingUp } from 'lucide-react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { getAnalysisResults, getRallyVideoUrl } from '@/lib/api';
import { ResultsResponse, Rally } from '@/lib/types';

export default function Dashboard() {
  const params = useParams();
  const router = useRouter();
  const jobId = params.id as string;
  const [results, setResults] = useState<ResultsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedRally, setSelectedRally] = useState<number | null>(null);

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

  // Prepare data for charts
  const rallyDurationData = analysisData.rallies.map((rally, idx) => ({
    name: `Rally ${idx + 1}`,
    duration: ((rally.end_frame - rally.start_frame) / 30).toFixed(1), // Assuming 30 FPS
    frames: rally.end_frame - rally.start_frame,
  }));

  // Count stroke types across all rallies
  const strokeCounts: Record<string, number> = {};
  analysisData.rallies.forEach((rally) => {
    rally.rally_frames.forEach((frame) => {
      if (frame.player1.stroke_type) {
        strokeCounts[frame.player1.stroke_type] = (strokeCounts[frame.player1.stroke_type] || 0) + 1;
      }
      if (frame.player2.stroke_type) {
        strokeCounts[frame.player2.stroke_type] = (strokeCounts[frame.player2.stroke_type] || 0) + 1;
      }
    });
  });

  const strokeData = Object.entries(strokeCounts).map(([type, count]) => ({
    type,
    count,
  }));

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

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <StatCard
            title="Total Rallies"
            value={analysisData.total_rallies}
            icon={<Activity className="w-6 h-6" />}
            primary
          />
          <StatCard
            title="Avg Rally Duration"
            value={analysisData.avg_rally_duration.toFixed(1)}
            unit="sec"
            icon={<Clock className="w-6 h-6" />}
          />
          <StatCard
            title="Total Strokes"
            value={Object.values(strokeCounts).reduce((a, b) => a + b, 0)}
            icon={<TrendingUp className="w-6 h-6" />}
          />
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Rally Duration Chart */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-black mb-6">
              Rally Durations
            </h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={rallyDurationData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#00000010" />
                <XAxis dataKey="name" stroke="#000000" tick={{ fill: '#000000' }} />
                <YAxis stroke="#000000" tick={{ fill: '#000000' }} label={{ value: 'Seconds', angle: -90, position: 'insideLeft' }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#FFFFFF',
                    border: '1px solid #00000020',
                    borderRadius: '8px',
                  }}
                />
                <Bar dataKey="duration" fill="#8B0000" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Stroke Type Distribution */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-black mb-6">
              Stroke Type Distribution
            </h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={strokeData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#00000010" />
                <XAxis dataKey="type" stroke="#000000" tick={{ fill: '#000000' }} />
                <YAxis stroke="#000000" tick={{ fill: '#000000' }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#FFFFFF',
                    border: '1px solid #00000020',
                    borderRadius: '8px',
                  }}
                />
                <Bar dataKey="count" fill="#000000" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Rally List */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-2xl font-bold text-black mb-6 flex items-center gap-2">
            <Video className="w-6 h-6 text-dark-red" />
            Rally Breakdown
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {analysisData.rallies.map((rally, index) => (
              <RallyCard
                key={index}
                rally={rally}
                rallyNumber={index + 1}
                jobId={jobId}
                isSelected={selectedRally === index + 1}
                onClick={() => setSelectedRally(index + 1)}
              />
            ))}
          </div>
        </div>

        {/* Rally Video Player */}
        {selectedRally !== null && (
          <div className="mt-8 bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-black mb-6">
              Rally {selectedRally} Video
            </h2>
            <div className="aspect-video bg-black rounded-lg overflow-hidden">
              <video
                key={selectedRally}
                controls
                className="w-full h-full"
                src={getRallyVideoUrl(jobId, selectedRally)}
              >
                Your browser does not support the video tag.
              </video>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

interface StatCardProps {
  title: string;
  value: number | string;
  unit?: string;
  icon?: React.ReactNode;
  primary?: boolean;
}

function StatCard({ title, value, unit = '', icon, primary = false }: StatCardProps) {
  return (
    <div
      className={`rounded-xl shadow-lg p-6 ${
        primary ? 'bg-dark-red text-white' : 'bg-white text-black'
      }`}
    >
      <div className="flex items-start justify-between mb-2">
        <h3
          className={`text-sm font-medium ${
            primary ? 'text-white/80' : 'text-black/60'
          }`}
        >
          {title}
        </h3>
        {icon && (
          <div className={primary ? 'text-white' : 'text-dark-red'}>
            {icon}
          </div>
        )}
      </div>
      <div className="flex items-baseline gap-1">
        <span className="text-5xl font-bold">{value}</span>
        {unit && (
          <span className={`text-xl ${primary ? 'text-white/60' : 'text-black/40'}`}>
            {unit}
          </span>
        )}
      </div>
    </div>
  );
}

interface RallyCardProps {
  rally: Rally;
  rallyNumber: number;
  jobId: string;
  isSelected: boolean;
  onClick: () => void;
}

function RallyCard({ rally, rallyNumber, isSelected, onClick }: RallyCardProps) {
  const duration = ((rally.end_frame - rally.start_frame) / 30).toFixed(1);
  const totalFrames = rally.rally_frames.length;

  // Count strokes in this rally
  const strokes = rally.rally_frames.filter(
    (f) => f.player1.stroke_type || f.player2.stroke_type
  ).length;

  return (
    <button
      onClick={onClick}
      className={`p-4 rounded-lg border-2 transition-all text-left ${
        isSelected
          ? 'border-dark-red bg-dark-red/5'
          : 'border-black/10 hover:border-dark-red/50 bg-white'
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-lg font-bold text-black">Rally {rallyNumber}</h3>
        <Video className={`w-5 h-5 ${isSelected ? 'text-dark-red' : 'text-black/40'}`} />
      </div>
      <div className="space-y-1 text-sm text-black/60">
        <p>Duration: {duration}s</p>
        <p>Frames: {totalFrames}</p>
        <p>Strokes: {strokes}</p>
      </div>
    </button>
  );
}
