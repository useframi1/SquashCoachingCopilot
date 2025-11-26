'use client';

import { useParams } from 'next/navigation';
import { Trophy, Target, Activity, Clock } from 'lucide-react';
import { KPICard } from '@/components/dashboard/KPICard';
import { MomentumChart } from '@/components/charts/MomentumChart';
import { EmptyState } from '@/components/error/EmptyState';
import { useMatchSummary, useMomentumTimeline } from '@/lib/hooks/useAnalytics';
import { formatDuration } from '@/lib/utils/formatters';

/**
 * Overview tab - High-level match summary + momentum visualization
 */
export default function OverviewPage() {
  const params = useParams();
  const videoId = params.videoId as string;

  const { data: matchSummary, isLoading: summaryLoading } = useMatchSummary(videoId);
  const { data: momentum, isLoading: momentumLoading } = useMomentumTimeline(videoId);

  if (summaryLoading) {
    return (
      <div className="p-8 space-y-6">
        {/* Loading skeletons */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-32 bg-gray-200 rounded-lg animate-pulse" />
          ))}
        </div>
        <div className="h-96 bg-gray-200 rounded-lg animate-pulse" />
      </div>
    );
  }

  if (!matchSummary) {
    return (
      <div className="p-8">
        <EmptyState
          icon={Trophy}
          title="No Match Data Available"
          description="We couldn't load the match summary. This might be because the analysis is still processing or there was an error."
          className="min-h-96"
        />
      </div>
    );
  }

  const { match, games } = matchSummary;

  // Calculate match duration from game times
  const matchDuration = games.reduce((total, game) => {
    if (game.start_time !== null && game.end_time !== null) {
      return total + (game.end_time - game.start_time);
    }
    return total;
  }, 0);

  return (
    <div className="p-8 space-y-8">
      {/* KPI Cards Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        <KPICard
          title="Match Winner"
          value={match.winner ? `Player ${match.winner}` : 'In Progress'}
          icon={Trophy}
          subtitle={
            match.winner
              ? `${match.player_1_games_won}-${match.player_2_games_won}`
              : undefined
          }
        />

        <KPICard
          title="Final Score"
          value={`${match.player_1_games_won}-${match.player_2_games_won}`}
          icon={Target}
          subtitle={`Best of ${match.best_of}`}
        />

        <KPICard
          title="Total Rallies"
          value={match.total_rallies}
          icon={Activity}
          subtitle={`${match.total_games} games played`}
        />

        <KPICard
          title="Match Duration"
          value={formatDuration(matchDuration)}
          icon={Clock}
          subtitle={match.scoring_system}
        />
      </div>

      {/* Game Scores Grid */}
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Game Scores</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
          {games.map((game) => (
            <div
              key={game.game_number}
              className="bg-white p-6 rounded-lg border border-gray-200"
            >
              <div className="text-center space-y-3">
                <p className="text-sm font-medium text-gray-600">
                  Game {game.game_number}
                </p>

                <div className="flex items-center justify-center gap-4">
                  <div
                    className={`text-2xl font-bold ${
                      game.winner === 1 ? 'text-red-700' : 'text-gray-900'
                    }`}
                  >
                    {game.player_1_score}
                  </div>
                  <div className="text-gray-400">-</div>
                  <div
                    className={`text-2xl font-bold ${
                      game.winner === 2 ? 'text-red-700' : 'text-gray-900'
                    }`}
                  >
                    {game.player_2_score}
                  </div>
                </div>

                {game.winner && (
                  <p className="text-xs text-gray-500">
                    Player {game.winner} wins
                  </p>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Momentum Chart */}
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Match Momentum</h3>
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          {momentumLoading ? (
            <div className="h-96 flex items-center justify-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-red-700" />
            </div>
          ) : momentum?.data && momentum.data.length > 0 ? (
            <MomentumChart data={momentum.data} />
          ) : (
            <EmptyState
              icon={Activity}
              title="No Momentum Data"
              description="Momentum data is not available for this match."
              className="h-96"
            />
          )}
        </div>
      </div>
    </div>
  );
}
