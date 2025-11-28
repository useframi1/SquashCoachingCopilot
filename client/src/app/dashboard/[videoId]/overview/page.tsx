'use client';

import { useParams } from 'next/navigation';
import { Trophy, Activity, Clock, Zap, Timer, Flag } from 'lucide-react';
import { KPICard } from '@/components/dashboard/KPICard';
import { ScoreCard } from '@/components/dashboard/ScoreCard';
import { MomentumChart } from '@/components/charts/MomentumChart';
import { EmptyState } from '@/components/error/EmptyState';
import {
  useMatchSummary,
  useMomentumTimeline,
  useLongestRally,
  useFastestShot,
  useLetStats,
  useBreakTime,
} from '@/lib/hooks/useAnalytics';
import { usePlayerNames } from '@/lib/hooks/usePlayerNames';
import { formatDuration } from '@/lib/utils/formatters';
import { Card, CardContent } from '@/components/ui/card';

/**
 * Overview tab - High-level match summary + momentum visualization
 */
export default function OverviewPage() {
  const params = useParams();
  const videoId = params.videoId as string;
  const { player1Name, player2Name, getPlayerName } = usePlayerNames(videoId);

  const { data: matchSummary, isLoading: summaryLoading } = useMatchSummary(videoId);
  const { data: momentum, isLoading: momentumLoading } = useMomentumTimeline(videoId);
  const { data: longestRally } = useLongestRally(videoId);
  const { data: fastestShot } = useFastestShot(videoId);
  const { data: letStats } = useLetStats(videoId);
  const { data: breakTime } = useBreakTime(videoId);

  if (summaryLoading) {
    return (
      <div className="p-8 space-y-6">
        {/* Loading skeletons */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-32 bg-muted rounded-lg animate-pulse" />
          ))}
        </div>
        <div className="h-96 bg-muted rounded-lg animate-pulse" />
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
      {/* Final Score - Large Display */}
      <div className="max-w-2xl mx-auto">
        <ScoreCard
          player1Name={player1Name}
          player2Name={player2Name}
          player1Score={match.player_1_games_won}
          player2Score={match.player_2_games_won}
          winner={match.winner as 1 | 2 | null}
          label="Final Score"
          subtitle={`Best of ${match.best_of} • ${match.scoring_system}`}
          variant="large"
        />
      </div>

      {/* KPI Cards Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        <KPICard
          title="Match Winner"
          value={match.winner ? getPlayerName(match.winner as 1 | 2) : 'In Progress'}
          icon={Trophy}
          subtitle={
            match.winner
              ? `${match.player_1_games_won}-${match.player_2_games_won}`
              : undefined
          }
          infoTooltip="The player who won the most games in the match."
        />

        <KPICard
          title="Total Rallies"
          value={match.total_rallies}
          icon={Activity}
          subtitle={`${match.total_games} games played`}
          infoTooltip="The total number of rallies played across all games in the match. Each rally ends when a player wins a point."
        />

        <KPICard
          title="Match Duration"
          value={formatDuration(matchDuration)}
          icon={Clock}
          subtitle={match.scoring_system}
          infoTooltip="The total time elapsed from the start of the first game to the end of the last game, including breaks between games."
        />
      </div>

      {/* Game Scores Grid */}
      <div>
        <h3 className="text-lg font-semibold mb-4">Game Scores</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
          {games.map((game) => (
            <ScoreCard
              key={game.game_number}
              player1Name={player1Name}
              player2Name={player2Name}
              player1Score={game.player_1_score}
              player2Score={game.player_2_score}
              winner={game.winner as 1 | 2 | null}
              label={`Game ${game.game_number}`}
              subtitle={
                game.start_time !== null && game.end_time !== null
                  ? formatDuration(game.end_time - game.start_time)
                  : undefined
              }
            />
          ))}
        </div>
      </div>

      {/* Momentum Chart */}
      <div>
        <h3 className="text-lg font-semibold mb-4">Match Momentum</h3>
        <Card>
          <CardContent className="pt-6">
            {momentumLoading ? (
              <div className="h-96 flex items-center justify-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary" />
              </div>
            ) : momentum?.data && momentum.data.length > 0 ? (
              <MomentumChart data={momentum.data} player1Name={player1Name} player2Name={player2Name} />
            ) : (
              <EmptyState
                icon={Activity}
                title="No Momentum Data"
                description="Momentum data is not available for this match."
                className="h-96"
              />
            )}
          </CardContent>
        </Card>
      </div>

      {/* Match Highlights - 4 Summary Cards */}
      <div>
        <h3 className="text-lg font-semibold mb-4">Match Highlights</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          <KPICard
            title="Longest Rally"
            value={longestRally?.data ? `${longestRally.data.shot_count} shots` : '-'}
            icon={Activity}
            subtitle={longestRally?.data ? formatDuration(longestRally.data.rally_duration) : undefined}
            infoTooltip="The rally with the highest number of shots in the match. Shows total shots and rally duration."
          />

          <KPICard
            title="Average Break"
            value={breakTime?.data ? formatDuration(breakTime.data.avg_break_time) : '-'}
            icon={Timer}
            subtitle={breakTime?.data ? `${breakTime.data.total_breaks} breaks` : undefined}
            infoTooltip="The average time between rallies. Breaks occur when players pause between points to serve or prepare."
          />

          <KPICard
            title="Fastest Shot"
            value={fastestShot?.data ? `${fastestShot.data.ball_speed.toFixed(1)} m/s` : '-'}
            icon={Zap}
            subtitle={fastestShot?.data ? getPlayerName(fastestShot.data.player_id as 1 | 2) : undefined}
            infoTooltip="The highest ball speed recorded during the match, measured in meters per second."
          />

          <KPICard
            title="Number of Lets"
            value={letStats?.data ? letStats.data.total_lets : '-'}
            icon={Flag}
            subtitle={letStats?.data ? `${letStats.data.let_percentage.toFixed(1)}% of rallies` : undefined}
            infoTooltip="The total number of let calls during the match. A let is when a rally is replayed, typically due to interference."
          />
        </div>
      </div>
    </div>
  );
}
