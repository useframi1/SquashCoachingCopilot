'use client';

import { useParams } from 'next/navigation';
import { Trophy, Target, Activity, Clock, Zap, Timer, Flag } from 'lucide-react';
import { KPICard } from '@/components/dashboard/KPICard';
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
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils/cn';

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
      {/* KPI Cards Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        <KPICard
          title="Match Winner"
          value={match.winner ? getPlayerName(match.winner as 1 | 2) : 'In Progress'}
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
        <h3 className="text-lg font-semibold mb-4">Game Scores</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
          {games.map((game) => (
            <Card key={game.game_number}>
              <CardContent className="pt-6">
                <div className="text-center space-y-3">
                  <Badge variant="secondary">Game {game.game_number}</Badge>

                  <div className="flex items-center justify-center gap-4">
                    <div
                      className={cn(
                        'text-2xl font-bold',
                        game.winner === 1 ? 'text-primary' : 'text-foreground'
                      )}
                    >
                      {game.player_1_score}
                    </div>
                    <div className="text-muted-foreground">-</div>
                    <div
                      className={cn(
                        'text-2xl font-bold',
                        game.winner === 2 ? 'text-primary' : 'text-foreground'
                      )}
                    >
                      {game.player_2_score}
                    </div>
                  </div>

                  {game.winner && (
                    <p className="text-xs text-muted-foreground">
                      {getPlayerName(game.winner as 1 | 2)} wins
                    </p>
                  )}

                  {game.start_time !== null && game.end_time !== null && (
                    <p className="text-xs text-muted-foreground">
                      {formatDuration(game.end_time - game.start_time)}
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>
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
          />

          <KPICard
            title="Average Break"
            value={breakTime?.data ? formatDuration(breakTime.data.avg_break_time) : '-'}
            icon={Timer}
            subtitle={breakTime?.data ? `${breakTime.data.total_breaks} breaks` : undefined}
          />

          <KPICard
            title="Fastest Shot"
            value={fastestShot?.data ? `${fastestShot.data.ball_speed.toFixed(1)} m/s` : '-'}
            icon={Zap}
            subtitle={fastestShot?.data ? getPlayerName(fastestShot.data.player_id as 1 | 2) : undefined}
          />

          <KPICard
            title="Number of Lets"
            value={letStats?.data ? letStats.data.total_lets : '-'}
            icon={Flag}
            subtitle={letStats?.data ? `${letStats.data.let_percentage.toFixed(1)}% of rallies` : undefined}
          />
        </div>
      </div>
    </div>
  );
}
