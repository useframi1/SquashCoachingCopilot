import { useQuery } from '@tanstack/react-query';
import { getVideoMetadata } from '@/lib/api/videos';

/**
 * Hook to get player display names for a video
 * Returns the custom name if set, otherwise falls back to "Player 1" / "Player 2"
 */
export function usePlayerNames(videoId: string) {
  const { data: videoMetadata } = useQuery({
    queryKey: ['video', videoId],
    queryFn: () => getVideoMetadata(videoId),
    staleTime: 5 * 60 * 1000,
  });

  const getPlayerName = (playerId: 1 | 2): string => {
    if (playerId === 1) {
      return videoMetadata?.player_1_name || 'Player 1';
    } else {
      return videoMetadata?.player_2_name || 'Player 2';
    }
  };

  return {
    player1Name: getPlayerName(1),
    player2Name: getPlayerName(2),
    getPlayerName,
    hasCustomNames: !!(videoMetadata?.player_1_name || videoMetadata?.player_2_name),
  };
}
