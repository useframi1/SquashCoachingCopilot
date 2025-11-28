'use client';

import { useState, useEffect } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { X } from 'lucide-react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { updatePlayerNames, getFirstFrameUrl } from '@/lib/api/videos';
import type { PlayerNamesUpdate } from '@/types/api';

interface PlayerNamingModalProps {
  videoId: string;
  isOpen: boolean;
  onClose: () => void;
  currentPlayer1Name?: string | null;
  currentPlayer2Name?: string | null;
}

/**
 * Modal for naming players in a match
 * Shows first annotated frame and allows users to assign names to Player 1 and Player 2
 */
export function PlayerNamingModal({
  videoId,
  isOpen,
  onClose,
  currentPlayer1Name,
  currentPlayer2Name,
}: PlayerNamingModalProps) {
  const queryClient = useQueryClient();
  const [player1Name, setPlayer1Name] = useState('');
  const [player2Name, setPlayer2Name] = useState('');
  const firstFrameUrl = getFirstFrameUrl(videoId);

  // Initialize names from props
  useEffect(() => {
    setPlayer1Name(currentPlayer1Name || '');
    setPlayer2Name(currentPlayer2Name || '');
  }, [currentPlayer1Name, currentPlayer2Name, isOpen]);

  const { mutate: saveNames, isPending } = useMutation({
    mutationFn: (names: PlayerNamesUpdate) => updatePlayerNames(videoId, names),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['video', videoId] });
      onClose();
    },
    onError: (error) => {
      console.error('Failed to update player names:', error);
    },
  });

  const handleSave = () => {
    saveNames({
      player_1_name: player1Name.trim() || null,
      player_2_name: player2Name.trim() || null,
    });
  };

  const handleSkip = () => {
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-7xl max-h-[95vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-xl font-semibold">
            Name the Players
          </DialogTitle>
        </DialogHeader>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* First Frame Preview */}
          <div className="lg:col-span-2 space-y-3">
            <h3 className="text-sm font-medium text-muted-foreground">
              First Frame (Player positions marked)
            </h3>
            <div className="border border-border rounded-lg overflow-hidden bg-muted">
              <img
                src={firstFrameUrl}
                alt="First frame with player annotations"
                className="w-full h-auto"
                onError={(e) => {
                  e.currentTarget.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><text x="50%" y="50%" text-anchor="middle" dy=".3em">No preview</text></svg>';
                }}
              />
            </div>
            <p className="text-xs text-muted-foreground">
              Use the player markers (P1, P2) in the image above to identify each player
            </p>
          </div>

          {/* Player Names Form */}
          <div className="space-y-6">
            <div className="space-y-4">
              <div className="space-y-2">
                <label
                  htmlFor="player1"
                  className="text-sm font-medium text-foreground"
                >
                  Player 1 Name
                </label>
                <input
                  id="player1"
                  type="text"
                  value={player1Name}
                  onChange={(e) => setPlayer1Name(e.target.value)}
                  placeholder="Enter Player 1 name"
                  className="w-full px-4 py-2 border border-input rounded-lg focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent bg-background text-foreground"
                  disabled={isPending}
                />
              </div>

              <div className="space-y-2">
                <label
                  htmlFor="player2"
                  className="text-sm font-medium text-foreground"
                >
                  Player 2 Name
                </label>
                <input
                  id="player2"
                  type="text"
                  value={player2Name}
                  onChange={(e) => setPlayer2Name(e.target.value)}
                  placeholder="Enter Player 2 name"
                  className="w-full px-4 py-2 border border-input rounded-lg focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent bg-background text-foreground"
                  disabled={isPending}
                />
              </div>
            </div>

            <div className="flex gap-3 pt-4">
              <Button
                onClick={handleSave}
                disabled={isPending}
                className="flex-1"
              >
                {isPending ? 'Saving...' : 'Save Names'}
              </Button>
              <Button
                onClick={handleSkip}
                variant="outline"
                disabled={isPending}
                className="flex-1"
              >
                Skip
              </Button>
            </div>

            <p className="text-xs text-muted-foreground text-center">
              You can always edit player names later from the dashboard
            </p>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
