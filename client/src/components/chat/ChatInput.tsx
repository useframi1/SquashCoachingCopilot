'use client';

import { useState, KeyboardEvent } from 'react';
import { Send, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

/**
 * Chat input component for sending messages to the LLM
 */
export function ChatInput({
  onSendMessage,
  disabled = false,
  placeholder = 'Ask about match analytics...',
}: ChatInputProps) {
  const [message, setMessage] = useState('');

  const handleSend = () => {
    const trimmedMessage = message.trim();
    if (trimmedMessage && !disabled) {
      onSendMessage(trimmedMessage);
      setMessage('');
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Send on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="border-t border-border bg-background p-4">
      <div className="flex gap-3 items-start max-w-5xl mx-auto">
        <div className="flex-1">
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            placeholder={placeholder}
            rows={2}
            className="w-full px-4 py-3 border border-input rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed bg-background text-foreground placeholder:text-muted-foreground"
          />
          <p className="text-xs text-muted-foreground mt-1.5">
            Press Enter to send, Shift + Enter for new line
          </p>
        </div>

        <Button
          onClick={handleSend}
          disabled={disabled || !message.trim()}
          className="px-6 py-3 shrink-0"
        >
          {disabled ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin mr-2" />
              <span>Thinking...</span>
            </>
          ) : (
            <>
              <Send className="w-4 h-4 mr-2" />
              <span>Send</span>
            </>
          )}
        </Button>
      </div>
    </div>
  );
}
