'use client';

import { User, Bot } from 'lucide-react';
import type { LLMMessage } from '@/types/api';
import { formatDistance } from 'date-fns';

interface ChatMessageProps {
  message: LLMMessage;
}

/**
 * Chat message component for displaying LLM conversation messages
 * Displays user messages, assistant responses, and function calls
 */
export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';
  const isAssistant = message.role === 'assistant';

  // Don't render system or tool messages
  if (message.role === 'system' || message.role === 'tool') {
    return null;
  }

  const timeAgo = message.timestamp
    ? formatDistance(new Date(message.timestamp), new Date(), { addSuffix: true })
    : '';

  return (
    <div className={`flex gap-3 mb-4 ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`flex gap-3 max-w-[85%] ${
          isUser ? 'flex-row-reverse' : 'flex-row'
        }`}
      >
        {/* Avatar */}
        <div
          className={`flex-shrink-0 w-9 h-9 rounded-full flex items-center justify-center shadow-sm ${
            isUser
              ? 'bg-primary text-primary-foreground'
              : 'bg-muted text-foreground border border-border'
          }`}
        >
          {isUser ? (
            <User className="w-5 h-5" />
          ) : (
            <Bot className="w-5 h-5" />
          )}
        </div>

        {/* Message Content */}
        <div className="flex flex-col gap-1.5 flex-1 min-w-0">
          <div
            className={`rounded-lg px-4 py-3 shadow-sm ${
              isUser
                ? 'bg-primary text-primary-foreground'
                : 'bg-card text-card-foreground border border-border'
            }`}
          >
            {/* Message text */}
            <div className="text-sm leading-relaxed whitespace-pre-wrap break-words">
              {message.content}
            </div>
          </div>


          {/* Timestamp */}
          {timeAgo && (
            <p className={`text-xs text-muted-foreground px-1 ${isUser ? 'text-right' : 'text-left'}`}>
              {timeAgo}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
