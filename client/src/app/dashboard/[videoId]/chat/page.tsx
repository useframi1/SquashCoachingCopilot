"use client";

import {useParams} from "next/navigation";
import {useState, useRef, useEffect} from "react";
import {useMutation} from "@tanstack/react-query";
import {MessageSquare, Trash2, Info} from "lucide-react";
import {ChatMessage} from "@/components/chat/ChatMessage";
import {ChatInput} from "@/components/chat/ChatInput";
import {sendLLMQuery} from "@/lib/api/llm";
import {usePlayerNames} from "@/lib/hooks/usePlayerNames";
import type {LLMMessage, LLMQueryResponse} from "@/types/api";

const STORAGE_KEY_PREFIX = "chat_conversation_";

/**
 * AI Chat Assistant tab
 * Allows users to query match analytics using natural language
 */
export default function ChatPage() {
    const params = useParams();
    const videoId = params.videoId as string;
    const storageKey = `${STORAGE_KEY_PREFIX}${videoId}`;
    const {player1Name, player2Name} = usePlayerNames(videoId);

    const [messages, setMessages] = useState<LLMMessage[]>([]);
    const [conversationId, setConversationId] = useState<string | null>(null);
    const [metadata, setMetadata] = useState<{
        tokens_used: number;
        execution_time_ms: number;
        functions_executed: number;
    } | null>(null);

    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Load conversation from localStorage on mount
    useEffect(() => {
        const stored = localStorage.getItem(storageKey);
        if (stored) {
            try {
                const data = JSON.parse(stored);
                setMessages(data.messages || []);
                setConversationId(data.conversationId || null);
                setMetadata(data.metadata || null);
            } catch (error) {
                console.error(
                    "Error loading conversation from localStorage:",
                    error
                );
            }
        }
    }, [storageKey]);

    // Save conversation to localStorage whenever it changes
    useEffect(() => {
        if (messages.length > 0 || conversationId) {
            localStorage.setItem(
                storageKey,
                JSON.stringify({
                    messages,
                    conversationId,
                    metadata,
                })
            );
        }
    }, [messages, conversationId, metadata, storageKey]);

    // Auto-scroll to bottom when new messages arrive
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({behavior: "smooth"});
    }, [messages]);

    // Send message mutation
    const {mutate: sendMessage, isPending} = useMutation({
        mutationFn: sendLLMQuery,
        onSuccess: (response: LLMQueryResponse) => {
            console.log("LLM Response:", response);
            console.log("Answer:", response.answer);
            console.log("Function calls:", response.function_calls);

            // Update conversation ID
            setConversationId(response.conversation_id);

            // Update metadata
            setMetadata(response.metadata);

            // Add assistant message to the list
            const assistantMessage: LLMMessage = {
                role: "assistant",
                content: response.answer || "No response received",
                timestamp: new Date().toISOString(),
                function_calls:
                    response.function_calls.length > 0
                        ? response.function_calls
                        : undefined,
            };

            console.log("Assistant message to display:", assistantMessage);
            setMessages((prev) => [...prev, assistantMessage]);
        },
        onError: (error: any) => {
            console.error("Error sending message:", error);

            // Add error message
            const errorMessage: LLMMessage = {
                role: "assistant",
                content: `Sorry, I encountered an error: ${
                    error.response?.data?.detail ||
                    error.message ||
                    "Unknown error"
                }`,
                timestamp: new Date().toISOString(),
            };

            setMessages((prev) => [...prev, errorMessage]);
        },
    });

    const handleSendMessage = (message: string) => {
        // Add user message to the list
        const userMessage: LLMMessage = {
            role: "user",
            content: message,
            timestamp: new Date().toISOString(),
        };

        setMessages((prev) => [...prev, userMessage]);

        // Send to API
        sendMessage({
            message,
            video_id: videoId,
            conversation_id: conversationId || undefined,
        });
    };

    const handleClearChat = () => {
        setMessages([]);
        setConversationId(null);
        setMetadata(null);
        localStorage.removeItem(storageKey);
    };

    return (
        <div className="flex flex-col h-full">
            {/* Header */}
            <div className="border-b border-border bg-card px-6 py-4 flex items-center justify-between shrink-0">
                <div className="flex items-center gap-3">
                    <MessageSquare className="w-6 h-6 text-primary" />
                    <div>
                        <h2 className="text-lg font-semibold text-foreground">
                            AI Assistant
                        </h2>
                        <p className="text-sm text-muted-foreground">
                            Ask questions about match analytics and performance
                        </p>
                    </div>
                </div>

                <div className="flex items-center gap-4">
                    {/* Metadata Display */}
                    {metadata && (
                        <div className="flex items-center gap-4 text-xs text-muted-foreground bg-muted px-4 py-2 rounded-lg border border-border">
                            <div className="flex items-center gap-1">
                                <Info className="w-4 h-4" />
                                <span>Tokens: {metadata.tokens_used}</span>
                            </div>
                            <div>Time: {metadata.execution_time_ms}ms</div>
                            <div>Functions: {metadata.functions_executed}</div>
                        </div>
                    )}

                    {/* Clear Chat Button */}
                    {messages.length > 0 && (
                        <button
                            onClick={handleClearChat}
                            className="flex items-center gap-2 px-4 py-2 text-sm text-destructive hover:bg-destructive/10 rounded-lg transition-colors"
                        >
                            <Trash2 className="w-4 h-4" />
                            Clear Chat
                        </button>
                    )}
                </div>
            </div>

            {/* Messages Container */}
            <div className="flex-1 overflow-y-auto bg-muted/30 px-6 py-6">
                {messages.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center text-center max-w-4xl mx-auto">
                        <div className="bg-muted/50 rounded-full p-6 mb-6">
                            <MessageSquare className="w-12 h-12 text-muted-foreground" />
                        </div>
                        <h3 className="text-xl font-semibold text-foreground mb-3">
                            Start a conversation
                        </h3>
                        <p className="text-muted-foreground max-w-md mb-8">
                            Ask me anything about the match analytics. I can
                            help you analyze player performance, shot patterns,
                            movement metrics, and more.
                        </p>
                        <div className="grid grid-cols-1 gap-3 w-full max-w-2xl">
                            <div className="bg-card border border-border rounded-lg p-4 text-left hover:border-primary/50 transition-colors">
                                <p className="text-sm text-foreground">
                                    <span className="font-semibold text-primary">
                                        Example:
                                    </span>{" "}
                                    "What was {player1Name}'s average ball speed?"
                                </p>
                            </div>
                            <div className="bg-card border border-border rounded-lg p-4 text-left hover:border-primary/50 transition-colors">
                                <p className="text-sm text-foreground">
                                    <span className="font-semibold text-primary">
                                        Example:
                                    </span>{" "}
                                    "Compare the T-zone occupancy between both
                                    players"
                                </p>
                            </div>
                            <div className="bg-card border border-border rounded-lg p-4 text-left hover:border-primary/50 transition-colors">
                                <p className="text-sm text-foreground">
                                    <span className="font-semibold text-primary">
                                        Example:
                                    </span>{" "}
                                    "Show me shot effectiveness metrics for
                                    {player2Name}"
                                </p>
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="max-w-5xl mx-auto">
                        {messages.map((msg, idx) => (
                            <ChatMessage key={idx} message={msg} />
                        ))}
                        <div ref={messagesEndRef} />
                    </div>
                )}
            </div>

            {/* Input Area - Fixed to bottom */}
            <div className="shrink-0">
                <ChatInput
                    onSendMessage={handleSendMessage}
                    disabled={isPending}
                    placeholder="Ask about match analytics..."
                />
            </div>
        </div>
    );
}
