import { apiClient } from './client';
import type {
  LLMQueryRequest,
  LLMQueryResponse,
  LLMConversationsResponse,
  LLMConversationDetail,
} from '@/types/api';

/**
 * Send a message to the LLM and get a response
 */
export const sendLLMQuery = async (
  request: LLMQueryRequest
): Promise<LLMQueryResponse> => {
  const { data } = await apiClient.post<LLMQueryResponse>(
    '/api/llm/query',
    request
  );
  return data;
};

/**
 * Get list of conversations with pagination
 */
export const getConversations = async (
  limit: number = 10,
  offset: number = 0
): Promise<LLMConversationsResponse> => {
  const { data } = await apiClient.get<LLMConversationsResponse>(
    `/api/llm/conversations?limit=${limit}&offset=${offset}`
  );
  return data;
};

/**
 * Get full conversation history by ID
 */
export const getConversation = async (
  conversationId: string
): Promise<LLMConversationDetail> => {
  const { data } = await apiClient.get<LLMConversationDetail>(
    `/api/llm/conversations/${conversationId}`
  );
  return data;
};

/**
 * Delete a conversation
 */
export const deleteConversation = async (
  conversationId: string
): Promise<void> => {
  await apiClient.delete(`/api/llm/conversations/${conversationId}`);
};
