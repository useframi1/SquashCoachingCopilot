"""Pydantic schemas for LLM-related API operations."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LLMQueryRequest(BaseModel):
    """Request for natural language analytics query."""

    message: str = Field(..., min_length=1, max_length=2000, description="User query")
    video_id: Optional[str] = Field(None, description="Video UUID (can be inferred from conversation)")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for multi-turn chat")
    player_id: Optional[int] = Field(None, ge=1, le=2, description="Optional player context (1 or 2)")


class ConversationContext(BaseModel):
    """Current conversation context."""

    video_id: Optional[str] = None
    player_id: Optional[int] = None


class FunctionCallMetadata(BaseModel):
    """Metadata about a function call."""

    function_name: str
    arguments: Dict[str, Any]
    result_summary: str  # Brief summary of the result


class ResponseMetadata(BaseModel):
    """Metadata about the LLM response."""

    tokens_used: Optional[int] = None
    execution_time_ms: Optional[int] = None
    functions_executed: int


class LLMQueryResponse(BaseModel):
    """Response from natural language query."""

    conversation_id: str
    answer: str  # Natural language response
    function_calls: List[FunctionCallMetadata] = Field(default_factory=list)
    context: ConversationContext
    metadata: ResponseMetadata


class ConversationMessage(BaseModel):
    """Single message in a conversation."""

    role: str  # user, assistant, system, tool
    content: str
    timestamp: str
    function_calls: Optional[List[Dict[str, Any]]] = None


class ConversationDetail(BaseModel):
    """Detailed conversation history."""

    id: str
    video_id: Optional[str]
    player_id: Optional[int]
    messages: List[ConversationMessage]
    created_at: str
    updated_at: str
