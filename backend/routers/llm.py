"""LLM API endpoints."""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.models.database import get_db
from backend.schemas.llm import (
    ConversationDetail,
    ConversationMessage,
    LLMQueryRequest,
    LLMQueryResponse,
)
from backend.services.llm_service import LLMService

router = APIRouter(prefix="/api/llm", tags=["llm"])


def get_llm_service(db: Session = Depends(get_db)) -> LLMService:
    """Dependency injection for LLM service."""
    return LLMService(db)


@router.post("/query", response_model=LLMQueryResponse)
async def query_llm(
    request: LLMQueryRequest,
    service: LLMService = Depends(get_llm_service),
):
    """
    Process a natural language analytics query.

    The LLM will translate the query into analytics function calls,
    execute them, and return a natural language response with insights.

    Features:
    - Multi-turn conversations (use conversation_id)
    - Context tracking (video_id, player_id)
    - Multi-function orchestration for complex queries
    - Clarifying questions when context is missing

    Example queries:
    - "Show me player 1's forehand vs backhand stats"
    - "Compare player 1 and 2's shot effectiveness"
    - "What's the average ball speed in game 2?"
    """
    try:
        return await service.process_query(
            message=request.message,
            video_id=request.video_id,
            conversation_id=request.conversation_id,
            player_id=request.player_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str,
    service: LLMService = Depends(get_llm_service),
):
    """
    Get conversation history by ID.

    Returns all messages in the conversation with timestamps.
    """
    conversation = service.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Convert to response schema
    messages = [
        ConversationMessage(
            role=msg.get("role"),
            content=msg.get("content", ""),
            timestamp=msg.get("timestamp", ""),
            function_calls=msg.get("tool_calls")
        )
        for msg in conversation.messages
    ]

    return ConversationDetail(
        id=conversation.id,
        video_id=conversation.video_id,
        player_id=conversation.player_id,
        messages=messages,
        created_at=conversation.created_at.isoformat(),
        updated_at=conversation.updated_at.isoformat(),
    )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    service: LLMService = Depends(get_llm_service),
):
    """
    Delete a conversation.

    This will permanently remove the conversation and all its messages.
    """
    deleted = service.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"message": "Conversation deleted successfully"}


@router.get("/conversations")
async def list_conversations(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    service: LLMService = Depends(get_llm_service),
):
    """
    List recent conversations.

    Returns conversations ordered by most recently updated.
    """
    conversations = service.list_conversations(limit=limit, offset=offset)

    return {
        "conversations": [
            {
                "id": conv.id,
                "video_id": conv.video_id,
                "player_id": conv.player_id,
                "message_count": len(conv.messages),
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
            }
            for conv in conversations
        ],
        "limit": limit,
        "offset": offset,
    }
