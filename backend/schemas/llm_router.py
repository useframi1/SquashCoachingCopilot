"""Schemas for LLM router endpoint."""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class NaturalLanguageQueryRequest(BaseModel):
    """Request for natural language analytics query."""
    
    query: str = Field(..., description="Natural language question about match analytics")
    player_id: Optional[int] = Field(None, ge=1, le=2, description="Optional player context")
    rally_id: Optional[int] = Field(None, description="Optional rally context")
    start_time: Optional[float] = Field(None, ge=0, description="Optional time range start")
    end_time: Optional[float] = Field(None, ge=0, description="Optional time range end")


class NaturalLanguageQueryResponse(BaseModel):
    """Response from natural language query routing."""
    
    success: bool
    function_called: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    error: Optional[str] = None
