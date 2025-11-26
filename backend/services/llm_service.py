"""LLM service for natural language analytics queries."""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI
from sqlalchemy.orm import Session

from backend.config import settings
from backend.models.llm_conversation import LLMConversation
from backend.models.video import Video
from backend.schemas.analysis import AnalyticsFilters
from backend.schemas.llm import (
    ConversationContext,
    FunctionCallMetadata,
    LLMQueryResponse,
    ResponseMetadata,
)
from backend.services.analysis_service import AnalysisService
from backend.services.llm_functions import get_analytics_functions, get_system_prompt

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM-powered natural language analytics queries."""

    def __init__(self, db: Session):
        self.db = db
        self.analysis_service = AnalysisService(db)
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key, timeout=30.0, max_retries=2
        )
        self.tools = get_analytics_functions()

    async def process_query(
        self,
        message: str,
        video_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        player_id: Optional[int] = None,
    ) -> LLMQueryResponse:
        """
        Process a natural language query.

        Args:
            message: User's natural language query
            video_id: Optional video ID context
            conversation_id: Optional conversation ID for multi-turn chat
            player_id: Optional player ID context

        Returns:
            LLMQueryResponse with answer and metadata
        """
        start_time = time.time()

        # 1. Get or create conversation
        conversation = self._get_or_create_conversation(
            conversation_id, video_id, player_id
        )

        # 2. Extract context
        context_video_id, context_player_id = self._extract_context(
            conversation, video_id, player_id
        )

        # 3. Build messages with system prompt
        messages = self._build_messages(
            conversation, message, context_video_id, context_player_id
        )

        # 4. Call OpenAI
        try:
            response = await self.client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                max_completion_tokens=settings.openai_max_tokens,
            )

            # Track tokens
            tokens_used = response.usage.total_tokens if response.usage else None

            # 5. Check for tool calls
            function_calls_metadata = []
            if response.choices[0].message.tool_calls:
                # Add assistant message to conversation
                messages.append(response.choices[0].message.model_dump())

                # Execute function calls
                for tool_call in response.choices[0].message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    logger.info(
                        f"Executing function: {function_name} with args: {arguments}"
                    )

                    # Inject video_id if not present
                    if "video_id" not in arguments and context_video_id:
                        arguments["video_id"] = context_video_id

                    # Execute function
                    try:
                        result = await self._execute_single_function(
                            function_name, arguments
                        )

                        # Add tool response to messages
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(result),
                            }
                        )

                        # Track function call
                        function_calls_metadata.append(
                            FunctionCallMetadata(
                                function_name=function_name,
                                arguments=arguments,
                                result_summary=self._summarize_result(
                                    function_name, result
                                ),
                            )
                        )
                    except Exception as e:
                        logger.error(f"Error executing function {function_name}: {e}")
                        # Add error response
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps({"error": str(e)}),
                            }
                        )

                # 6. Second call to synthesize results
                final_response = await self.client.chat.completions.create(
                    model=settings.openai_model,
                    messages=messages,
                    max_completion_tokens=settings.openai_max_tokens,
                )

                answer = final_response.choices[0].message.content
                if final_response.usage:
                    tokens_used = (tokens_used or 0) + final_response.usage.total_tokens

                # Add final assistant message to conversation
                messages.append(final_response.choices[0].message.model_dump())
            else:
                # No function calls - direct response
                answer = response.choices[0].message.content
                messages.append(response.choices[0].message.model_dump())

            # 7. Save conversation
            self._save_conversation(
                conversation, messages, context_video_id, context_player_id
            )

            # 8. Return response
            execution_time_ms = int((time.time() - start_time) * 1000)

            return LLMQueryResponse(
                conversation_id=conversation.id,
                answer=answer,
                function_calls=function_calls_metadata,
                context=ConversationContext(
                    video_id=context_video_id, player_id=context_player_id
                ),
                metadata=ResponseMetadata(
                    tokens_used=tokens_used,
                    execution_time_ms=execution_time_ms,
                    functions_executed=len(function_calls_metadata),
                ),
            )

        except Exception as e:
            logger.error(f"Error processing LLM query: {e}", exc_info=True)
            raise ValueError(f"Failed to process query: {str(e)}")

    def _get_or_create_conversation(
        self,
        conversation_id: Optional[str],
        video_id: Optional[str],
        player_id: Optional[int],
    ) -> LLMConversation:
        """Get existing conversation or create new one."""
        if conversation_id:
            conversation = (
                self.db.query(LLMConversation)
                .filter(LLMConversation.id == conversation_id)
                .first()
            )
            if not conversation:
                raise ValueError(f"Conversation {conversation_id} not found")
            return conversation

        # Create new conversation
        conversation = LLMConversation(
            id=str(uuid.uuid4()), video_id=video_id, player_id=player_id, messages=[]
        )
        self.db.add(conversation)
        self.db.commit()
        return conversation

    def _extract_context(
        self,
        conversation: LLMConversation,
        video_id: Optional[str],
        player_id: Optional[int],
    ) -> Tuple[Optional[str], Optional[int]]:
        """
        Extract video_id and player_id from conversation or parameters.

        Priority: parameter > conversation context > None

        Returns:
            Tuple of (video_id, player_id)
        """
        context_video_id = video_id or conversation.video_id
        context_player_id = player_id or conversation.player_id

        return context_video_id, context_player_id

    def _build_messages(
        self,
        conversation: LLMConversation,
        new_message: str,
        video_id: Optional[str],
        player_id: Optional[int],
    ) -> List[Dict[str, Any]]:
        """
        Build messages array for OpenAI API.

        Args:
            conversation: Current conversation
            new_message: New user message
            video_id: Current video context
            player_id: Current player context

        Returns:
            List of message dictionaries
        """
        messages = []

        # Add system prompt
        system_prompt = get_system_prompt(video_id, player_id)
        messages.append({"role": "system", "content": system_prompt})

        # Add conversation history (excluding system messages)
        for msg in conversation.messages:
            if msg.get("role") != "system":
                messages.append(msg)

        # Add new user message
        messages.append({"role": "user", "content": new_message})

        return messages

    async def _execute_single_function(
        self, function_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single analytics function.

        Args:
            function_name: Name of the function to execute
            arguments: Function arguments

        Returns:
            Function result as dictionary
        """
        video_id = arguments.pop("video_id")

        # Build filters from remaining arguments
        filters = AnalyticsFilters(
            **{
                k: v
                for k, v in arguments.items()
                if k in ["game_number", "player_id", "start_time", "end_time"]
            }
        )

        # Map function name to AnalysisService method
        if function_name == "get_stroke_distribution":
            result = self.analysis_service.get_stroke_distribution(video_id, filters)
        elif function_name == "get_shot_types_distribution":
            result = self.analysis_service.get_shot_type_distribution(video_id, filters)
        elif function_name == "get_ball_speed":
            result = self.analysis_service.get_ball_speed_analytics(video_id, filters)
        elif function_name == "get_rhythm_disruption":
            result = self.analysis_service.get_rhythm_disruption(video_id, filters)
        elif function_name == "get_court_quadrants":
            result = self.analysis_service.get_court_quadrant_distribution(
                video_id, filters
            )
        elif function_name == "get_wall_quadrants":
            result = self.analysis_service.get_wall_quadrant_distribution(
                video_id, filters
            )
        elif function_name == "get_shot_placement":
            # Requires player_id in arguments
            player_id = arguments.get("player_id")
            if not player_id:
                raise ValueError("get_shot_placement requires player_id")
            result = self.analysis_service.get_shot_placement_effectiveness(
                video_id, player_id, filters
            )
        elif function_name == "get_shot_effectiveness":
            # Requires player_id in arguments
            player_id = arguments.get("player_id")
            if not player_id:
                raise ValueError("get_shot_effectiveness requires player_id")
            result = self.analysis_service.get_shot_effectiveness(
                video_id, player_id, filters
            )
        elif function_name == "get_winning_stats":
            # Requires player_id in arguments
            player_id = arguments.get("player_id")
            if not player_id:
                raise ValueError("get_winning_stats requires player_id")
            result = self.analysis_service.get_winning_stats(
                video_id, player_id, filters
            )
        elif function_name == "get_movement_metrics":
            result = self.analysis_service.get_movement_metrics(video_id, filters)
        elif function_name == "get_t_zone_occupancy":
            result = self.analysis_service.get_t_zone_occupancy(video_id, filters)
        elif function_name == "get_rally_intensity":
            result = self.analysis_service.get_rally_intensity(video_id, filters)
        else:
            raise ValueError(f"Unknown function: {function_name}")

        return result.dict()

    def _summarize_result(self, function_name: str, result: Dict[str, Any]) -> str:
        """
        Create a brief summary of the function result.

        Args:
            function_name: Name of the function
            result: Function result

        Returns:
            Brief summary string
        """
        if "data" in result:
            data = result["data"]
            if isinstance(data, dict):
                if "total" in data:
                    return f"Total: {data['total']}"
                elif "mean" in data:
                    return f"Mean: {data.get('mean', 'N/A')}"
            return "Data retrieved successfully"
        return "Success"

    def _save_conversation(
        self,
        conversation: LLMConversation,
        messages: List[Dict[str, Any]],
        video_id: Optional[str],
        player_id: Optional[int],
    ):
        """
        Save conversation messages and context to database.

        Args:
            conversation: Conversation object
            messages: All messages (including system, user, assistant, tool)
            video_id: Current video context
            player_id: Current player context
        """
        # Filter out system messages and convert to storable format
        storable_messages = []
        for msg in messages:
            if msg.get("role") != "system":
                # Convert message to storable format with timestamp
                storable_msg = {
                    "role": msg.get("role"),
                    "content": msg.get("content"),
                    "timestamp": datetime.utcnow().isoformat(),
                }

                # Include tool_call_id for tool messages
                if "tool_call_id" in msg:
                    storable_msg["tool_call_id"] = msg["tool_call_id"]

                # Include tool_calls for assistant messages
                if "tool_calls" in msg:
                    storable_msg["tool_calls"] = msg["tool_calls"]

                storable_messages.append(storable_msg)

        conversation.messages = storable_messages
        conversation.video_id = video_id
        conversation.player_id = player_id
        conversation.updated_at = datetime.utcnow()

        self.db.commit()

    def get_conversation(self, conversation_id: str) -> Optional[LLMConversation]:
        """Get conversation by ID."""
        return (
            self.db.query(LLMConversation)
            .filter(LLMConversation.id == conversation_id)
            .first()
        )

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete conversation by ID.

        Args:
            conversation_id: Conversation ID

        Returns:
            True if deleted, False if not found
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False

        self.db.delete(conversation)
        self.db.commit()
        return True

    def list_conversations(
        self, limit: int = 10, offset: int = 0
    ) -> List[LLMConversation]:
        """
        List conversations.

        Args:
            limit: Maximum number of conversations to return
            offset: Number of conversations to skip

        Returns:
            List of conversations
        """
        return (
            self.db.query(LLMConversation)
            .order_by(LLMConversation.updated_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )
