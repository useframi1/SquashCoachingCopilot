"""LLM-powered natural language routing to analytics endpoints."""

import json
import httpx
from typing import Optional, Dict, Any
from openai import OpenAI

from backend.services.analytics_functions import ANALYTICS_FUNCTIONS


class LLMRouterService:
    """Service for routing natural language queries to analytics APIs using LLM."""
        
    def __init__(self, api_base_url: str, openrouter_api_key: str = None):
        """
        Args:
            api_base_url: Base URL of analytics API
            openrouter_api_key: Not used for Ollama (kept for compatibility)
        """
        self.api_base_url = api_base_url.rstrip('/')
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",  # Ollama's API endpoint
            api_key="ollama"  # Ollama doesn't need real key but OpenAI client requires one
        )
        self.model = "qwen2.5:7b"  # Or any model you pulled
        self.http_client = httpx.Client(timeout=30.0)
    
    def route_query(
        self, 
        user_query: str, 
        video_id: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Route natural language query to appropriate analytics endpoint.
        
        Args:
            user_query: Natural language question
            video_id: Match/video ID
            context: Optional context (player_id, rally_id, etc.)
        
        Returns:
            Dict with success status, function called, and data
        """
        context = context or {}
        
        system_prompt = self._build_system_prompt(video_id, context)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        tools = self._prepare_tools()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Call the analytics API
                result = self.call_analytics_api(function_name, function_args)
                
                return {
                    "success": True,
                    "function_called": function_name,
                    "arguments": function_args,
                    "data": result
                }
            else:
                return {
                    "success": False,
                    "function_called": None,
                    "message": message.content
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def call_analytics_api(self, function_name: str, args: dict) -> Dict[str, Any]:
        """Execute API call to analytics endpoint."""
        func_def = next(
            (f for f in ANALYTICS_FUNCTIONS if f["name"] == function_name), 
            None
        )
        if not func_def:
            return {"error": f"Unknown function: {function_name}"}
        
        endpoint = func_def["endpoint"]
        video_id = args.pop("video_id")
        
        # Handle path parameters
        if "{player_id}" in endpoint:
            player_id = args.pop("player_id", None)
            if player_id is None:
                return {"error": "player_id is required for this endpoint"}
            endpoint = endpoint.format(player_id=player_id)
        
        url = f"{self.api_base_url}/{video_id}{endpoint}"
        params = {k: v for k, v in args.items() if v is not None}
        
        try:
            response = self.http_client.get(url, params=params)
            response.raise_for_status()
            print(response.json())
            return response.json()
        except httpx.HTTPStatusError as e:
            return {
                "error": f"API error: {e.response.status_code}",
                "detail": e.response.text
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _build_system_prompt(self, video_id: str, context: dict) -> str:
        """Build system prompt with context."""
        return f"""You are a squash match analytics assistant. Route user queries to the appropriate analytics function.

Current context:
- Video ID: {video_id}
- Player ID: {context.get('player_id', 'Not specified')}
- Rally ID: {context.get('rally_id', 'Not specified')}

Guidelines:
- If a query needs a player_id but none is provided, ask the user to specify which player (1 or 2)
- Use provided context values when appropriate
- Extract time ranges (start_time, end_time) from natural language if mentioned
"""
    
    def _prepare_tools(self) -> list:
        """Prepare tool definitions for OpenAI API."""
        tools = []
        for func in ANALYTICS_FUNCTIONS:
            tool_func = {
                "name": func["name"],
                "description": func["description"],
                "parameters": func["parameters"]
            }
            tools.append({"type": "function", "function": tool_func})
        return tools
    
    def close(self):
        """Close HTTP client."""
        self.http_client.close()
