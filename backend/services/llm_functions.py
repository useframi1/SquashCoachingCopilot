"""OpenAI function definitions for analytics endpoints."""

from typing import List, Dict, Any


def get_analytics_functions() -> List[Dict[str, Any]]:
    """
    Get all analytics function definitions in OpenAI tools format.

    Returns:
        List of function definitions compatible with OpenAI Chat Completions API v1.0+
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "get_stroke_distribution",
                "description": "Get forehand vs backhand stroke distribution with counts and percentages for the specified video",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "Video UUID identifier"
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number (1, 2, 3, etc.)"
                        },
                        "player_id": {
                            "type": "integer",
                            "enum": [1, 2],
                            "description": "Optional: Filter by player (1 or 2). If omitted, returns aggregate for both players"
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds to filter the analysis window"
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds to filter the analysis window"
                        }
                    },
                    "required": ["video_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_shot_types_distribution",
                "description": "Get distribution of shot types (drives, drops, lobs, boasts, kills) with counts and percentages",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "Video UUID identifier"
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number"
                        },
                        "player_id": {
                            "type": "integer",
                            "enum": [1, 2],
                            "description": "Optional: Filter by player (1 or 2). If omitted, returns aggregate"
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds"
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds"
                        }
                    },
                    "required": ["video_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_ball_speed",
                "description": "Get ball speed statistics including mean, min, max, and standard deviation in meters per second",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "Video UUID identifier"
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number"
                        },
                        "player_id": {
                            "type": "integer",
                            "enum": [1, 2],
                            "description": "Optional: Filter by player who hit the ball"
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds"
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds"
                        }
                    },
                    "required": ["video_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_rhythm_disruption",
                "description": "Get rhythm disruption metrics including coefficient of variation for ball speed and wall hit height to measure unpredictability",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "Video UUID identifier"
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number"
                        },
                        "player_id": {
                            "type": "integer",
                            "enum": [1, 2],
                            "description": "Optional: Filter by player"
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds"
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds"
                        }
                    },
                    "required": ["video_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_court_quadrants",
                "description": "Get distribution of player positioning across 4 court quadrants (Front-Left, Front-Right, Back-Left, Back-Right)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "Video UUID identifier"
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number"
                        },
                        "player_id": {
                            "type": "integer",
                            "enum": [1, 2],
                            "description": "Optional: Filter by player"
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds"
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds"
                        }
                    },
                    "required": ["video_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_wall_quadrants",
                "description": "Get distribution of wall hits across 4 wall quadrants (Top-Left, Top-Right, Bottom-Left, Bottom-Right)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "Video UUID identifier"
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number"
                        },
                        "player_id": {
                            "type": "integer",
                            "enum": [1, 2],
                            "description": "Optional: Filter by player who hit the wall"
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds"
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds"
                        }
                    },
                    "required": ["video_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_shot_placement",
                "description": "Get shot placement effectiveness measured by average distance opponent had to move to reach the ball. Requires player_id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "Video UUID identifier"
                        },
                        "player_id": {
                            "type": "integer",
                            "enum": [1, 2],
                            "description": "Required: Player ID (1 or 2) whose shot placement to analyze"
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number"
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds"
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds"
                        }
                    },
                    "required": ["video_id", "player_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_shot_effectiveness",
                "description": "Get shot effectiveness metrics including average displacement from T-zone, depth dominance, and straight shot quality. Requires player_id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "Video UUID identifier"
                        },
                        "player_id": {
                            "type": "integer",
                            "enum": [1, 2],
                            "description": "Required: Player ID (1 or 2) whose shot effectiveness to analyze"
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number"
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds"
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds"
                        }
                    },
                    "required": ["video_id", "player_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_winning_stats",
                "description": "Get winning statistics including efficiency (points won per shot), points won, total shots, and rallies played. Requires player_id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "Video UUID identifier"
                        },
                        "player_id": {
                            "type": "integer",
                            "enum": [1, 2],
                            "description": "Required: Player ID (1 or 2) whose winning stats to analyze"
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number"
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds"
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds"
                        }
                    },
                    "required": ["video_id", "player_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_movement_metrics",
                "description": "Get movement metrics including total distance covered, average distance per rally, and distance to ball",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "Video UUID identifier"
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number"
                        },
                        "player_id": {
                            "type": "integer",
                            "enum": [1, 2],
                            "description": "Optional: Filter by player"
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds"
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds"
                        }
                    },
                    "required": ["video_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_t_zone_occupancy",
                "description": "Get T-zone positioning metrics including percentage of time in T-zone, average time to reach T, and T-zone success rate",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "Video UUID identifier"
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number"
                        },
                        "player_id": {
                            "type": "integer",
                            "enum": [1, 2],
                            "description": "Optional: Filter by player"
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds"
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds"
                        }
                    },
                    "required": ["video_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_rally_intensity",
                "description": "Get rally intensity metrics measured by average seconds per shot, indicating match pace",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "Video UUID identifier"
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number"
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds"
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds"
                        }
                    },
                    "required": ["video_id"]
                }
            }
        }
    ]


# System prompt template
SYSTEM_PROMPT_TEMPLATE = """You are an AI assistant for SquashCoachingCopilot analytics.

Your role:
- Help users query squash match analytics using provided functions
- Provide clear, data-driven insights with numbers and percentages
- Ask clarifying questions when context is missing
- Call multiple functions when needed to answer complex queries

Available analytics:
- Stroke distribution, shot types, ball speed, rhythm
- Court/wall quadrants distribution
- Shot placement, effectiveness, winning stats
- Movement metrics, T-zone occupancy, rally intensity

Guidelines:
1. Always use functions - never make up data
2. Ask for video_id if not in context
3. Ask for player_id if needed for player-specific queries
4. Call multiple functions for comparisons
5. Highlight key insights and patterns

Current context:
- Video ID: {video_id}
- Player ID: {player_id}
"""


def get_system_prompt(video_id: str = None, player_id: int = None) -> str:
    """
    Get system prompt with current context.

    Args:
        video_id: Current video ID context
        player_id: Current player ID context

    Returns:
        Formatted system prompt
    """
    return SYSTEM_PROMPT_TEMPLATE.format(
        video_id=video_id or "Not set",
        player_id=player_id or "Not set"
    )
