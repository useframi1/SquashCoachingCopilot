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
                            "description": "Video UUID identifier",
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number (1, 2, 3, etc.)",
                        },
                        "player_id": {
                            "type": "integer",
                            "enum": [1, 2],
                            "description": "Optional: Filter by player (1 or 2). If omitted, returns aggregate for both players",
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds to filter the analysis window",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds to filter the analysis window",
                        },
                    },
                    "required": ["video_id"],
                },
            },
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
                            "description": "Video UUID identifier",
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number",
                        },
                        "player_id": {
                            "type": "integer",
                            "enum": [1, 2],
                            "description": "Optional: Filter by player (1 or 2). If omitted, returns aggregate",
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds",
                        },
                    },
                    "required": ["video_id"],
                },
            },
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
                            "description": "Video UUID identifier",
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number",
                        },
                        "player_id": {
                            "type": "integer",
                            "enum": [1, 2],
                            "description": "Optional: Filter by player who hit the ball",
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds",
                        },
                    },
                    "required": ["video_id"],
                },
            },
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
                            "description": "Video UUID identifier",
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number",
                        },
                        "player_id": {
                            "type": "integer",
                            "enum": [1, 2],
                            "description": "Optional: Filter by player",
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds",
                        },
                    },
                    "required": ["video_id"],
                },
            },
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
                            "description": "Video UUID identifier",
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number",
                        },
                        "player_id": {
                            "type": "integer",
                            "enum": [1, 2],
                            "description": "Optional: Filter by player",
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds",
                        },
                    },
                    "required": ["video_id"],
                },
            },
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
                            "description": "Video UUID identifier",
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number",
                        },
                        "player_id": {
                            "type": "integer",
                            "enum": [1, 2],
                            "description": "Optional: Filter by player who hit the wall",
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds",
                        },
                    },
                    "required": ["video_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_shot_effectiveness",
                "description": "Get comprehensive shot effectiveness metrics including shot placement (opponent distance moved), displacement from T-zone, depth dominance, and straight shot quality. Requires player_id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "Video UUID identifier",
                        },
                        "player_id": {
                            "type": "integer",
                            "enum": [1, 2],
                            "description": "Required: Player ID (1 or 2) whose shot effectiveness to analyze",
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number",
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds",
                        },
                    },
                    "required": ["video_id", "player_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_winning_efficiency",
                "description": "Get how many shots a player needed to make to win a point. Requires player_id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "Video UUID identifier",
                        },
                        "player_id": {
                            "type": "integer",
                            "enum": [1, 2],
                            "description": "Required: Player ID (1 or 2) whose winning stats to analyze",
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number",
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds",
                        },
                    },
                    "required": ["video_id", "player_id"],
                },
            },
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
                            "description": "Video UUID identifier",
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number",
                        },
                        "player_id": {
                            "type": "integer",
                            "enum": [1, 2],
                            "description": "Optional: Filter by player",
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds",
                        },
                    },
                    "required": ["video_id"],
                },
            },
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
                            "description": "Video UUID identifier",
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number",
                        },
                        "player_id": {
                            "type": "integer",
                            "enum": [1, 2],
                            "description": "Optional: Filter by player",
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds",
                        },
                    },
                    "required": ["video_id"],
                },
            },
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
                            "description": "Video UUID identifier",
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number",
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds",
                        },
                    },
                    "required": ["video_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_player_position_heatmap",
                "description": "Get player position heatmap showing where players spent most of their time on court as a 2D grid",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "Video UUID identifier",
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number",
                        },
                        "player_id": {
                            "type": "integer",
                            "enum": [1, 2],
                            "description": "Optional: Filter by player",
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds",
                        },
                    },
                    "required": ["video_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_wall_hits_heatmap",
                "description": "Get wall hit heatmap showing where the ball hits the front wall as a 2D grid",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "Video UUID identifier",
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number",
                        },
                        "player_id": {
                            "type": "integer",
                            "enum": [1, 2],
                            "description": "Optional: Filter by player who hit the ball",
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds",
                        },
                    },
                    "required": ["video_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_rally_timeline",
                "description": "Get per-rally timeline data showing progression of the match rally by rally",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "Video UUID identifier",
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number",
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds",
                        },
                    },
                    "required": ["video_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_momentum_timeline",
                "description": "Get momentum timeline showing cumulative score differential between players over time to identify momentum shifts",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "Video UUID identifier",
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number",
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds",
                        },
                    },
                    "required": ["video_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_match_summary",
                "description": "Get match summary including game results, overall match winner, and scoring information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "Video UUID identifier",
                        }
                    },
                    "required": ["video_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_let_stats",
                "description": "Get let/replay statistics showing frequency of lets during the match",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "Video UUID identifier",
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number",
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds",
                        },
                    },
                    "required": ["video_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_break_time",
                "description": "Get break time statistics between rallies including average, min, max, and standard deviation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "Video UUID identifier",
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number",
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds",
                        },
                    },
                    "required": ["video_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_longest_rally",
                "description": "Get the longest rally in the match by duration and shot count",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "Video UUID identifier",
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number",
                        },
                        "player_id": {
                            "type": "integer",
                            "enum": [1, 2],
                            "description": "Optional: Filter by player who won the rally",
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds",
                        },
                    },
                    "required": ["video_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_fastest_shot",
                "description": "Get the fastest shot in the match including ball speed, player who hit it, stroke type, and shot type",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "Video UUID identifier",
                        },
                        "game_number": {
                            "type": "integer",
                            "description": "Optional: Filter by specific game number",
                        },
                        "player_id": {
                            "type": "integer",
                            "enum": [1, 2],
                            "description": "Optional: Filter by player who hit the shot",
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Optional: Start time in seconds",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Optional: End time in seconds",
                        },
                    },
                    "required": ["video_id"],
                },
            },
        },
    ]


# System prompt template
SYSTEM_PROMPT_TEMPLATE = """You are an AI assistant for SquashCoachingCopilot, a professional squash match analytics platform.

## Your Role
You help coaches and players analyze squash match performance by retrieving and interpreting analytics data through specialized functions. You are a data analyst assistant, not a general chatbot.

## Available Analytics Functions

**Basic Performance Metrics:**
- get_stroke_distribution: Forehand vs backhand distribution with counts and percentages
- get_shot_types_distribution: Shot type breakdown (drives, drops, lobs, boasts, kills)
- get_ball_speed: Ball speed statistics (mean, min, max, std dev in m/s)
- get_rhythm_disruption: Rhythm metrics (coefficient of variation for unpredictability)

**Positioning & Court Coverage:**
- get_court_quadrants: Player positioning across 4 court quadrants
- get_wall_quadrants: Wall hit distribution across 4 wall quadrants
- get_player_position_heatmap: 2D heatmap of player court positioning
- get_wall_hits_heatmap: 2D heatmap of wall hit locations

**Shot Quality & Effectiveness:**
- get_shot_effectiveness: Shot placement (opponent distance moved), displacement from T, depth dominance, straight shot quality (requires player_id)
- get_winning_efficiency: Win efficiency, points won, shots taken, rallies played (requires player_id)

**Movement Analysis:**
- get_movement_metrics: Distance covered, average per rally, distance to ball
- get_t_zone_occupancy: T-zone positioning metrics (% time in T, avg time to T, success rate)

**Rally & Match Analysis:**
- get_rally_intensity: Average seconds per shot (match pace indicator)
- get_rally_timeline: Per-rally timeline showing match progression
- get_momentum_timeline: Score differential over time to identify momentum shifts
- get_match_summary: Game results, match winner, scoring information
- get_let_stats: Let/replay frequency statistics
- get_break_time: Break time statistics between rallies

**Match Highlights:**
- get_longest_rally: Longest rally by duration and shot count
- get_fastest_shot: Fastest shot with player, stroke type, and shot type

## Critical Guidelines

1. **ALWAYS use functions - NEVER make up, estimate, or guess data**
   - If you cannot answer with available functions, clearly state: "I don't have access to that specific data through the available analytics functions."
   - Never provide hypothetical numbers or estimates

2. **Context Requirements:**
   - If video_id is not in context, ask: "Which video/match would you like me to analyze? I need a video ID to retrieve analytics data."
   - For player-specific queries (shot effectiveness, winning stats), ask which player (1 or 2) if not specified
   - Be explicit about missing context before attempting function calls

3. **Data-Driven Responses:**
   - Always cite specific numbers, percentages, and metrics from function results
   - Compare metrics when relevant (e.g., "Player 1: 65.2%, Player 2: 34.8%")
   - Provide context for numbers (e.g., "15.3 m/s, which is above average for recreational play")

4. **Multi-Function Queries:**
   - For player comparisons, call the same function with different player_ids
   - For comprehensive analysis, combine related functions (e.g., ball speed + rhythm disruption for pace analysis)
   - Execute all necessary function calls in parallel when possible

5. **Clear Communication:**
   - Structure responses with clear sections (e.g., "Player 1 Analysis:", "Player 2 Analysis:", "Key Insights:")
   - Use bullet points for multiple metrics
   - Highlight significant differences or patterns
   - Translate technical metrics into actionable insights when appropriate

6. **Limitations:**
   - If asked about data not available through functions (e.g., video replay, frame-by-frame analysis), respond: "That functionality is not available through the analytics API. I can only access aggregated statistical data."
   - For questions outside squash analytics (general chat, unrelated topics), redirect: "I'm specifically designed for squash match analytics. Please ask about match statistics, player performance, or game analysis."

7. **Error Handling:**
   - If a function returns no data or an error, inform the user clearly
   - Suggest alternative queries or functions that might help
   - Don't assume why data is missing - just report it

## Current Context
- Video ID: {video_id}
- Player ID: {player_id}

## Response Format
- Be concise but informative
- Lead with the most important insight
- Support claims with specific data points
- End with actionable takeaways when appropriate

Remember: You are a specialized analytics assistant. Stay focused on data retrieval and interpretation. Never make up data, and always be transparent about limitations."""


def get_system_prompt(video_id: str = None, player_id: int = None, db_session = None) -> str:
    """
    Get system prompt with current context.

    Args:
        video_id: Current video ID context
        player_id: Current player ID context
        db_session: Optional database session for fetching player names

    Returns:
        Formatted system prompt
    """
    from backend.models.video import Video

    player_1_name = "Player 1"
    player_2_name = "Player 2"

    # Fetch player names from database if video_id is provided
    if video_id and db_session:
        try:
            video = db_session.query(Video).filter(Video.id == video_id).first()
            if video:
                player_1_name = video.player_1_name or "Player 1"
                player_2_name = video.player_2_name or "Player 2"
        except Exception:
            pass  # Fall back to default names if query fails

    return SYSTEM_PROMPT_TEMPLATE.format(
        video_id=video_id or "Not set", player_id=player_id or "Not set"
    ) + f"\n\n## Player Names\n- Player 1: {player_1_name}\n- Player 2: {player_2_name}\n\nIMPORTANT: When referring to players in your responses, ALWAYS use their actual names ({player_1_name} and {player_2_name}) instead of 'Player 1' and 'Player 2'. Make your responses personal and specific to these players."
