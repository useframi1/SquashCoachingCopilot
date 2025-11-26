"""Function definitions for LLM routing to analytics endpoints."""

ANALYTICS_FUNCTIONS = [
    {
        "name": "get_stroke_distribution",
        "description": "Get distribution of different stroke types (forehand vs backhand) in the match",
        "endpoint": "/analytics/stroke-distribution",
        "parameters": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string", "description": "The video/match ID"},
                "rally_id": {"type": "integer", "description": "Optional specific rally ID"},
                "player_id": {"type": "integer", "description": "Optional player ID (1 or 2)", "enum": [1, 2]},
                "start_time": {"type": "number", "description": "Optional start time filter"},
                "end_time": {"type": "number", "description": "Optional end time filter"}
            },
            "required": ["video_id"]
        }
    },
    {
        "name": "get_shot_types_distribution",
        "description": "Get distribution of shot types (drives, drops, lobs, boasts, kills, etc.)",
        "endpoint": "/analytics/shot-types-distribution",
        "parameters": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string"},
                "rally_id": {"type": "integer"},
                "player_id": {"type": "integer", "enum": [1, 2]},
                "start_time": {"type": "number"},
                "end_time": {"type": "number"}
            },
            "required": ["video_id"]
        }
    },
    {
        "name": "get_ball_speed",
        "description": "Get ball speed analytics including mean, min, max, and standard deviation",
        "endpoint": "/analytics/ball-speed",
        "parameters": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string"},
                "rally_id": {"type": "integer"},
                "player_id": {"type": "integer", "enum": [1, 2]},
                "start_time": {"type": "number"},
                "end_time": {"type": "number"}
            },
            "required": ["video_id"]
        }
    },
    {
        "name": "get_rhythm_disruption",
        "description": "Analyze rhythm disruption through variance and coefficient of variation in ball speed and shot height. Higher CV means more unpredictable play",
        "endpoint": "/analytics/rhythm-disruption",
        "parameters": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string"},
                "rally_id": {"type": "integer"},
                "player_id": {"type": "integer", "enum": [1, 2]},
                "start_time": {"type": "number"},
                "end_time": {"type": "number"}
            },
            "required": ["video_id"]
        }
    },
    {
        "name": "get_player_heatmap",
        "description": "Get player position data for heatmap showing where player spent time on court",
        "endpoint": "/analytics/player-heatmap/{player_id}",
        "parameters": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string"},
                "player_id": {"type": "integer", "enum": [1, 2], "description": "REQUIRED: Player ID (1 or 2)"},
                "rally_id": {"type": "integer"},
                "start_time": {"type": "number"},
                "end_time": {"type": "number"}
            },
            "required": ["video_id", "player_id"]
        }
    },
    {
        "name": "get_shot_placement",
        "description": "Analyze shot placement effectiveness by tracking how much the opponent had to move after each shot",
        "endpoint": "/analytics/shot-placement/{player_id}",
        "parameters": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string"},
                "player_id": {"type": "integer", "enum": [1, 2], "description": "REQUIRED: Player ID (1 or 2)"},
                "rally_id": {"type": "integer"},
                "start_time": {"type": "number"},
                "end_time": {"type": "number"}
            },
            "required": ["video_id", "player_id"]
        }
    },
    {
        "name": "get_court_quadrants",
        "description": "Get distribution of time spent in court quadrants (Front-Left, Front-Right, Back-Left, Back-Right)",
        "endpoint": "/analytics/court-quadrants",
        "parameters": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string"},
                "rally_id": {"type": "integer"},
                "player_id": {"type": "integer", "enum": [1, 2]},
                "start_time": {"type": "number"},
                "end_time": {"type": "number"}
            },
            "required": ["video_id"]
        }
    },
    {
        "name": "get_rally_stats",
        "description": "Get rally statistics including duration, stroke count, and aggregate stats for analyzing match pace",
        "endpoint": "/analytics/rally-stats",
        "parameters": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string"},
                "rally_id": {"type": "integer"},
                "start_time": {"type": "number"},
                "end_time": {"type": "number"}
            },
            "required": ["video_id"]
        }
    },
    {
        "name": "get_wall_hits_heatmap",
        "description": "Get wall hit positions for heatmap showing where ball hits the wall (shot placement patterns)",
        "endpoint": "/analytics/wall-hits-heatmap",
        "parameters": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string"},
                "rally_id": {"type": "integer"},
                "player_id": {"type": "integer", "enum": [1, 2]},
                "start_time": {"type": "number"},
                "end_time": {"type": "number"}
            },
            "required": ["video_id"]
        }
    },
    {
        "name": "get_wall_quadrants",
        "description": "Get distribution of wall hits across front wall quadrants (Top-Left, Top-Right, Bottom-Left, Bottom-Right)",
        "endpoint": "/analytics/wall-quadrants",
        "parameters": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string"},
                "rally_id": {"type": "integer"},
                "player_id": {"type": "integer", "enum": [1, 2]},
                "start_time": {"type": "number"},
                "end_time": {"type": "number"}
            },
            "required": ["video_id"]
        }
    },
    {
        "name": "get_winning_stats",
        "description": "Get winning statistics and efficiency metrics (points won per shot ratio, shot efficiency)",
        "endpoint": "/analytics/winning-stats",
        "parameters": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string"},
                "rally_id": {"type": "integer"},
                "player_id": {"type": "integer", "enum": [1, 2]},
                "start_time": {"type": "number"},
                "end_time": {"type": "number"}
            },
            "required": ["video_id"]
        }
    }
]
