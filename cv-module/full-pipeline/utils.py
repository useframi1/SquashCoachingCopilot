def get_real_coordinates(player_results: dict) -> dict:
    """Convert player pixel positions to real-world court coordinates."""
    return {
        1: player_results[1]["real_position"],
        2: player_results[2]["real_position"],
    }
