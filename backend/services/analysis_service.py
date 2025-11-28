"""Analysis service for querying processed video data - PostgreSQL optimized."""

import logging
from typing import Dict, List, Tuple

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.models.frame_data import FrameData
from backend.models.video import Video
from backend.models.game import Game
from backend.models.match import Match
from backend.schemas.analysis import (
    # Base schemas
    AnalyticsFilters,
    # Pattern 1: Distribution
    DistributionItem,
    SingleDistribution,
    # Pattern 2: Aggregate
    SingleAggregate,
    BallSpeedData,
    RhythmDisruptionData,
    WinningEfficiencyData,
    RallyIntensityData,
    # Pattern 3: Spatial
    HeatmapGrid,
    SpatialData,
    # Pattern 5: Extended
    SingleMovementMetrics,
    SingleTZoneMetrics,
    SingleShotEffectivenessMetrics,
    # Pattern 6: Time-Series
    RallyTimelineItem,
    MomentumTimelineItem,
    # Aggregate response schemas
    StrokeDistributionResponse,
    ShotTypeDistributionResponse,
    BallSpeedResponse,
    RhythmDisruptionResponse,
    PlayerPositionHeatmapResponse,
    CourtQuadrantResponse,
    WallHitHeatmapResponse,
    WallQuadrantResponse,
    WinningEfficiencyResponse,
    MovementMetricsResponse,
    TZoneOccupancyResponse,
    ShotEffectivenessResponse,
    RallyIntensityResponse,
    LetStatsData,
    LetStatsResponse,
    BreakTimeData,
    BreakTimeResponse,
    RallyTimelineResponse,
    MomentumTimelineResponse,
    # Per-game time-series schemas
    StrokeDistributionPerGameResponse,
    StrokeDistributionPerGameItem,
    ShotTypeDistributionPerGameResponse,
    ShotTypeDistributionPerGameItem,
    BallSpeedPerGameResponse,
    BallSpeedPerGameItem,
    RhythmDisruptionPerGameResponse,
    RhythmDisruptionPerGameItem,
    CourtQuadrantPerGameResponse,
    CourtQuadrantPerGameItem,
    WallQuadrantPerGameResponse,
    WallQuadrantPerGameItem,
    MovementMetricsPerGameResponse,
    MovementMetricsPerGameItem,
    TZoneOccupancyPerGameResponse,
    TZoneOccupancyPerGameItem,
    ShotEffectivenessPerGameResponse,
    ShotEffectivenessPerGameItem,
    WinningEfficiencyPerGameResponse,
    WinningEfficiencyPerGameItem,
    RallyIntensityPerGameResponse,
    RallyIntensityPerGameItem,
    # Per-rally time-series schemas
    StrokeDistributionPerRallyResponse,
    StrokeDistributionPerRallyItem,
    ShotTypeDistributionPerRallyResponse,
    ShotTypeDistributionPerRallyItem,
    BallSpeedPerRallyResponse,
    BallSpeedPerRallyItem,
    RhythmDisruptionPerRallyResponse,
    RhythmDisruptionPerRallyItem,
    CourtQuadrantPerRallyResponse,
    CourtQuadrantPerRallyItem,
    WallQuadrantPerRallyResponse,
    WallQuadrantPerRallyItem,
    MovementMetricsPerRallyResponse,
    MovementMetricsPerRallyItem,
    TZoneOccupancyPerRallyResponse,
    TZoneOccupancyPerRallyItem,
    ShotEffectivenessPerRallyResponse,
    ShotEffectivenessPerRallyItem,
    WinningEfficiencyPerRallyResponse,
    WinningEfficiencyPerRallyItem,
    RallyIntensityPerRallyResponse,
    RallyIntensityPerRallyItem,
    # Match highlights schemas
    LongestRallyResponse,
    LongestRallyData,
    FastestShotResponse,
    FastestShotData,
)
from backend.schemas.match import MatchSummaryResponse

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service for querying and analyzing processed video data - PostgreSQL optimized."""

    def __init__(self, db: Session):
        self.db = db

    def _get_video_or_404(self, video_id: str) -> Video:
        """Get video or raise error."""
        video = self.db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise ValueError(f"Video {video_id} not found")
        return video

    def _check_processed(self, video_id: str) -> None:
        """Check if video has been processed."""
        count = self.db.query(FrameData).filter(FrameData.video_id == video_id).count()
        if count == 0:
            raise ValueError(f"Video {video_id} has not been processed yet")

    def _build_where_clause(
        self, video_id: str, filters: AnalyticsFilters, include_player_id: bool = True
    ) -> Tuple[str, Dict]:
        """Build WHERE clause SQL and parameters for filters.

        Args:
            video_id: Video identifier
            filters: Analytics filters
            include_player_id: Whether to include player_id filter (default True).
                              Set to False for queries that handle player filtering manually
                              (e.g., position-based queries, movement metrics).
        """
        where_clauses = ["video_id = :video_id"]
        params = {"video_id": video_id}

        # If filtering by game number, get the rally range for that game
        if filters.game_number is not None:
            game = (
                self.db.query(Game)
                .filter(
                    Game.video_id == video_id, Game.game_number == filters.game_number
                )
                .first()
            )
            if not game:
                raise ValueError(
                    f"Game {filters.game_number} not found for video {video_id}"
                )
            where_clauses.append("rally_id BETWEEN :start_rally_id AND :end_rally_id")
            params["start_rally_id"] = game.start_rally_id
            params["end_rally_id"] = game.end_rally_id

        if filters.player_id is not None and include_player_id:
            where_clauses.append("racket_hit_player_id = :player_id")
            params["player_id"] = filters.player_id

        if filters.start_time is not None:
            where_clauses.append("timestamp >= :start_time")
            params["start_time"] = filters.start_time

        if filters.end_time is not None:
            where_clauses.append("timestamp <= :end_time")
            params["end_time"] = filters.end_time

        return " AND ".join(where_clauses), params

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _build_single_distribution(self, data: Dict[str, int]) -> SingleDistribution:
        """Build single distribution from count data.

        Args:
            data: Dictionary mapping labels to counts

        Returns:
            SingleDistribution with items and total
        """
        total = sum(data.values())
        items = [
            DistributionItem(
                label=label,
                count=count,
                percentage=(count / total * 100) if total > 0 else 0.0,
            )
            for label, count in sorted(data.items())
        ]
        return SingleDistribution(distribution=items, total=total)

    def _compute_heatmap_grid(
        self, points: List[Tuple[float, float]], bounds: Dict[str, float]
    ) -> HeatmapGrid:
        """Compute 32x50 density grid for court visualization (vectorized)."""
        WIDTH, HEIGHT = 32, 50
        grid = np.zeros((HEIGHT, WIDTH))

        if not points:
            return HeatmapGrid(
                grid=grid.tolist(), width=WIDTH, height=HEIGHT, bounds=bounds
            )

        x_min, x_max = bounds["x_min"], bounds["x_max"]
        y_min, y_max = bounds["y_min"], bounds["y_max"]

        # Convert to NumPy array for vectorized operations
        points_array = np.array(points)

        # Vectorized normalization
        x_norm = (points_array[:, 0] - x_min) / (x_max - x_min) * WIDTH
        y_norm = (points_array[:, 1] - y_min) / (y_max - y_min) * HEIGHT

        # Vectorized clamping and conversion to integers
        cols = np.clip(x_norm.astype(int), 0, WIDTH - 1)
        rows = np.clip(y_norm.astype(int), 0, HEIGHT - 1)

        # Efficient accumulation using np.add.at (handles duplicate indices)
        np.add.at(grid, (rows, cols), 1)

        # Normalize to percentages
        total = grid.sum()
        if total > 0:
            grid = grid / total * 100

        return HeatmapGrid(
            grid=grid.tolist(), width=WIDTH, height=HEIGHT, bounds=bounds
        )

    def _get_games_metadata(
        self, video_id: str, filters: AnalyticsFilters
    ) -> List[Dict]:
        """Get list of games with metadata for per-game time-series queries.

        Returns list of game metadata including game_number, rally range, times, etc.
        Respects game_number filter if provided.
        """
        params = {"video_id": video_id}
        where_clauses = ["video_id = :video_id"]

        if filters.game_number is not None:
            where_clauses.append("game_number = :game_number")
            params["game_number"] = filters.game_number

        where_sql = " AND ".join(where_clauses)

        query = text(
            f"""
            SELECT
                game_number,
                start_rally_id,
                end_rally_id,
                start_time,
                end_time,
                (end_time - start_time) as duration,
                (end_rally_id - start_rally_id + 1) as rally_count
            FROM games
            WHERE {where_sql}
            ORDER BY game_number
        """
        )

        results = self.db.execute(query, params).fetchall()

        return [
            {
                "game_number": row.game_number,
                "start_rally_id": row.start_rally_id,
                "end_rally_id": row.end_rally_id,
                "start_time": float(row.start_time),
                "end_time": float(row.end_time),
                "duration": float(row.duration),
                "rally_count": int(row.rally_count),
            }
            for row in results
        ]

    def _pivot_by_game(
        self, rows: List, game_metadata: List[Dict], metric_builder
    ) -> List[Dict]:
        """Pivot player rows into game items with player_1 and player_2 fields.

        Args:
            rows: Database rows with game_number and racket_hit_player_id
            game_metadata: List of game metadata dicts
            metric_builder: Callable that takes a row and returns metric dict

        Returns:
            List of game items with player_1 and player_2 nested data
        """
        # Group rows by game_number
        games_data = {}
        for row in rows:
            game_num = row.game_number
            player_id = (
                row.racket_hit_player_id
                if hasattr(row, "racket_hit_player_id")
                else None
            )

            if game_num not in games_data:
                games_data[game_num] = {"player_1": None, "player_2": None}

            if player_id:
                metrics = metric_builder(row)
                if player_id == 1:
                    games_data[game_num]["player_1"] = metrics
                else:
                    games_data[game_num]["player_2"] = metrics

        # Combine with game metadata
        result = []
        for game_meta in game_metadata:
            game_num = game_meta["game_number"]
            game_item = {**game_meta}

            if game_num in games_data:
                game_item["player_1"] = games_data[game_num]["player_1"]
                game_item["player_2"] = games_data[game_num]["player_2"]
            else:
                game_item["player_1"] = None
                game_item["player_2"] = None

            result.append(game_item)

        return result

    def _pivot_by_rally(self, rows: List, metric_builder) -> List[Dict]:
        """Pivot player rows into rally items with player_1 and player_2 fields.

        Args:
            rows: Database rows with rally_id and racket_hit_player_id
            metric_builder: Callable that takes a row and returns metric dict

        Returns:
            List of rally items with player_1 and player_2 nested data
        """
        rallies_data = {}

        for row in rows:
            rally_id = row.rally_id
            player_id = (
                row.racket_hit_player_id
                if hasattr(row, "racket_hit_player_id")
                else None
            )

            if rally_id not in rallies_data:
                rallies_data[rally_id] = {
                    "rally_id": rally_id,
                    "game_number": (
                        row.game_number if hasattr(row, "game_number") else None
                    ),
                    "rally_start_time": (
                        float(row.rally_start_time)
                        if hasattr(row, "rally_start_time")
                        else 0.0
                    ),
                    "rally_duration": (
                        float(row.rally_duration)
                        if hasattr(row, "rally_duration")
                        else 0.0
                    ),
                    "shot_count": (
                        int(row.shot_count)
                        if hasattr(row, "shot_count") and row.shot_count is not None
                        else 0
                    ),
                    "point_winner": (
                        int(row.point_winner)
                        if hasattr(row, "point_winner") and row.point_winner
                        else None
                    ),
                    "player_1": None,
                    "player_2": None,
                }

            if player_id:
                metrics = metric_builder(row)
                if player_id == 1:
                    rallies_data[rally_id]["player_1"] = metrics
                else:
                    rallies_data[rally_id]["player_2"] = metrics

                # Accumulate shot count
                if hasattr(row, "player_shot_count"):
                    rallies_data[rally_id]["shot_count"] += int(row.player_shot_count)

        return list(rallies_data.values())

    # ========================================================================
    # ANALYTICS ENDPOINT METHODS
    # ========================================================================

    def get_stroke_distribution(
        self, video_id: str, filters: AnalyticsFilters
    ) -> StrokeDistributionResponse:
        """Get stroke distribution analytics (forehand vs backhand) - PostgreSQL optimized.

        Returns aggregated totals if player_id filter is not specified,
        otherwise returns data for the specified player only.
        """
        logger.info(f"Computing stroke distribution for video {video_id}")
        self._check_processed(video_id)

        # Build WHERE clause
        where_sql, params = self._build_where_clause(video_id, filters)

        # Single SQL query to get counts by stroke type
        # If player_id filter is set, the WHERE clause already filters to that player
        query = text(
            f"""
            SELECT stroke_type, COUNT(*) as count
            FROM frame_data
            WHERE {where_sql}
              AND is_racket_hit = TRUE
              AND stroke_type IS NOT NULL
            GROUP BY stroke_type
            ORDER BY stroke_type
        """
        )

        results = self.db.execute(query, params).fetchall()

        # Build stroke data dictionary
        stroke_data = {"forehand": 0, "backhand": 0}
        for row in results:
            if row.stroke_type in stroke_data:
                stroke_data[row.stroke_type] = row.count

        distribution = self._build_single_distribution(stroke_data)

        return StrokeDistributionResponse(
            video_id=video_id, filters=filters, data=distribution
        )

    def get_shot_type_distribution(
        self, video_id: str, filters: AnalyticsFilters
    ) -> ShotTypeDistributionResponse:
        """Get shot type distribution analytics - PostgreSQL optimized.

        Returns aggregated totals if player_id filter is not specified,
        otherwise returns data for the specified player only.
        """
        logger.info(f"Computing shot type distribution for video {video_id}")
        self._check_processed(video_id)

        where_sql, params = self._build_where_clause(video_id, filters)

        # Single SQL query to get counts by shot type
        # If player_id filter is set, the WHERE clause already filters to that player
        query = text(
            f"""
            SELECT shot_type, COUNT(*) as count
            FROM frame_data
            WHERE {where_sql}
              AND is_racket_hit = TRUE
              AND shot_type IS NOT NULL
            GROUP BY shot_type
            ORDER BY shot_type
        """
        )

        results = self.db.execute(query, params).fetchall()

        # Build shot type data dictionary
        shot_data: Dict[str, int] = {}
        all_shot_types = set()

        for row in results:
            shot_data[row.shot_type] = row.count
            all_shot_types.add(row.shot_type)

        distribution = self._build_single_distribution(shot_data)

        return ShotTypeDistributionResponse(
            video_id=video_id,
            filters=filters,
            data=distribution,
            all_shot_types=sorted(list(all_shot_types)),
        )

    def get_ball_speed_analytics(
        self, video_id: str, filters: AnalyticsFilters
    ) -> BallSpeedResponse:
        """Get ball speed aggregate statistics - PostgreSQL optimized with STDDEV_POP.

        Returns aggregated totals if player_id filter is not specified,
        otherwise returns data for the specified player only.
        """
        logger.info(f"Computing ball speed analytics for video {video_id}")
        self._check_processed(video_id)

        where_sql, params = self._build_where_clause(video_id, filters)

        # Use PostgreSQL's built-in STDDEV_POP function
        # If player_id filter is set, the WHERE clause already filters to that player
        query = text(
            f"""
            SELECT
                AVG(ball_speed) as mean,
                MIN(ball_speed) as min,
                MAX(ball_speed) as max,
                STDDEV_POP(ball_speed) as std_dev,
                COUNT(*) as count
            FROM frame_data
            WHERE {where_sql}
              AND is_racket_hit = TRUE
              AND ball_speed IS NOT NULL
        """
        )

        result = self.db.execute(query, params).fetchone()

        # Build ball speed data
        data = BallSpeedData(
            mean_speed=float(result.mean) if result.mean else 0.0,
            min_speed=float(result.min) if result.min else 0.0,
            max_speed=float(result.max) if result.max else 0.0,
            std_dev=float(result.std_dev) if result.std_dev else 0.0,
            shot_count=int(result.count) if result.count else 0,
        )

        return BallSpeedResponse(video_id=video_id, filters=filters, data=data)

    def get_rhythm_disruption(
        self, video_id: str, filters: AnalyticsFilters
    ) -> RhythmDisruptionResponse:
        """Calculate rhythm disruption metrics - PostgreSQL optimized with VAR_POP.

        Returns aggregated metrics (average CV and variance) filtered by the provided filters.
        Higher CV indicates more unpredictable, rhythm-disrupting play.
        """
        logger.info(f"Computing rhythm disruption for video {video_id}")
        self._check_processed(video_id)

        where_sql, params = self._build_where_clause(video_id, filters)

        # Aggregate across all qualifying shots
        # Calculate variance and CV directly across all shots (not per-rally)
        query = text(
            f"""
            SELECT
                VAR_POP(ball_speed) as ball_speed_variance,
                STDDEV_POP(ball_speed) / NULLIF(AVG(ball_speed), 0) as ball_speed_cv,
                VAR_POP(wall_hit_height) as wall_hit_height_variance,
                STDDEV_POP(wall_hit_height) / NULLIF(AVG(wall_hit_height), 0) as wall_hit_height_cv,
                COUNT(*) as shot_count
            FROM frame_data
            WHERE {where_sql}
              AND is_racket_hit = TRUE
              AND ball_speed IS NOT NULL
              AND wall_hit_height IS NOT NULL
        """
        )

        result = self.db.execute(query, params).fetchone()

        data = RhythmDisruptionData(
            ball_speed_cv=float(result.ball_speed_cv) if result.ball_speed_cv else 0.0,
            ball_speed_variance=(
                float(result.ball_speed_variance) if result.ball_speed_variance else 0.0
            ),
            wall_hit_height_cv=(
                float(result.wall_hit_height_cv) if result.wall_hit_height_cv else 0.0
            ),
            wall_hit_height_variance=(
                float(result.wall_hit_height_variance)
                if result.wall_hit_height_variance
                else 0.0
            ),
            shot_count=int(result.shot_count) if result.shot_count else 0,
        )

        return RhythmDisruptionResponse(video_id=video_id, filters=filters, data=data)

    def get_player_position_heatmap(
        self, video_id: str, filters: AnalyticsFilters
    ) -> PlayerPositionHeatmapResponse:
        """Get player position heatmap - PostgreSQL optimized.

        Returns aggregated position points if player_id filter is not specified,
        otherwise returns data for the specified player only.
        """
        logger.info(f"Computing position heatmap for video {video_id}")
        self._check_processed(video_id)

        # Build WHERE clause WITHOUT player_id filter (we handle it separately for positions)
        where_sql, params = self._build_where_clause(
            video_id, filters, include_player_id=False
        )

        # If player_id is specified, query that player's positions only
        # Otherwise, aggregate positions from both players
        if filters.player_id:
            x_col = f"player_{filters.player_id}_x_meter"
            y_col = f"player_{filters.player_id}_y_meter"

            query = text(
                f"""
                SELECT {x_col} as x, {y_col} as y
                FROM frame_data
                WHERE {where_sql}
                  AND is_rally_frame = TRUE
                  AND {x_col} IS NOT NULL
                  AND {y_col} IS NOT NULL
            """
            )
        else:
            # Aggregate both players' positions
            query = text(
                f"""
                SELECT x, y FROM (
                    SELECT player_1_x_meter as x, player_1_y_meter as y
                    FROM frame_data
                    WHERE {where_sql}
                      AND is_rally_frame = TRUE
                      AND player_1_x_meter IS NOT NULL
                      AND player_1_y_meter IS NOT NULL

                    UNION ALL

                    SELECT player_2_x_meter as x, player_2_y_meter as y
                    FROM frame_data
                    WHERE {where_sql}
                      AND is_rally_frame = TRUE
                      AND player_2_x_meter IS NOT NULL
                      AND player_2_y_meter IS NOT NULL
                ) combined_positions
            """
            )

        results = self.db.execute(query, params).fetchall()
        points = [(row.x, row.y) for row in results]

        court_bounds = {"x_min": 0.0, "x_max": 6.4, "y_min": 0.0, "y_max": 9.75}
        heatmap_grid = self._compute_heatmap_grid(points, court_bounds)

        spatial_data = SpatialData(heatmap_grid=heatmap_grid)

        return PlayerPositionHeatmapResponse(
            video_id=video_id, filters=filters, data=spatial_data
        )

    # get_shot_placement_effectiveness removed - use get_shot_effectiveness instead

    def get_court_quadrant_distribution(
        self, video_id: str, filters: AnalyticsFilters
    ) -> CourtQuadrantResponse:
        """Calculate time spent in each court quadrant - PostgreSQL optimized.

        Returns aggregated totals if player_id filter is not specified,
        otherwise returns data for the specified player only.
        """
        logger.info(f"Computing court quadrant distribution for video {video_id}")
        self._check_processed(video_id)

        X_CUT = 3.2
        Y_CUT = 5.44

        # Build WHERE clause WITHOUT player_id filter (we handle it separately for positions)
        where_sql, params = self._build_where_clause(
            video_id, filters, include_player_id=False
        )

        # If player_id is specified, query that player's positions only
        # Otherwise, aggregate positions from both players
        if filters.player_id:
            x_col = f"player_{filters.player_id}_x_meter"
            y_col = f"player_{filters.player_id}_y_meter"

            query = text(
                f"""
                SELECT
                    quadrant,
                    COUNT(*) as count
                FROM (
                    SELECT
                        CASE
                            WHEN {y_col} < {Y_CUT} THEN
                                CASE WHEN {x_col} < {X_CUT} THEN 'Front-Left' ELSE 'Front-Right' END
                            ELSE
                                CASE WHEN {x_col} < {X_CUT} THEN 'Back-Left' ELSE 'Back-Right' END
                        END as quadrant
                    FROM frame_data
                    WHERE {where_sql}
                      AND is_rally_frame = TRUE
                      AND {x_col} IS NOT NULL
                      AND {y_col} IS NOT NULL
                ) subq
                GROUP BY quadrant
                ORDER BY quadrant
            """
            )
        else:
            # Aggregate both players' positions
            query = text(
                f"""
                SELECT
                    quadrant,
                    COUNT(*) as count
                FROM (
                    SELECT
                        CASE
                            WHEN player_1_y_meter < {Y_CUT} THEN
                                CASE WHEN player_1_x_meter < {X_CUT} THEN 'Front-Left' ELSE 'Front-Right' END
                            ELSE
                                CASE WHEN player_1_x_meter < {X_CUT} THEN 'Back-Left' ELSE 'Back-Right' END
                        END as quadrant
                    FROM frame_data
                    WHERE {where_sql}
                      AND is_rally_frame = TRUE
                      AND player_1_x_meter IS NOT NULL
                      AND player_1_y_meter IS NOT NULL

                    UNION ALL

                    SELECT
                        CASE
                            WHEN player_2_y_meter < {Y_CUT} THEN
                                CASE WHEN player_2_x_meter < {X_CUT} THEN 'Front-Left' ELSE 'Front-Right' END
                            ELSE
                                CASE WHEN player_2_x_meter < {X_CUT} THEN 'Back-Left' ELSE 'Back-Right' END
                        END as quadrant
                    FROM frame_data
                    WHERE {where_sql}
                      AND is_rally_frame = TRUE
                      AND player_2_x_meter IS NOT NULL
                      AND player_2_y_meter IS NOT NULL
                ) subq
                GROUP BY quadrant
                ORDER BY quadrant
            """
            )

        results = self.db.execute(query, params).fetchall()

        # Build quadrant data dictionary
        quadrant_data = {
            "Front-Left": 0,
            "Front-Right": 0,
            "Back-Left": 0,
            "Back-Right": 0,
        }
        for row in results:
            if row.quadrant in quadrant_data:
                quadrant_data[row.quadrant] = row.count

        distribution = self._build_single_distribution(quadrant_data)

        return CourtQuadrantResponse(
            video_id=video_id,
            filters=filters,
            data=distribution,
            quadrant_boundaries={"x_cut": X_CUT, "y_cut": Y_CUT},
        )

    def get_wall_hit_heatmap(
        self, video_id: str, filters: AnalyticsFilters
    ) -> WallHitHeatmapResponse:
        """Get wall hit distribution heatmap - PostgreSQL optimized.

        Returns aggregated wall hit positions if player_id filter is not specified,
        otherwise returns data for the specified player only.
        """
        logger.info(f"Computing wall hit distribution for video {video_id}")
        self._check_processed(video_id)

        # Build WHERE clause WITHOUT player_id filter (we use wall_hit_player_id instead)
        where_sql, params = self._build_where_clause(
            video_id, filters, include_player_id=False
        )

        # Add wall_hit_player_id filter if specified
        if filters.player_id:
            params["wall_hit_player_id"] = filters.player_id
            where_sql += " AND wall_hit_player_id = :wall_hit_player_id"

        query = text(
            f"""
            SELECT wall_hit_x_meter as x, wall_hit_y_meter as y
            FROM frame_data
            WHERE {where_sql}
              AND is_wall_hit = TRUE
              AND wall_hit_x_meter IS NOT NULL
              AND wall_hit_y_meter IS NOT NULL
        """
        )

        results = self.db.execute(query, params).fetchall()
        points = [(row.x, row.y) for row in results]

        wall_bounds = {"x_min": 0.0, "x_max": 6.4, "y_min": 0.0, "y_max": 4.57}
        heatmap_grid = self._compute_heatmap_grid(points, wall_bounds)

        spatial_data = SpatialData(heatmap_grid=heatmap_grid)

        return WallHitHeatmapResponse(
            video_id=video_id, filters=filters, data=spatial_data
        )

    def get_wall_quadrant_distribution(
        self, video_id: str, filters: AnalyticsFilters
    ) -> WallQuadrantResponse:
        """Calculate wall hit distribution across quadrants - PostgreSQL optimized.

        Returns aggregated totals if player_id filter is not specified,
        otherwise returns data for the specified player only.
        """
        logger.info(f"Computing wall quadrant distribution for video {video_id}")
        self._check_processed(video_id)

        X_CUT = 3.2
        Y_CUT = 2.285

        # Build WHERE clause WITHOUT player_id filter (we use wall_hit_player_id instead)
        where_sql, params = self._build_where_clause(
            video_id, filters, include_player_id=False
        )

        # Add wall_hit_player_id filter if specified
        if filters.player_id:
            params["wall_hit_player_id"] = filters.player_id
            where_sql += " AND wall_hit_player_id = :wall_hit_player_id"

        # Single query with CASE statements for quadrant classification
        query = text(
            f"""
            SELECT
                CASE
                    WHEN wall_hit_y_meter < {Y_CUT} THEN
                        CASE WHEN wall_hit_x_meter < {X_CUT} THEN 'Bottom-Left' ELSE 'Bottom-Right' END
                    ELSE
                        CASE WHEN wall_hit_x_meter < {X_CUT} THEN 'Top-Left' ELSE 'Top-Right' END
                END as quadrant,
                COUNT(*) as count
            FROM frame_data
            WHERE {where_sql}
              AND is_wall_hit = TRUE
              AND wall_hit_x_meter IS NOT NULL
              AND wall_hit_y_meter IS NOT NULL
            GROUP BY quadrant
            ORDER BY quadrant
        """
        )

        results = self.db.execute(query, params).fetchall()

        quadrant_counts = {
            "Bottom-Left": 0,
            "Bottom-Right": 0,
            "Top-Left": 0,
            "Top-Right": 0,
        }

        for row in results:
            quadrant_counts[row.quadrant] = row.count

        distribution = self._build_single_distribution(quadrant_counts)

        return WallQuadrantResponse(
            video_id=video_id,
            filters=filters,
            data=distribution,
            quadrant_boundaries={"x_cut": X_CUT, "y_cut": Y_CUT},
        )

    def get_winning_efficiency(
        self, video_id: str, player_id: int, filters: AnalyticsFilters
    ) -> WinningEfficiencyResponse:
        """Calculate winning efficiency for a specific player - PostgreSQL optimized.

        Returns aggregate winning efficiency metrics showing how many shots are needed to win a point.
        Lower values indicate better efficiency (winning points with fewer shots).
        """
        logger.info(f"Computing winning efficiency for player {player_id}")
        self._check_processed(video_id)

        where_sql, params = self._build_where_clause(video_id, filters)
        params["player_id"] = player_id

        # Query to get total shots per rally and points won
        query = text(
            f"""
            WITH rally_stats AS (
                SELECT
                    rally_id,
                    SUM(CASE WHEN is_racket_hit = TRUE AND racket_hit_player_id = :player_id THEN 1 ELSE 0 END) as player_shots,
                    MAX(CASE WHEN point_winner = :player_id THEN 1 ELSE 0 END) as won_point
                FROM frame_data
                WHERE {where_sql}
                  AND rally_id IS NOT NULL
                GROUP BY rally_id
            )
            SELECT
                SUM(player_shots) as total_shots,
                SUM(won_point) as points_won,
                COUNT(*) as rallies_played,
                SUM(CASE WHEN won_point = 1 THEN player_shots ELSE 0 END) as shots_in_won_rallies
            FROM rally_stats
        """
        )

        result = self.db.execute(query, params).fetchone()

        total_shots = int(result.total_shots) if result.total_shots else 0
        points_won = int(result.points_won) if result.points_won else 0
        rallies_played = int(result.rallies_played) if result.rallies_played else 0
        shots_in_won_rallies = (
            int(result.shots_in_won_rallies) if result.shots_in_won_rallies else 0
        )

        # Calculate shots needed per point won (lower is better)
        shots_per_point_won = (
            (shots_in_won_rallies / points_won) if points_won > 0 else 0.0
        )
        win_rate = (points_won / rallies_played * 100) if rallies_played > 0 else 0.0

        data = WinningEfficiencyData(
            shots_per_point_won=shots_per_point_won,
            points_won=points_won,
            total_shots=total_shots,
            win_rate=win_rate,
            rallies_played=rallies_played,
        )

        return WinningEfficiencyResponse(video_id=video_id, filters=filters, data=data)

    def get_movement_metrics(
        self, video_id: str, filters: AnalyticsFilters
    ) -> MovementMetricsResponse:
        """Calculate movement metrics - PostgreSQL optimized with self-joins.

        Returns aggregated totals if player_id filter is not specified,
        otherwise returns data for the specified player only.
        """
        logger.info(f"Computing movement metrics for video {video_id}")
        self._check_processed(video_id)

        # Build WHERE clause WITHOUT player_id filter (we handle it separately for movement)
        where_sql, params = self._build_where_clause(
            video_id, filters, include_player_id=False
        )

        # Build query based on player_id filter
        if filters.player_id:
            # Single player query
            player_col_x = f"player_{filters.player_id}_x_meter"
            player_col_y = f"player_{filters.player_id}_y_meter"
            opponent_id = 2 if filters.player_id == 1 else 1
            opponent_col_x = f"player_{opponent_id}_x_meter"
            opponent_col_y = f"player_{opponent_id}_y_meter"

            query = text(
                f"""
                WITH frame_distances AS (
                    SELECT
                        rally_id,
                        frame_number,
                        -- Player distance from previous frame
                        SQRT(
                            POW({player_col_x} - LAG({player_col_x}) OVER (PARTITION BY rally_id ORDER BY frame_number), 2) +
                            POW({player_col_y} - LAG({player_col_y}) OVER (PARTITION BY rally_id ORDER BY frame_number), 2)
                        ) as frame_distance
                    FROM frame_data
                    WHERE {where_sql}
                      AND is_rally_frame = TRUE
                      AND {player_col_x} IS NOT NULL
                      AND {player_col_y} IS NOT NULL
                ),
                rally_totals AS (
                    SELECT
                        rally_id,
                        SUM(frame_distance) as rally_distance
                    FROM frame_distances
                    GROUP BY rally_id
                )
                SELECT
                    SUM(rally_distance) as total_distance,
                    AVG(rally_distance) as avg_distance_per_rally
                FROM rally_totals
            """
            )
            params["player_id_param"] = filters.player_id
        else:
            # Aggregate both players
            query = text(
                f"""
                WITH frame_distances AS (
                    SELECT
                        rally_id,
                        frame_number,
                        -- Player 1 distance from previous frame
                        SQRT(
                            POW(player_1_x_meter - LAG(player_1_x_meter) OVER (PARTITION BY rally_id ORDER BY frame_number), 2) +
                            POW(player_1_y_meter - LAG(player_1_y_meter) OVER (PARTITION BY rally_id ORDER BY frame_number), 2)
                        ) as p1_frame_distance,
                        -- Player 2 distance from previous frame
                        SQRT(
                            POW(player_2_x_meter - LAG(player_2_x_meter) OVER (PARTITION BY rally_id ORDER BY frame_number), 2) +
                            POW(player_2_y_meter - LAG(player_2_y_meter) OVER (PARTITION BY rally_id ORDER BY frame_number), 2)
                        ) as p2_frame_distance
                    FROM frame_data
                    WHERE {where_sql}
                      AND is_rally_frame = TRUE
                      AND player_1_x_meter IS NOT NULL
                      AND player_1_y_meter IS NOT NULL
                      AND player_2_x_meter IS NOT NULL
                      AND player_2_y_meter IS NOT NULL
                ),
                rally_distances AS (
                    SELECT
                        rally_id,
                        SUM(p1_frame_distance) as p1_rally_distance,
                        SUM(p2_frame_distance) as p2_rally_distance
                    FROM frame_distances
                    GROUP BY rally_id
                )
                SELECT
                    SUM(p1_rally_distance) + SUM(p2_rally_distance) as total_distance,
                    AVG(p1_rally_distance) + AVG(p2_rally_distance) as avg_distance_per_rally
                FROM rally_distances
            """
            )

        result = self.db.execute(query, params).fetchone()

        movement_metrics = SingleMovementMetrics(
            total_distance=(
                float(result.total_distance)
                if result and result.total_distance
                else 0.0
            ),
            avg_distance_per_rally=(
                float(result.avg_distance_per_rally)
                if result and result.avg_distance_per_rally
                else 0.0
            ),
        )

        return MovementMetricsResponse(
            video_id=video_id, filters=filters, data=movement_metrics
        )

    def get_t_zone_occupancy(
        self, video_id: str, filters: AnalyticsFilters
    ) -> TZoneOccupancyResponse:
        """Calculate T-zone occupancy metrics using precomputed data.

        Returns aggregated totals if player_id filter is not specified,
        otherwise returns data for the specified player only.

        Note: Uses precomputed T-zone data from pipeline Stage 8 for performance.
        """
        logger.info(f"Computing T-zone occupancy for video {video_id}")
        self._check_processed(video_id)

        # Build WHERE clause WITHOUT player_id filter (we handle it separately)
        where_sql, params = self._build_where_clause(
            video_id, filters, include_player_id=False
        )

        # Query precomputed T-zone data
        query = text(
            f"""
            SELECT
                player_1_in_t_zone,
                player_2_in_t_zone,
                player_1_time_to_t,
                player_2_time_to_t,
                is_racket_hit,
                racket_hit_player_id
            FROM frame_data
            WHERE {where_sql}
              AND is_rally_frame = TRUE
            ORDER BY frame_number
        """
        )

        results = self.db.execute(query, params).fetchall()

        # Calculate % time in T
        total_frames = len(results)
        p1_frames_in_t = sum(1 for r in results if r.player_1_in_t_zone)
        p2_frames_in_t = sum(1 for r in results if r.player_2_in_t_zone)

        p1_pct_time_in_t = (
            (p1_frames_in_t / total_frames * 100) if total_frames > 0 else 0.0
        )
        p2_pct_time_in_t = (
            (p2_frames_in_t / total_frames * 100) if total_frames > 0 else 0.0
        )

        # Collect time-to-T measurements (already precomputed in pipeline)
        p1_time_to_t = [
            r.player_1_time_to_t for r in results if r.player_1_time_to_t is not None
        ]
        p2_time_to_t = [
            r.player_2_time_to_t for r in results if r.player_2_time_to_t is not None
        ]

        # Count total shots per player
        p1_total_shots = sum(
            1 for r in results if r.is_racket_hit and r.racket_hit_player_id == 1
        )
        p2_total_shots = sum(
            1 for r in results if r.is_racket_hit and r.racket_hit_player_id == 2
        )

        p1_successful_returns = len(p1_time_to_t)
        p2_successful_returns = len(p2_time_to_t)

        # Branch based on player_id filter
        if filters.player_id:
            # Return data for single player
            if filters.player_id == 1:
                pct_time_in_t = p1_pct_time_in_t
                time_to_t = p1_time_to_t
                total_shots = p1_total_shots
                successful_returns = p1_successful_returns
            else:
                pct_time_in_t = p2_pct_time_in_t
                time_to_t = p2_time_to_t
                total_shots = p2_total_shots
                successful_returns = p2_successful_returns

            times_array = np.array(time_to_t) if time_to_t else np.array([])

            metrics = SingleTZoneMetrics(
                pct_time_in_t=float(pct_time_in_t),
                avg_time_to_t=(
                    float(times_array.mean()) if len(times_array) > 0 else None
                ),
                min_time_to_t=(
                    float(times_array.min()) if len(times_array) > 0 else None
                ),
                max_time_to_t=(
                    float(times_array.max()) if len(times_array) > 0 else None
                ),
                time_to_t_variance=(
                    float(times_array.var()) if len(times_array) > 0 else None
                ),
                t_zone_success_rate=(
                    (successful_returns / total_shots * 100)
                    if total_shots > 0
                    else None
                ),
                total_shots_taken=total_shots,
                successful_returns=successful_returns,
            )
        else:
            # Aggregate both players
            combined_frames_in_t = p1_frames_in_t + p2_frames_in_t
            combined_total_frames = total_frames * 2  # Both players across all frames
            combined_pct_time_in_t = (
                (combined_frames_in_t / combined_total_frames * 100)
                if combined_total_frames > 0
                else 0.0
            )

            combined_time_to_t = p1_time_to_t + p2_time_to_t
            combined_total_shots = p1_total_shots + p2_total_shots
            combined_successful_returns = p1_successful_returns + p2_successful_returns

            combined_times = (
                np.array(combined_time_to_t) if combined_time_to_t else np.array([])
            )

            metrics = SingleTZoneMetrics(
                pct_time_in_t=float(combined_pct_time_in_t),
                avg_time_to_t=(
                    float(combined_times.mean()) if len(combined_times) > 0 else None
                ),
                min_time_to_t=(
                    float(combined_times.min()) if len(combined_times) > 0 else None
                ),
                max_time_to_t=(
                    float(combined_times.max()) if len(combined_times) > 0 else None
                ),
                time_to_t_variance=(
                    float(combined_times.var()) if len(combined_times) > 0 else None
                ),
                t_zone_success_rate=(
                    (combined_successful_returns / combined_total_shots * 100)
                    if combined_total_shots > 0
                    else None
                ),
                total_shots_taken=combined_total_shots,
                successful_returns=combined_successful_returns,
            )

        return TZoneOccupancyResponse(video_id=video_id, filters=filters, data=metrics)

    def get_shot_effectiveness(
        self, video_id: str, player_id: int, filters: AnalyticsFilters
    ) -> ShotEffectivenessResponse:
        """Calculate shot effectiveness metrics for a specific player - PostgreSQL optimized with self-join.

        Uses self-join to find opponent's response shot for displacement calculation.

        Args:
            video_id: Video identifier
            player_id: Player to analyze (1 or 2)
            filters: Analytics filters (rally_id, start_time, end_time)
        """
        logger.info(
            f"Computing shot effectiveness for video {video_id}, player {player_id}"
        )
        self._check_processed(video_id)

        where_sql, params = self._build_where_clause(video_id, filters)

        # Determine player and opponent columns
        opponent_id = 2 if player_id == 1 else 1
        player_col_x = f"player_{player_id}_x_meter"
        player_col_y = f"player_{player_id}_y_meter"
        opponent_col_x = f"player_{opponent_id}_x_meter"
        opponent_col_y = f"player_{opponent_id}_y_meter"

        # Define T-zone parameters (standard squash court)
        T_X = 3.05  # meters (half of 6.1m court width)
        T_Y = 5.44  # meters

        # Fetch shot frames with computed distance from T and self-join for opponent's response shot
        query = text(
            f"""
            WITH shot_frames AS (
                SELECT
                    rally_id,
                    frame_number,
                    racket_hit_player_id,
                    {player_col_x} as player_x,
                    {player_col_y} as player_y,
                    {opponent_col_x} as opponent_x,
                    {opponent_col_y} as opponent_y,
                    SQRT(POW({opponent_col_x} - {T_X}, 2) + POW({opponent_col_y} - {T_Y}, 2)) as opponent_distance_from_t,
                    shot_type
                FROM frame_data
                WHERE {where_sql}
                  AND is_racket_hit = TRUE
                  AND {player_col_x} IS NOT NULL
                  AND {opponent_col_x} IS NOT NULL
            )
            SELECT
                curr.player_x,
                curr.player_y,
                curr.opponent_x,
                curr.opponent_y,
                curr.opponent_distance_from_t,
                curr.shot_type,
                -- Get opponent's distance from T at their next shot (response shot)
                next_shot.opponent_distance_from_t as next_opponent_dist_from_t,
                -- Get opponent's position at their next shot (to calculate distance moved)
                next_shot.opponent_x as next_opponent_x,
                next_shot.opponent_y as next_opponent_y
            FROM shot_frames curr
            LEFT JOIN LATERAL (
                SELECT opponent_distance_from_t, opponent_x, opponent_y
                FROM shot_frames next
                WHERE next.rally_id = curr.rally_id
                  AND next.frame_number > curr.frame_number
                  AND next.racket_hit_player_id != curr.racket_hit_player_id
                ORDER BY next.frame_number ASC
                LIMIT 1
            ) next_shot ON TRUE
            WHERE curr.racket_hit_player_id = :player_id_param
            ORDER BY curr.frame_number
        """
        )

        params["player_id_param"] = player_id
        results = self.db.execute(query, params).fetchall()

        # Track metrics
        displacements = []
        opponent_distances_moved = []
        depth_diffs = []
        straight_shots = 0
        shots_close_to_wall = 0

        for row in results:
            # Measure opponent's displacement from T (current to next shot)
            if (
                row.opponent_distance_from_t is not None
                and row.next_opponent_dist_from_t is not None
            ):
                displacement = (
                    row.next_opponent_dist_from_t - row.opponent_distance_from_t
                )
                displacements.append(displacement)

            # Calculate absolute distance opponent moved to return shot
            if (
                row.opponent_x is not None
                and row.opponent_y is not None
                and row.next_opponent_x is not None
                and row.next_opponent_y is not None
            ):
                distance_moved = np.sqrt(
                    (row.next_opponent_x - row.opponent_x) ** 2
                    + (row.next_opponent_y - row.opponent_y) ** 2
                )
                opponent_distances_moved.append(distance_moved)

            # Depth dominance
            depth_diff = row.opponent_y - row.player_y
            depth_diffs.append(depth_diff)

            # Straight shot quality
            if row.shot_type in ["straight_drive", "straight_drop"]:
                straight_shots += 1
                dist_to_wall = min(row.player_x, 6.1 - row.player_x)
                if dist_to_wall <= 1.2:
                    shots_close_to_wall += 1

        # Calculate aggregates (vectorized)
        displ = np.array(displacements) if displacements else np.array([])
        opp_dist = np.array(opponent_distances_moved) if opponent_distances_moved else np.array([])
        depth = np.array(depth_diffs) if depth_diffs else np.array([])

        metrics = SingleShotEffectivenessMetrics(
            avg_displacement_from_t=float(displ.mean()) if len(displ) > 0 else None,
            max_displacement_from_t=float(displ.max()) if len(displ) > 0 else None,
            displacement_variance=float(displ.var()) if len(displ) > 0 else None,
            avg_opponent_distance_moved=float(opp_dist.mean()) if len(opp_dist) > 0 else None,
            max_opponent_distance_moved=float(opp_dist.max()) if len(opp_dist) > 0 else None,
            opponent_distance_moved_variance=float(opp_dist.var()) if len(opp_dist) > 0 else None,
            depth_dominance_pct=(
                (np.sum(depth > 0) / len(depth) * 100) if len(depth) > 0 else None
            ),
            avg_depth_difference=float(depth.mean()) if len(depth) > 0 else None,
            min_depth_difference=float(depth.min()) if len(depth) > 0 else None,
            max_depth_difference=float(depth.max()) if len(depth) > 0 else None,
            straight_shot_quality_pct=(
                (shots_close_to_wall / straight_shots * 100)
                if straight_shots > 0
                else None
            ),
            straight_shots_count=straight_shots,
            shots_close_to_wall=shots_close_to_wall,
        )

        return ShotEffectivenessResponse(
            video_id=video_id, filters=filters, data=metrics
        )

    def get_rally_intensity(
        self, video_id: str, filters: AnalyticsFilters
    ) -> RallyIntensityResponse:
        """Calculate rally intensity and pace metrics - PostgreSQL optimized.

        Returns aggregate intensity metrics (seconds per shot).
        Lower seconds per shot indicates faster/more intense play.
        """
        logger.info(f"Computing rally intensity for video {video_id}")
        self._check_processed(video_id)

        where_sql, params = self._build_where_clause(video_id, filters)

        # Aggregate directly in SQL - calculate seconds per shot across all rallies
        query = text(
            f"""
            WITH rally_metrics AS (
                SELECT
                    rally_id,
                    (MAX(timestamp) - MIN(timestamp)) / NULLIF(SUM(CASE WHEN is_racket_hit = TRUE THEN 1 ELSE 0 END), 0) as sec_per_shot
                FROM frame_data
                WHERE {where_sql}
                  AND rally_id IS NOT NULL
                GROUP BY rally_id
                HAVING SUM(CASE WHEN is_racket_hit = TRUE THEN 1 ELSE 0 END) > 0
            )
            SELECT
                AVG(sec_per_shot) as avg_sec_per_shot,
                MIN(sec_per_shot) as min_sec_per_shot,
                MAX(sec_per_shot) as max_sec_per_shot,
                STDDEV_POP(sec_per_shot) as std_dev,
                COUNT(*) as rally_count
            FROM rally_metrics
        """
        )

        result = self.db.execute(query, params).fetchone()

        data = RallyIntensityData(
            avg_seconds_per_shot=(
                float(result.avg_sec_per_shot) if result.avg_sec_per_shot else 0.0
            ),
            min_seconds_per_shot=(
                float(result.min_sec_per_shot) if result.min_sec_per_shot else 0.0
            ),
            max_seconds_per_shot=(
                float(result.max_sec_per_shot) if result.max_sec_per_shot else 0.0
            ),
            std_dev=float(result.std_dev) if result.std_dev else 0.0,
            rally_count=int(result.rally_count) if result.rally_count else 0,
        )

        return RallyIntensityResponse(video_id=video_id, filters=filters, data=data)

    def get_let_stats(
        self, video_id: str, filters: AnalyticsFilters
    ) -> LetStatsResponse:
        """Get let/replay statistics.

        Counts rallies where point_winner = 0 (indicates a let).

        Args:
            video_id: Video identifier
            filters: Analytics filters

        Returns:
            LetStatsResponse with let statistics
        """
        logger.info(f"Computing let stats for video {video_id}")
        self._check_processed(video_id)

        where_sql, params = self._build_where_clause(
            video_id, filters, include_player_id=False
        )

        query = text(
            f"""
            SELECT DISTINCT
                rally_id,
                MAX(point_winner) as point_winner
            FROM frame_data
            WHERE {where_sql}
              AND rally_id IS NOT NULL
            GROUP BY rally_id
        """
        )

        results = self.db.execute(query, params).fetchall()

        total_rallies = len(results)
        total_lets = sum(1 for row in results if row.point_winner == 0)
        let_percentage = (total_lets / total_rallies * 100) if total_rallies > 0 else 0.0

        data = LetStatsData(
            total_lets=total_lets,
            total_rallies=total_rallies,
            let_percentage=let_percentage,
        )

        return LetStatsResponse(video_id=video_id, filters=filters, data=data)

    def get_break_time(
        self, video_id: str, filters: AnalyticsFilters
    ) -> BreakTimeResponse:
        """Get break time statistics between rallies.

        Calculates time between end of one rally and start of next rally.

        Args:
            video_id: Video identifier
            filters: Analytics filters

        Returns:
            BreakTimeResponse with break time statistics
        """
        logger.info(f"Computing break time stats for video {video_id}")
        self._check_processed(video_id)

        where_sql, params = self._build_where_clause(
            video_id, filters, include_player_id=False
        )

        query = text(
            f"""
            WITH rally_times AS (
                SELECT DISTINCT
                    rally_id,
                    MIN(timestamp) as rally_start,
                    MAX(timestamp) as rally_end
                FROM frame_data
                WHERE {where_sql}
                  AND rally_id IS NOT NULL
                GROUP BY rally_id
                ORDER BY rally_id
            ),
            break_times AS (
                SELECT
                    curr.rally_id,
                    curr.rally_start - prev.rally_end as break_time
                FROM rally_times curr
                LEFT JOIN LATERAL (
                    SELECT rally_end
                    FROM rally_times prev_rally
                    WHERE prev_rally.rally_id < curr.rally_id
                    ORDER BY prev_rally.rally_id DESC
                    LIMIT 1
                ) prev ON TRUE
                WHERE prev.rally_end IS NOT NULL
            )
            SELECT
                AVG(break_time) as avg_break_time,
                MIN(break_time) as min_break_time,
                MAX(break_time) as max_break_time,
                STDDEV_POP(break_time) as std_dev,
                COUNT(*) as total_breaks
            FROM break_times
        """
        )

        result = self.db.execute(query, params).fetchone()

        data = BreakTimeData(
            avg_break_time=(
                float(result.avg_break_time) if result.avg_break_time else 0.0
            ),
            min_break_time=(
                float(result.min_break_time) if result.min_break_time else 0.0
            ),
            max_break_time=(
                float(result.max_break_time) if result.max_break_time else 0.0
            ),
            std_dev=float(result.std_dev) if result.std_dev else 0.0,
            total_breaks=int(result.total_breaks) if result.total_breaks else 0,
        )

        return BreakTimeResponse(video_id=video_id, filters=filters, data=data)

    def get_match_summary(self, video_id: str) -> MatchSummaryResponse:
        """Get match summary including game results and overall match winner.

        Args:
            video_id: Video identifier

        Returns:
            MatchSummaryResponse with match and game details

        Raises:
            ValueError: If match data not found
        """
        logger.info(f"Getting match summary for video {video_id}")

        # Get match result
        match = self.db.query(Match).filter(Match.video_id == video_id).first()
        if not match:
            raise ValueError(
                f"Match data not found for video {video_id}. Video may not be processed yet."
            )

        # Get all games for this match
        games = (
            self.db.query(Game)
            .filter(Game.video_id == video_id)
            .order_by(Game.game_number)
            .all()
        )

        return MatchSummaryResponse(video_id=video_id, match=match, games=games)

    def get_rally_timeline(
        self, video_id: str, filters: AnalyticsFilters
    ) -> RallyTimelineResponse:
        """Get rally-by-rally timeline with key metrics.

        Returns chronological sequence of rallies with duration, shots, speed, etc.
        Supports filtering by game_number, timestamp range.

        Args:
            video_id: Video identifier
            filters: Analytics filters

        Returns:
            RallyTimelineResponse with rally metrics

        Raises:
            ValueError: If video not processed or invalid filters
        """
        logger.info(f"Computing rally timeline for video {video_id}")
        self._check_processed(video_id)

        where_sql, params = self._build_where_clause(
            video_id, filters, include_player_id=False
        )

        query = text(
            f"""
            WITH rally_aggregates AS (
                SELECT
                    rally_id,
                    MIN(timestamp) as rally_start_time,
                    MAX(timestamp) - MIN(timestamp) as rally_duration,
                    COUNT(CASE WHEN is_racket_hit = TRUE THEN 1 END) as shot_count,
                    MAX(point_winner) as point_winner,
                    COUNT(CASE WHEN is_wall_hit = TRUE THEN 1 END) as wall_hit_count
                FROM frame_data
                WHERE {where_sql} AND rally_id IS NOT NULL
                GROUP BY rally_id
                HAVING COUNT(CASE WHEN is_racket_hit = TRUE THEN 1 END) > 0
            )
            SELECT * FROM rally_aggregates ORDER BY rally_id
        """
        )

        results = self.db.execute(query, params).fetchall()

        timeline_items = [
            RallyTimelineItem(
                rally_id=row.rally_id,
                rally_start_time=float(row.rally_start_time),
                rally_duration=float(row.rally_duration),
                shot_count=int(row.shot_count),
                point_winner=int(row.point_winner) if row.point_winner else None,
                wall_hit_count=int(row.wall_hit_count),
            )
            for row in results
        ]

        return RallyTimelineResponse(
            video_id=video_id,
            filters=filters,
            data=timeline_items,
            total_rallies=len(timeline_items),
        )

    def get_momentum_timeline(
        self, video_id: str, filters: AnalyticsFilters
    ) -> MomentumTimelineResponse:
        """Get cumulative score progression and momentum shifts.

        Returns running scoreboard showing cumulative scores after each rally.
        When player_id filter is set, only that player's wins increment their score.

        Args:
            video_id: Video identifier
            filters: Analytics filters

        Returns:
            MomentumTimelineResponse with cumulative scores

        Raises:
            ValueError: If video not processed or invalid filters
        """
        logger.info(f"Computing momentum timeline for video {video_id}")
        self._check_processed(video_id)

        where_sql, params = self._build_where_clause(
            video_id, filters, include_player_id=False
        )

        # Build query with window functions for cumulative scores
        query = text(
            f"""
            WITH rally_outcomes AS (
                SELECT
                    rally_id,
                    MIN(timestamp) as timestamp,
                    MAX(point_winner) as point_winner
                FROM frame_data
                WHERE {where_sql} AND is_rally_frame = TRUE
                GROUP BY rally_id
                ORDER BY rally_id
            ),
            cumulative_score AS (
                SELECT
                    rally_id,
                    timestamp,
                    point_winner,
                    SUM(CASE WHEN point_winner = 1 THEN 1 ELSE 0 END)
                        OVER (ORDER BY rally_id ROWS UNBOUNDED PRECEDING) as player_1_score,
                    SUM(CASE WHEN point_winner = 2 THEN 1 ELSE 0 END)
                        OVER (ORDER BY rally_id ROWS UNBOUNDED PRECEDING) as player_2_score
                FROM rally_outcomes
            )
            SELECT
                rally_id,
                timestamp,
                point_winner,
                player_1_score,
                player_2_score,
                (player_1_score - player_2_score) as score_differential
            FROM cumulative_score
            ORDER BY rally_id
        """
        )

        results = self.db.execute(query, params).fetchall()

        timeline_items = [
            MomentumTimelineItem(
                rally_id=row.rally_id,
                timestamp=float(row.timestamp),
                point_winner=int(row.point_winner) if row.point_winner else None,
                player_1_score=int(row.player_1_score),
                player_2_score=int(row.player_2_score),
                score_differential=int(row.score_differential),
            )
            for row in results
        ]

        return MomentumTimelineResponse(
            video_id=video_id, filters=filters, data=timeline_items
        )

    # ========================================================================
    # PER-GAME TIME-SERIES ANALYTICS METHODS
    # ========================================================================

    def get_ball_speed_per_game(
        self, video_id: str, filters: AnalyticsFilters
    ) -> BallSpeedPerGameResponse:
        """Get ball speed statistics per game with both players' data.

        Returns per-game breakdown of ball speed metrics for both players.

        Args:
            video_id: Video identifier
            filters: Analytics filters (game_number, time range)

        Returns:
            BallSpeedPerGameResponse with per-game data for both players
        """
        logger.info(f"Computing ball speed per-game for video {video_id}")
        self._check_processed(video_id)

        # Get game metadata
        games_metadata = self._get_games_metadata(video_id, filters)

        if not games_metadata:
            return BallSpeedPerGameResponse(
                video_id=video_id, filters=filters, data=[], total_games=0
            )

        # Build WHERE clause
        where_sql, params = self._build_where_clause(
            video_id, filters, include_player_id=False
        )

        # Query per-game ball speed grouped by game and player
        query = text(
            f"""
            WITH game_player_speeds AS (
                SELECT
                    g.game_number,
                    f.racket_hit_player_id,
                    AVG(f.ball_speed) as mean_speed,
                    MIN(f.ball_speed) as min_speed,
                    MAX(f.ball_speed) as max_speed,
                    STDDEV_POP(f.ball_speed) as std_dev,
                    COUNT(*) as shot_count
                FROM games g
                JOIN frame_data f ON f.video_id = g.video_id
                    AND f.rally_id BETWEEN g.start_rally_id AND g.end_rally_id
                WHERE g.{where_sql}
                  AND f.is_racket_hit = TRUE
                  AND f.ball_speed IS NOT NULL
                GROUP BY g.game_number, f.racket_hit_player_id
                ORDER BY g.game_number, f.racket_hit_player_id
            )
            SELECT * FROM game_player_speeds
        """
        )

        results = self.db.execute(query, params).fetchall()

        # Define metric builder function
        def build_ball_speed_data(row):
            return BallSpeedData(
                mean_speed=float(row.mean_speed) if row.mean_speed else 0.0,
                min_speed=float(row.min_speed) if row.min_speed else 0.0,
                max_speed=float(row.max_speed) if row.max_speed else 0.0,
                std_dev=float(row.std_dev) if row.std_dev else 0.0,
                shot_count=int(row.shot_count) if row.shot_count else 0,
            )

        # Pivot data
        game_items_data = self._pivot_by_game(
            results, games_metadata, build_ball_speed_data
        )

        # Create empty metrics for missing players
        empty_ball_speed = BallSpeedData(
            mean_speed=0.0,
            min_speed=0.0,
            max_speed=0.0,
            std_dev=0.0,
            shot_count=0,
        )

        # Convert to response items, filling in empty metrics for missing players
        game_items = []
        for item in game_items_data:
            game_items.append(
                BallSpeedPerGameItem(
                    game_number=item["game_number"],
                    start_rally_id=item["start_rally_id"],
                    end_rally_id=item["end_rally_id"],
                    start_time=item["start_time"],
                    end_time=item["end_time"],
                    duration=item["duration"],
                    rally_count=item["rally_count"],
                    player_1=(
                        item["player_1"]
                        if item["player_1"] is not None
                        else empty_ball_speed
                    ),
                    player_2=(
                        item["player_2"]
                        if item["player_2"] is not None
                        else empty_ball_speed
                    ),
                )
            )

        return BallSpeedPerGameResponse(
            video_id=video_id,
            filters=filters,
            data=game_items,
            total_games=len(game_items),
        )

    # ========================================================================
    # PER-RALLY TIME-SERIES ANALYTICS METHODS
    # ========================================================================

    def get_ball_speed_per_rally(
        self, video_id: str, filters: AnalyticsFilters
    ) -> BallSpeedPerRallyResponse:
        """Get ball speed statistics per rally with both players' data.

        Returns per-rally breakdown of ball speed metrics for both players.

        Args:
            video_id: Video identifier
            filters: Analytics filters (game_number, time range)

        Returns:
            BallSpeedPerRallyResponse with per-rally data for both players
        """
        logger.info(f"Computing ball speed per-rally for video {video_id}")
        self._check_processed(video_id)

        # Build WHERE clause
        where_sql, params = self._build_where_clause(
            video_id, filters, include_player_id=False
        )

        # Query per-rally ball speed grouped by rally and player
        query = text(
            f"""
            WITH rally_metadata AS (
                SELECT DISTINCT
                    rally_id,
                    MIN(timestamp) as rally_start_time,
                    MAX(timestamp) - MIN(timestamp) as rally_duration,
                    COUNT(CASE WHEN is_racket_hit = TRUE THEN 1 END) as shot_count,
                    MAX(point_winner) as point_winner
                FROM frame_data
                WHERE {where_sql}
                  AND rally_id IS NOT NULL
                GROUP BY rally_id
            ),
            rally_game_mapping AS (
                SELECT f.rally_id, MIN(g.game_number) as game_number
                FROM frame_data f
                JOIN games g ON f.video_id = g.video_id
                    AND f.rally_id BETWEEN g.start_rally_id AND g.end_rally_id
                WHERE f.video_id = :video_id
                GROUP BY f.rally_id
            ),
            rally_player_speeds AS (
                SELECT
                    f.rally_id,
                    f.racket_hit_player_id,
                    AVG(f.ball_speed) as mean_speed,
                    MIN(f.ball_speed) as min_speed,
                    MAX(f.ball_speed) as max_speed,
                    STDDEV_POP(f.ball_speed) as std_dev,
                    COUNT(*) as player_shot_count
                FROM frame_data f
                WHERE {where_sql}
                  AND f.rally_id IS NOT NULL
                  AND f.is_racket_hit = TRUE
                  AND f.ball_speed IS NOT NULL
                GROUP BY f.rally_id, f.racket_hit_player_id
            )
            SELECT
                rm.rally_id,
                rm.rally_start_time,
                rm.rally_duration,
                rm.shot_count,
                rm.point_winner,
                rgm.game_number,
                rps.racket_hit_player_id,
                rps.mean_speed,
                rps.min_speed,
                rps.max_speed,
                rps.std_dev,
                rps.player_shot_count
            FROM rally_metadata rm
            LEFT JOIN rally_game_mapping rgm ON rm.rally_id = rgm.rally_id
            LEFT JOIN rally_player_speeds rps ON rm.rally_id = rps.rally_id
            WHERE rps.racket_hit_player_id IS NOT NULL
            ORDER BY rm.rally_id, rps.racket_hit_player_id
        """
        )

        results = self.db.execute(query, params).fetchall()

        # Define metric builder function
        def build_ball_speed_data(row):
            return BallSpeedData(
                mean_speed=float(row.mean_speed) if row.mean_speed else 0.0,
                min_speed=float(row.min_speed) if row.min_speed else 0.0,
                max_speed=float(row.max_speed) if row.max_speed else 0.0,
                std_dev=float(row.std_dev) if row.std_dev else 0.0,
                shot_count=int(row.player_shot_count) if row.player_shot_count else 0,
            )

        # Pivot data
        rally_items_data = self._pivot_by_rally(results, build_ball_speed_data)

        # Create empty metrics for missing players
        empty_ball_speed = BallSpeedData(
            mean_speed=0.0,
            min_speed=0.0,
            max_speed=0.0,
            std_dev=0.0,
            shot_count=0,
        )

        # Convert to response items, filling in empty metrics for missing players
        rally_items = []
        for item in rally_items_data:
            # Convert -1 point_winner to None (indicates unknown/not set)
            point_winner = (
                item["point_winner"] if item["point_winner"] not in [-1, None] else None
            )

            rally_items.append(
                BallSpeedPerRallyItem(
                    rally_id=item["rally_id"],
                    game_number=item["game_number"],
                    rally_start_time=item["rally_start_time"],
                    rally_duration=item["rally_duration"],
                    shot_count=item["shot_count"],
                    point_winner=point_winner,
                    player_1=(
                        item["player_1"]
                        if item["player_1"] is not None
                        else empty_ball_speed
                    ),
                    player_2=(
                        item["player_2"]
                        if item["player_2"] is not None
                        else empty_ball_speed
                    ),
                )
            )

        return BallSpeedPerRallyResponse(
            video_id=video_id,
            filters=filters,
            data=rally_items,
            total_rallies=len(rally_items),
        )

    def get_stroke_distribution_per_game(
        self, video_id: str, filters: AnalyticsFilters
    ) -> StrokeDistributionPerGameResponse:
        """Get stroke distribution per game with both players' data - Pure SQL."""
        logger.info(f"Computing stroke distribution per-game for video {video_id}")
        self._check_processed(video_id)

        games_metadata = self._get_games_metadata(video_id, filters)
        if not games_metadata:
            return StrokeDistributionPerGameResponse(
                video_id=video_id, filters=filters, data=[], total_games=0
            )

        where_sql, params = self._build_where_clause(
            video_id, filters, include_player_id=False
        )

        # Pure SQL query - aggregates stroke counts by game and player
        query = text(
            f"""
            WITH game_player_strokes AS (
                SELECT
                    g.game_number,
                    f.racket_hit_player_id as player_id,
                    f.stroke_type,
                    COUNT(*) as stroke_count
                FROM games g
                JOIN frame_data f ON f.video_id = g.video_id
                    AND f.rally_id BETWEEN g.start_rally_id AND g.end_rally_id
                WHERE g.{where_sql}
                  AND f.is_racket_hit = TRUE
                  AND f.stroke_type IS NOT NULL
                  AND f.racket_hit_player_id IN (1, 2)
                GROUP BY g.game_number, f.racket_hit_player_id, f.stroke_type
            ),
            player_totals AS (
                SELECT
                    game_number,
                    player_id,
                    SUM(stroke_count) as total_strokes
                FROM game_player_strokes
                GROUP BY game_number, player_id
            )
            SELECT
                gps.game_number,
                gps.player_id,
                gps.stroke_type,
                gps.stroke_count,
                pt.total_strokes,
                CASE
                    WHEN pt.total_strokes > 0
                    THEN (gps.stroke_count::float / pt.total_strokes * 100)
                    ELSE 0
                END as percentage
            FROM game_player_strokes gps
            JOIN player_totals pt ON gps.game_number = pt.game_number
                AND gps.player_id = pt.player_id
            ORDER BY gps.game_number, gps.player_id, gps.stroke_type
        """
        )

        results = self.db.execute(query, params).fetchall()

        # Build game items from SQL results
        games_data = {}
        for row in results:
            game_num = row.game_number
            player_id = row.player_id

            if game_num not in games_data:
                games_data[game_num] = {
                    "player_1": [],
                    "player_2": [],
                    "total_1": 0,
                    "total_2": 0,
                }

            dist_item = DistributionItem(
                label=row.stroke_type,
                count=int(row.stroke_count),
                percentage=float(row.percentage),
            )

            if player_id == 1:
                games_data[game_num]["player_1"].append(dist_item)
                games_data[game_num]["total_1"] = int(row.total_strokes)
            else:
                games_data[game_num]["player_2"].append(dist_item)
                games_data[game_num]["total_2"] = int(row.total_strokes)

        # Combine with game metadata
        result = []
        for game_meta in games_metadata:
            game_num = game_meta["game_number"]
            game_item = {**game_meta}

            if game_num in games_data:
                game_item["player_1"] = SingleDistribution(
                    distribution=games_data[game_num]["player_1"],
                    total=games_data[game_num]["total_1"],
                )
                game_item["player_2"] = SingleDistribution(
                    distribution=games_data[game_num]["player_2"],
                    total=games_data[game_num]["total_2"],
                )
            else:
                game_item["player_1"] = SingleDistribution(distribution=[], total=0)
                game_item["player_2"] = SingleDistribution(distribution=[], total=0)

            result.append(StrokeDistributionPerGameItem(**game_item))

        return StrokeDistributionPerGameResponse(
            video_id=video_id,
            filters=filters,
            data=result,
            total_games=len(result),
        )

    def get_stroke_distribution_per_rally(
        self, video_id: str, filters: AnalyticsFilters
    ) -> StrokeDistributionPerRallyResponse:
        """Get stroke distribution per rally with both players' data - Pure SQL."""
        logger.info(f"Computing stroke distribution per-rally for video {video_id}")
        self._check_processed(video_id)

        where_sql, params = self._build_where_clause(
            video_id, filters, include_player_id=False
        )

        # Pure SQL query - aggregates stroke counts by rally and player
        query = text(
            f"""
            WITH rally_metadata AS (
                SELECT DISTINCT
                    rally_id,
                    MIN(timestamp) as rally_start_time,
                    MAX(timestamp) - MIN(timestamp) as rally_duration,
                    COUNT(CASE WHEN is_racket_hit = TRUE THEN 1 END) as shot_count,
                    MAX(point_winner) as point_winner
                FROM frame_data
                WHERE {where_sql}
                  AND rally_id IS NOT NULL
                GROUP BY rally_id
            ),
            rally_game_mapping AS (
                SELECT f.rally_id, MIN(g.game_number) as game_number
                FROM frame_data f
                JOIN games g ON f.video_id = g.video_id
                    AND f.rally_id BETWEEN g.start_rally_id AND g.end_rally_id
                WHERE f.video_id = :video_id
                GROUP BY f.rally_id
            ),
            rally_player_strokes AS (
                SELECT
                    f.rally_id,
                    f.racket_hit_player_id as player_id,
                    f.stroke_type,
                    COUNT(*) as stroke_count
                FROM frame_data f
                WHERE {where_sql}
                  AND f.rally_id IS NOT NULL
                  AND f.is_racket_hit = TRUE
                  AND f.stroke_type IS NOT NULL
                  AND f.racket_hit_player_id IN (1, 2)
                GROUP BY f.rally_id, f.racket_hit_player_id, f.stroke_type
            ),
            player_totals AS (
                SELECT
                    rally_id,
                    player_id,
                    SUM(stroke_count) as total_strokes
                FROM rally_player_strokes
                GROUP BY rally_id, player_id
            )
            SELECT
                rm.rally_id,
                rm.rally_start_time,
                rm.rally_duration,
                rm.shot_count,
                rm.point_winner,
                rgm.game_number,
                rps.player_id,
                rps.stroke_type,
                rps.stroke_count,
                pt.total_strokes,
                CASE
                    WHEN pt.total_strokes > 0
                    THEN (rps.stroke_count::float / pt.total_strokes * 100)
                    ELSE 0
                END as percentage
            FROM rally_metadata rm
            LEFT JOIN rally_game_mapping rgm ON rm.rally_id = rgm.rally_id
            LEFT JOIN rally_player_strokes rps ON rm.rally_id = rps.rally_id
            LEFT JOIN player_totals pt ON rps.rally_id = pt.rally_id
                AND rps.player_id = pt.player_id
            WHERE rps.player_id IS NOT NULL
            ORDER BY rm.rally_id, rps.player_id, rps.stroke_type
        """
        )

        results = self.db.execute(query, params).fetchall()

        # Build rally items from SQL results
        rallies_data = {}
        rally_metadata = {}

        for row in results:
            rally_id = row.rally_id

            if rally_id not in rally_metadata:
                rally_metadata[rally_id] = {
                    "rally_id": rally_id,
                    "game_number": row.game_number,
                    "rally_start_time": float(row.rally_start_time),
                    "rally_duration": float(row.rally_duration),
                    "shot_count": int(row.shot_count),
                    "point_winner": int(row.point_winner) if row.point_winner else None,
                }

            if rally_id not in rallies_data:
                rallies_data[rally_id] = {
                    "player_1": [],
                    "player_2": [],
                    "total_1": 0,
                    "total_2": 0,
                }

            player_id = row.player_id
            dist_item = DistributionItem(
                label=row.stroke_type,
                count=int(row.stroke_count),
                percentage=float(row.percentage),
            )

            if player_id == 1:
                rallies_data[rally_id]["player_1"].append(dist_item)
                rallies_data[rally_id]["total_1"] = int(row.total_strokes)
            else:
                rallies_data[rally_id]["player_2"].append(dist_item)
                rallies_data[rally_id]["total_2"] = int(row.total_strokes)

        # Build result
        result = []
        for rally_id, meta in rally_metadata.items():
            rally_item = {**meta}

            if rally_id in rallies_data:
                rally_item["player_1"] = SingleDistribution(
                    distribution=rallies_data[rally_id]["player_1"],
                    total=rallies_data[rally_id]["total_1"],
                )
                rally_item["player_2"] = SingleDistribution(
                    distribution=rallies_data[rally_id]["player_2"],
                    total=rallies_data[rally_id]["total_2"],
                )
            else:
                rally_item["player_1"] = SingleDistribution(distribution=[], total=0)
                rally_item["player_2"] = SingleDistribution(distribution=[], total=0)

            result.append(StrokeDistributionPerRallyItem(**rally_item))

        return StrokeDistributionPerRallyResponse(
            video_id=video_id,
            filters=filters,
            data=result,
            total_rallies=len(result),
        )

    def get_shot_type_distribution_per_game(
        self, video_id: str, filters: AnalyticsFilters
    ) -> ShotTypeDistributionPerGameResponse:
        """Get shot type distribution per game with both players' data - Pure SQL."""
        games_metadata = self._get_games_metadata(video_id, filters)
        if not games_metadata:
            return ShotTypeDistributionPerGameResponse(
                video_id=video_id,
                filters=filters,
                data=[],
                total_games=0,
                all_shot_types=[],
            )

        where_sql, params = self._build_where_clause(video_id, filters)
        params["video_id"] = video_id

        query = text(
            f"""
            WITH game_player_shot_types AS (
                SELECT
                    g.game_number,
                    f.racket_hit_player_id as player_id,
                    f.shot_type,
                    COUNT(*) as shot_count
                FROM games g
                JOIN frame_data f ON f.video_id = g.video_id
                    AND f.rally_id BETWEEN g.start_rally_id AND g.end_rally_id
                WHERE g.{where_sql}
                  AND f.is_racket_hit = TRUE
                  AND f.shot_type IS NOT NULL
                  AND f.racket_hit_player_id IN (1, 2)
                GROUP BY g.game_number, f.racket_hit_player_id, f.shot_type
            ),
            player_totals AS (
                SELECT
                    game_number,
                    player_id,
                    SUM(shot_count) as total_shots
                FROM game_player_shot_types
                GROUP BY game_number, player_id
            )
            SELECT
                gpst.game_number,
                gpst.player_id,
                gpst.shot_type,
                gpst.shot_count,
                pt.total_shots,
                CASE
                    WHEN pt.total_shots > 0
                    THEN (gpst.shot_count::float / pt.total_shots * 100)
                    ELSE 0
                END as percentage
            FROM game_player_shot_types gpst
            JOIN player_totals pt ON gpst.game_number = pt.game_number
                AND gpst.player_id = pt.player_id
            ORDER BY gpst.game_number, gpst.player_id, gpst.shot_type
        """
        )

        results = self.db.execute(query, params).fetchall()

        # Get all shot types from separate query for response
        all_types_query = text(
            f"""
            SELECT DISTINCT shot_type
            FROM frame_data
            WHERE {where_sql}
              AND shot_type IS NOT NULL
              AND is_racket_hit = TRUE
            ORDER BY shot_type
        """
        )
        all_shot_types = [
            row[0] for row in self.db.execute(all_types_query, params).fetchall()
        ]

        # Build distribution data per game and player
        def build_shot_type_dist(rows):
            distribution = []
            total = 0
            for r in rows:
                shot_type = r.shot_type
                count = r.shot_count
                percentage = r.percentage
                distribution.append(
                    DistributionItem(
                        label=shot_type, count=count, percentage=percentage
                    )
                )
                total += count
            return SingleDistribution(distribution=distribution, total=total)

        # Organize by game -> player
        game_player_rows = {}
        for row in results:
            game_num = row.game_number
            player_id = row.player_id
            if game_num not in game_player_rows:
                game_player_rows[game_num] = {1: [], 2: []}
            game_player_rows[game_num][player_id].append(row)

        # Build final response
        game_items = []
        for game_meta in games_metadata:
            game_num = game_meta["game_number"]
            player_data = game_player_rows.get(game_num, {1: [], 2: []})

            p1_dist = build_shot_type_dist(player_data.get(1, []))
            p2_dist = build_shot_type_dist(player_data.get(2, []))

            game_items.append(
                ShotTypeDistributionPerGameItem(
                    game_number=game_num,
                    start_rally_id=game_meta["start_rally_id"],
                    end_rally_id=game_meta["end_rally_id"],
                    start_time=game_meta["start_time"],
                    end_time=game_meta["end_time"],
                    duration=game_meta["duration"],
                    rally_count=game_meta["rally_count"],
                    player_1=p1_dist,
                    player_2=p2_dist,
                )
            )

        return ShotTypeDistributionPerGameResponse(
            video_id=video_id,
            filters=filters,
            data=game_items,
            total_games=len(game_items),
            all_shot_types=all_shot_types,
        )

    def get_shot_type_distribution_per_rally(
        self, video_id: str, filters: AnalyticsFilters
    ) -> ShotTypeDistributionPerRallyResponse:
        """Get shot type distribution per rally with both players' data - Pure SQL."""
        where_sql, params = self._build_where_clause(video_id, filters)
        params["video_id"] = video_id

        query = text(
            f"""
            WITH rally_metadata AS (
                SELECT DISTINCT
                    rally_id,
                    MIN(timestamp) as rally_start_time,
                    MAX(timestamp) - MIN(timestamp) as rally_duration,
                    COUNT(CASE WHEN is_racket_hit = TRUE THEN 1 END) as shot_count,
                    MAX(point_winner) as point_winner
                FROM frame_data
                WHERE {where_sql} AND rally_id IS NOT NULL
                GROUP BY rally_id
            ),
            rally_game_mapping AS (
                SELECT f.rally_id, MIN(g.game_number) as game_number
                FROM frame_data f
                JOIN games g ON f.video_id = g.video_id
                    AND f.rally_id BETWEEN g.start_rally_id AND g.end_rally_id
                WHERE f.video_id = :video_id
                GROUP BY f.rally_id
            ),
            rally_player_shot_types AS (
                SELECT
                    f.rally_id,
                    f.racket_hit_player_id as player_id,
                    f.shot_type,
                    COUNT(*) as shot_count
                FROM frame_data f
                WHERE {where_sql}
                  AND f.rally_id IS NOT NULL
                  AND f.is_racket_hit = TRUE
                  AND f.shot_type IS NOT NULL
                  AND f.racket_hit_player_id IN (1, 2)
                GROUP BY f.rally_id, f.racket_hit_player_id, f.shot_type
            ),
            player_totals AS (
                SELECT rally_id, player_id, SUM(shot_count) as total_shots
                FROM rally_player_shot_types
                GROUP BY rally_id, player_id
            )
            SELECT
                rm.rally_id,
                rm.rally_start_time,
                rm.rally_duration,
                rm.shot_count,
                rm.point_winner,
                rgm.game_number,
                rpst.player_id,
                rpst.shot_type,
                rpst.shot_count as type_count,
                pt.total_shots,
                CASE
                    WHEN pt.total_shots > 0
                    THEN (rpst.shot_count::float / pt.total_shots * 100)
                    ELSE 0
                END as percentage
            FROM rally_metadata rm
            LEFT JOIN rally_game_mapping rgm ON rm.rally_id = rgm.rally_id
            LEFT JOIN rally_player_shot_types rpst ON rm.rally_id = rpst.rally_id
            LEFT JOIN player_totals pt ON rpst.rally_id = pt.rally_id
                AND rpst.player_id = pt.player_id
            WHERE rpst.player_id IS NOT NULL
            ORDER BY rm.rally_id, rpst.player_id, rpst.shot_type
        """
        )

        results = self.db.execute(query, params).fetchall()

        # Get all shot types from separate query
        all_types_query = text(
            f"""
            SELECT DISTINCT shot_type
            FROM frame_data
            WHERE {where_sql}
              AND shot_type IS NOT NULL
              AND is_racket_hit = TRUE
            ORDER BY shot_type
        """
        )
        all_shot_types = [
            row[0] for row in self.db.execute(all_types_query, params).fetchall()
        ]

        # Build distribution data per rally
        def build_shot_type_dist(rows):
            distribution = []
            total = 0
            for r in rows:
                shot_type = r.shot_type
                count = r.type_count
                percentage = r.percentage
                distribution.append(
                    DistributionItem(
                        label=shot_type, count=count, percentage=percentage
                    )
                )
                total += count
            return SingleDistribution(distribution=distribution, total=total)

        # Organize by rally -> player
        rally_player_rows = {}
        rally_metadata_map = {}
        for row in results:
            rally_id = row.rally_id
            player_id = row.player_id

            if rally_id not in rally_metadata_map:
                rally_metadata_map[rally_id] = {
                    "rally_start_time": row.rally_start_time,
                    "rally_duration": row.rally_duration,
                    "shot_count": row.shot_count,
                    "point_winner": row.point_winner,
                    "game_number": row.game_number,
                }

            if rally_id not in rally_player_rows:
                rally_player_rows[rally_id] = {1: [], 2: []}
            rally_player_rows[rally_id][player_id].append(row)

        # Build final response
        rally_items = []
        for rally_id, metadata in rally_metadata_map.items():
            player_data = rally_player_rows.get(rally_id, {1: [], 2: []})

            p1_dist = build_shot_type_dist(player_data.get(1, []))
            p2_dist = build_shot_type_dist(player_data.get(2, []))

            rally_items.append(
                ShotTypeDistributionPerRallyItem(
                    rally_id=rally_id,
                    game_number=metadata["game_number"],
                    rally_start_time=metadata["rally_start_time"],
                    rally_duration=metadata["rally_duration"],
                    shot_count=metadata["shot_count"],
                    point_winner=metadata["point_winner"],
                    player_1=p1_dist,
                    player_2=p2_dist,
                )
            )

        return ShotTypeDistributionPerRallyResponse(
            video_id=video_id,
            filters=filters,
            data=rally_items,
            total_rallies=len(rally_items),
            all_shot_types=all_shot_types,
        )

    def get_rhythm_disruption_per_game(
        self, video_id: str, filters: AnalyticsFilters
    ) -> RhythmDisruptionPerGameResponse:
        """Get rhythm disruption statistics per game with both players' data - Pure SQL."""
        games_metadata = self._get_games_metadata(video_id, filters)
        if not games_metadata:
            return RhythmDisruptionPerGameResponse(
                video_id=video_id, data=[], total_games=0
            )

        where_sql, params = self._build_where_clause(video_id, filters)
        params["video_id"] = video_id

        query = text(
            f"""
            WITH game_player_stats AS (
                SELECT
                    g.game_number,
                    f.racket_hit_player_id as player_id,
                    AVG(f.ball_speed) as mean_ball_speed,
                    STDDEV_POP(f.ball_speed) as stddev_ball_speed,
                    VARIANCE(f.ball_speed) as ball_speed_variance,
                    AVG(f.wall_hit_height) as mean_wall_height,
                    STDDEV_POP(f.wall_hit_height) as stddev_wall_height,
                    VARIANCE(f.wall_hit_height) as wall_height_variance,
                    COUNT(*) FILTER (WHERE f.is_racket_hit = TRUE) as player_shot_count
                FROM games g
                JOIN frame_data f ON f.video_id = g.video_id
                    AND f.rally_id BETWEEN g.start_rally_id AND g.end_rally_id
                WHERE g.{where_sql}
                  AND f.racket_hit_player_id IN (1, 2)
                  AND f.ball_speed IS NOT NULL
                  AND f.wall_hit_height IS NOT NULL
                GROUP BY g.game_number, f.racket_hit_player_id
            )
            SELECT
                game_number,
                player_id,
                player_shot_count,
                mean_ball_speed,
                stddev_ball_speed,
                ball_speed_variance,
                CASE
                    WHEN mean_ball_speed > 0
                    THEN (stddev_ball_speed / mean_ball_speed)
                    ELSE 0
                END as ball_speed_cv,
                mean_wall_height,
                stddev_wall_height,
                wall_height_variance,
                CASE
                    WHEN mean_wall_height > 0
                    THEN (stddev_wall_height / mean_wall_height)
                    ELSE 0
                END as wall_height_cv
            FROM game_player_stats
            ORDER BY game_number, player_id
        """
        )

        results = self.db.execute(query, params).fetchall()

        # Build rhythm disruption data
        def build_rhythm_data(row):
            if row is None:
                return RhythmDisruptionData(
                    ball_speed_cv=0.0,
                    ball_speed_variance=0.0,
                    wall_hit_height_cv=0.0,
                    wall_hit_height_variance=0.0,
                    shot_count=0,
                )
            return RhythmDisruptionData(
                ball_speed_cv=(
                    float(row.ball_speed_cv) if row.ball_speed_cv is not None else 0.0
                ),
                ball_speed_variance=(
                    float(row.ball_speed_variance)
                    if row.ball_speed_variance is not None
                    else 0.0
                ),
                wall_hit_height_cv=(
                    float(row.wall_height_cv) if row.wall_height_cv is not None else 0.0
                ),
                wall_hit_height_variance=(
                    float(row.wall_height_variance)
                    if row.wall_height_variance is not None
                    else 0.0
                ),
                shot_count=int(row.player_shot_count) if row.player_shot_count else 0,
            )

        # Organize by game -> player
        game_player_rows = {}
        for row in results:
            game_num = row.game_number
            player_id = row.player_id
            if game_num not in game_player_rows:
                game_player_rows[game_num] = {}
            game_player_rows[game_num][player_id] = row

        # Build final response
        game_items = []
        for game_meta in games_metadata:
            game_num = game_meta["game_number"]
            player_data = game_player_rows.get(game_num, {})

            p1_data = build_rhythm_data(player_data.get(1))
            p2_data = build_rhythm_data(player_data.get(2))

            game_items.append(
                RhythmDisruptionPerGameItem(
                    game_number=game_num,
                    start_rally_id=game_meta["start_rally_id"],
                    end_rally_id=game_meta["end_rally_id"],
                    start_time=game_meta["start_time"],
                    end_time=game_meta["end_time"],
                    duration=game_meta["duration"],
                    rally_count=game_meta["rally_count"],
                    player_1=p1_data,
                    player_2=p2_data,
                )
            )

        return RhythmDisruptionPerGameResponse(
            video_id=video_id, data=game_items, total_games=len(game_items)
        )

    def get_rhythm_disruption_per_rally(
        self, video_id: str, filters: AnalyticsFilters
    ) -> RhythmDisruptionPerRallyResponse:
        """Get rhythm disruption statistics per rally with both players' data - Pure SQL."""
        where_sql, params = self._build_where_clause(video_id, filters)
        params["video_id"] = video_id

        query = text(
            f"""
            WITH rally_metadata AS (
                SELECT DISTINCT
                    rally_id,
                    MIN(timestamp) as rally_start_time,
                    MAX(timestamp) - MIN(timestamp) as rally_duration,
                    COUNT(CASE WHEN is_racket_hit = TRUE THEN 1 END) as shot_count,
                    MAX(point_winner) as point_winner
                FROM frame_data
                WHERE {where_sql} AND rally_id IS NOT NULL
                GROUP BY rally_id
            ),
            rally_game_mapping AS (
                SELECT f.rally_id, MIN(g.game_number) as game_number
                FROM frame_data f
                JOIN games g ON f.video_id = g.video_id
                    AND f.rally_id BETWEEN g.start_rally_id AND g.end_rally_id
                WHERE f.video_id = :video_id
                GROUP BY f.rally_id
            ),
            rally_player_stats AS (
                SELECT
                    f.rally_id,
                    f.racket_hit_player_id as player_id,
                    AVG(f.ball_speed) as mean_ball_speed,
                    STDDEV_POP(f.ball_speed) as stddev_ball_speed,
                    VARIANCE(f.ball_speed) as ball_speed_variance,
                    AVG(f.wall_hit_height) as mean_wall_height,
                    STDDEV_POP(f.wall_hit_height) as stddev_wall_height,
                    VARIANCE(f.wall_hit_height) as wall_height_variance,
                    COUNT(*) FILTER (WHERE f.is_racket_hit = TRUE) as player_shot_count
                FROM frame_data f
                WHERE {where_sql}
                  AND f.rally_id IS NOT NULL
                  AND f.racket_hit_player_id IN (1, 2)
                  AND f.ball_speed IS NOT NULL
                  AND f.wall_hit_height IS NOT NULL
                GROUP BY f.rally_id, f.racket_hit_player_id
            )
            SELECT
                rm.rally_id,
                rm.rally_start_time,
                rm.rally_duration,
                rm.shot_count,
                rm.point_winner,
                rgm.game_number,
                rps.player_id,
                rps.player_shot_count,
                rps.mean_ball_speed,
                rps.stddev_ball_speed,
                rps.ball_speed_variance,
                CASE
                    WHEN rps.mean_ball_speed > 0
                    THEN (rps.stddev_ball_speed / rps.mean_ball_speed)
                    ELSE 0
                END as ball_speed_cv,
                rps.mean_wall_height,
                rps.stddev_wall_height,
                rps.wall_height_variance,
                CASE
                    WHEN rps.mean_wall_height > 0
                    THEN (rps.stddev_wall_height / rps.mean_wall_height)
                    ELSE 0
                END as wall_height_cv
            FROM rally_metadata rm
            LEFT JOIN rally_game_mapping rgm ON rm.rally_id = rgm.rally_id
            LEFT JOIN rally_player_stats rps ON rm.rally_id = rps.rally_id
            WHERE rps.player_id IS NOT NULL
            ORDER BY rm.rally_id, rps.player_id
        """
        )

        results = self.db.execute(query, params).fetchall()

        # Build rhythm disruption data
        def build_rhythm_data(row):
            if row is None:
                return RhythmDisruptionData(
                    ball_speed_cv=0.0,
                    ball_speed_variance=0.0,
                    wall_hit_height_cv=0.0,
                    wall_hit_height_variance=0.0,
                    shot_count=0,
                )
            return RhythmDisruptionData(
                ball_speed_cv=(
                    float(row.ball_speed_cv) if row.ball_speed_cv is not None else 0.0
                ),
                ball_speed_variance=(
                    float(row.ball_speed_variance)
                    if row.ball_speed_variance is not None
                    else 0.0
                ),
                wall_hit_height_cv=(
                    float(row.wall_height_cv) if row.wall_height_cv is not None else 0.0
                ),
                wall_hit_height_variance=(
                    float(row.wall_height_variance)
                    if row.wall_height_variance is not None
                    else 0.0
                ),
                shot_count=int(row.player_shot_count) if row.player_shot_count else 0,
            )

        # Organize by rally -> player
        rally_player_rows = {}
        rally_metadata_map = {}
        for row in results:
            rally_id = row.rally_id
            player_id = row.player_id

            if rally_id not in rally_metadata_map:
                rally_metadata_map[rally_id] = {
                    "rally_start_time": row.rally_start_time,
                    "rally_duration": row.rally_duration,
                    "shot_count": row.shot_count,
                    "point_winner": row.point_winner,
                    "game_number": row.game_number,
                }

            if rally_id not in rally_player_rows:
                rally_player_rows[rally_id] = {}
            rally_player_rows[rally_id][player_id] = row

        # Build final response
        rally_items = []
        for rally_id, metadata in rally_metadata_map.items():
            player_data = rally_player_rows.get(rally_id, {})

            p1_data = build_rhythm_data(player_data.get(1))
            p2_data = build_rhythm_data(player_data.get(2))

            rally_items.append(
                RhythmDisruptionPerRallyItem(
                    rally_id=rally_id,
                    game_number=metadata["game_number"],
                    rally_start_time=metadata["rally_start_time"],
                    rally_duration=metadata["rally_duration"],
                    shot_count=metadata["shot_count"],
                    point_winner=metadata["point_winner"],
                    player_1=p1_data,
                    player_2=p2_data,
                )
            )

        return RhythmDisruptionPerRallyResponse(
            video_id=video_id, data=rally_items, total_rallies=len(rally_items)
        )

    def get_court_quadrant_per_game(
        self, video_id: str, filters: AnalyticsFilters
    ) -> CourtQuadrantPerGameResponse:
        """Get court quadrant distribution per game with both players' data - Pure SQL."""
        X_CUT = 3.2
        Y_CUT = 5.44

        games_metadata = self._get_games_metadata(video_id, filters)
        if not games_metadata:
            return CourtQuadrantPerGameResponse(
                video_id=video_id,
                filters=filters,
                data=[],
                total_games=0,
                quadrant_boundaries={"x_cut": X_CUT, "y_cut": Y_CUT},
            )

        where_sql, params = self._build_where_clause(video_id, filters)
        params["video_id"] = video_id

        query = text(
            f"""
            WITH game_player_quadrants AS (
                SELECT
                    g.game_number,
                    1 as player_id,
                    CASE
                        WHEN player_1_y_meter < {Y_CUT} THEN
                            CASE WHEN player_1_x_meter < {X_CUT} THEN 'Front-Left' ELSE 'Front-Right' END
                        ELSE
                            CASE WHEN player_1_x_meter < {X_CUT} THEN 'Back-Left' ELSE 'Back-Right' END
                    END as quadrant
                FROM games g
                JOIN frame_data f ON f.video_id = g.video_id
                    AND f.rally_id BETWEEN g.start_rally_id AND g.end_rally_id
                WHERE g.{where_sql}
                  AND f.is_rally_frame = TRUE
                  AND f.player_1_x_meter IS NOT NULL
                  AND f.player_1_y_meter IS NOT NULL

                UNION ALL

                SELECT
                    g.game_number,
                    2 as player_id,
                    CASE
                        WHEN player_2_y_meter < {Y_CUT} THEN
                            CASE WHEN player_2_x_meter < {X_CUT} THEN 'Front-Left' ELSE 'Front-Right' END
                        ELSE
                            CASE WHEN player_2_x_meter < {X_CUT} THEN 'Back-Left' ELSE 'Back-Right' END
                    END as quadrant
                FROM games g
                JOIN frame_data f ON f.video_id = g.video_id
                    AND f.rally_id BETWEEN g.start_rally_id AND g.end_rally_id
                WHERE g.{where_sql}
                  AND f.is_rally_frame = TRUE
                  AND f.player_2_x_meter IS NOT NULL
                  AND f.player_2_y_meter IS NOT NULL
            ),
            quadrant_counts AS (
                SELECT
                    game_number,
                    player_id,
                    quadrant,
                    COUNT(*) as count
                FROM game_player_quadrants
                GROUP BY game_number, player_id, quadrant
            ),
            player_totals AS (
                SELECT
                    game_number,
                    player_id,
                    SUM(count) as total
                FROM quadrant_counts
                GROUP BY game_number, player_id
            )
            SELECT
                qc.game_number,
                qc.player_id,
                qc.quadrant,
                qc.count,
                pt.total,
                CASE
                    WHEN pt.total > 0
                    THEN (qc.count::float / pt.total * 100)
                    ELSE 0
                END as percentage
            FROM quadrant_counts qc
            JOIN player_totals pt ON qc.game_number = pt.game_number
                AND qc.player_id = pt.player_id
            ORDER BY qc.game_number, qc.player_id, qc.quadrant
        """
        )

        results = self.db.execute(query, params).fetchall()

        # Build quadrant distribution
        def build_quadrant_dist(rows):
            distribution = []
            total = 0
            for r in rows:
                quadrant = r.quadrant
                count = r.count
                percentage = r.percentage
                distribution.append(
                    DistributionItem(label=quadrant, count=count, percentage=percentage)
                )
                total += count
            return SingleDistribution(distribution=distribution, total=total)

        # Organize by game -> player
        game_player_rows = {}
        for row in results:
            game_num = row.game_number
            player_id = row.player_id
            if game_num not in game_player_rows:
                game_player_rows[game_num] = {1: [], 2: []}
            game_player_rows[game_num][player_id].append(row)

        # Build final response
        game_items = []
        for game_meta in games_metadata:
            game_num = game_meta["game_number"]
            player_data = game_player_rows.get(game_num, {1: [], 2: []})

            p1_dist = build_quadrant_dist(player_data.get(1, []))
            p2_dist = build_quadrant_dist(player_data.get(2, []))

            game_items.append(
                CourtQuadrantPerGameItem(
                    game_number=game_num,
                    start_rally_id=game_meta["start_rally_id"],
                    end_rally_id=game_meta["end_rally_id"],
                    start_time=game_meta["start_time"],
                    end_time=game_meta["end_time"],
                    duration=game_meta["duration"],
                    rally_count=game_meta["rally_count"],
                    player_1=p1_dist,
                    player_2=p2_dist,
                )
            )

        return CourtQuadrantPerGameResponse(
            video_id=video_id,
            filters=filters,
            data=game_items,
            total_games=len(game_items),
            quadrant_boundaries={"x_cut": X_CUT, "y_cut": Y_CUT},
        )

    def get_court_quadrant_per_rally(
        self, video_id: str, filters: AnalyticsFilters
    ) -> CourtQuadrantPerRallyResponse:
        """Get court quadrant distribution per rally with both players' data - Pure SQL."""
        X_CUT = 3.2
        Y_CUT = 5.44

        where_sql, params = self._build_where_clause(video_id, filters)
        params["video_id"] = video_id

        query = text(
            f"""
            WITH rally_metadata AS (
                SELECT DISTINCT
                    rally_id,
                    MIN(timestamp) as rally_start_time,
                    MAX(timestamp) - MIN(timestamp) as rally_duration,
                    COUNT(CASE WHEN is_racket_hit = TRUE THEN 1 END) as shot_count,
                    MAX(point_winner) as point_winner
                FROM frame_data
                WHERE {where_sql} AND rally_id IS NOT NULL
                GROUP BY rally_id
            ),
            rally_game_mapping AS (
                SELECT f.rally_id, MIN(g.game_number) as game_number
                FROM frame_data f
                JOIN games g ON f.video_id = g.video_id
                    AND f.rally_id BETWEEN g.start_rally_id AND g.end_rally_id
                WHERE f.video_id = :video_id
                GROUP BY f.rally_id
            ),
            rally_player_quadrants AS (
                SELECT
                    f.rally_id,
                    1 as player_id,
                    CASE
                        WHEN player_1_y_meter < {Y_CUT} THEN
                            CASE WHEN player_1_x_meter < {X_CUT} THEN 'Front-Left' ELSE 'Front-Right' END
                        ELSE
                            CASE WHEN player_1_x_meter < {X_CUT} THEN 'Back-Left' ELSE 'Back-Right' END
                    END as quadrant
                FROM frame_data f
                WHERE {where_sql}
                  AND f.rally_id IS NOT NULL
                  AND f.is_rally_frame = TRUE
                  AND f.player_1_x_meter IS NOT NULL
                  AND f.player_1_y_meter IS NOT NULL

                UNION ALL

                SELECT
                    f.rally_id,
                    2 as player_id,
                    CASE
                        WHEN player_2_y_meter < {Y_CUT} THEN
                            CASE WHEN player_2_x_meter < {X_CUT} THEN 'Front-Left' ELSE 'Front-Right' END
                        ELSE
                            CASE WHEN player_2_x_meter < {X_CUT} THEN 'Back-Left' ELSE 'Back-Right' END
                    END as quadrant
                FROM frame_data f
                WHERE {where_sql}
                  AND f.rally_id IS NOT NULL
                  AND f.is_rally_frame = TRUE
                  AND f.player_2_x_meter IS NOT NULL
                  AND f.player_2_y_meter IS NOT NULL
            ),
            quadrant_counts AS (
                SELECT
                    rally_id,
                    player_id,
                    quadrant,
                    COUNT(*) as count
                FROM rally_player_quadrants
                GROUP BY rally_id, player_id, quadrant
            ),
            player_totals AS (
                SELECT
                    rally_id,
                    player_id,
                    SUM(count) as total
                FROM quadrant_counts
                GROUP BY rally_id, player_id
            )
            SELECT
                rm.rally_id,
                rm.rally_start_time,
                rm.rally_duration,
                rm.shot_count,
                rm.point_winner,
                rgm.game_number,
                qc.player_id,
                qc.quadrant,
                qc.count,
                pt.total,
                CASE
                    WHEN pt.total > 0
                    THEN (qc.count::float / pt.total * 100)
                    ELSE 0
                END as percentage
            FROM rally_metadata rm
            LEFT JOIN rally_game_mapping rgm ON rm.rally_id = rgm.rally_id
            LEFT JOIN quadrant_counts qc ON rm.rally_id = qc.rally_id
            LEFT JOIN player_totals pt ON qc.rally_id = pt.rally_id
                AND qc.player_id = pt.player_id
            WHERE qc.player_id IS NOT NULL
            ORDER BY rm.rally_id, qc.player_id, qc.quadrant
        """
        )

        results = self.db.execute(query, params).fetchall()

        # Build quadrant distribution
        def build_quadrant_dist(rows):
            distribution = []
            total = 0
            for r in rows:
                quadrant = r.quadrant
                count = r.count
                percentage = r.percentage
                distribution.append(
                    DistributionItem(label=quadrant, count=count, percentage=percentage)
                )
                total += count
            return SingleDistribution(distribution=distribution, total=total)

        # Organize by rally -> player
        rally_player_rows = {}
        rally_metadata_map = {}
        for row in results:
            rally_id = row.rally_id
            player_id = row.player_id

            if rally_id not in rally_metadata_map:
                rally_metadata_map[rally_id] = {
                    "rally_start_time": row.rally_start_time,
                    "rally_duration": row.rally_duration,
                    "shot_count": row.shot_count,
                    "point_winner": row.point_winner,
                    "game_number": row.game_number,
                }

            if rally_id not in rally_player_rows:
                rally_player_rows[rally_id] = {1: [], 2: []}
            rally_player_rows[rally_id][player_id].append(row)

        # Build final response
        rally_items = []
        for rally_id, metadata in rally_metadata_map.items():
            player_data = rally_player_rows.get(rally_id, {1: [], 2: []})

            p1_dist = build_quadrant_dist(player_data.get(1, []))
            p2_dist = build_quadrant_dist(player_data.get(2, []))

            rally_items.append(
                CourtQuadrantPerRallyItem(
                    rally_id=rally_id,
                    game_number=metadata["game_number"],
                    rally_start_time=metadata["rally_start_time"],
                    rally_duration=metadata["rally_duration"],
                    shot_count=metadata["shot_count"],
                    point_winner=metadata["point_winner"],
                    player_1=p1_dist,
                    player_2=p2_dist,
                )
            )

        return CourtQuadrantPerRallyResponse(
            video_id=video_id,
            filters=filters,
            data=rally_items,
            total_rallies=len(rally_items),
            quadrant_boundaries={"x_cut": X_CUT, "y_cut": Y_CUT},
        )

    def get_wall_quadrant_per_game(
        self, video_id: str, filters: AnalyticsFilters
    ) -> WallQuadrantPerGameResponse:
        """Get wall quadrant distribution per game with both players' data - Pure SQL."""
        X_CUT = 3.2
        Y_CUT = 2.285

        games_metadata = self._get_games_metadata(video_id, filters)
        if not games_metadata:
            return WallQuadrantPerGameResponse(
                video_id=video_id,
                filters=filters,
                data=[],
                total_games=0,
                quadrant_boundaries={"x_cut": X_CUT, "y_cut": Y_CUT},
            )

        where_sql, params = self._build_where_clause(video_id, filters)
        params["video_id"] = video_id

        query = text(
            f"""
            WITH game_player_wall_quadrants AS (
                SELECT
                    g.game_number,
                    f.wall_hit_player_id as player_id,
                    CASE
                        WHEN wall_hit_y_meter < {Y_CUT} THEN
                            CASE WHEN wall_hit_x_meter < {X_CUT} THEN 'Bottom-Left' ELSE 'Bottom-Right' END
                        ELSE
                            CASE WHEN wall_hit_x_meter < {X_CUT} THEN 'Top-Left' ELSE 'Top-Right' END
                    END as quadrant
                FROM games g
                JOIN frame_data f ON f.video_id = g.video_id
                    AND f.rally_id BETWEEN g.start_rally_id AND g.end_rally_id
                WHERE g.{where_sql}
                  AND f.is_wall_hit = TRUE
                  AND f.wall_hit_x_meter IS NOT NULL
                  AND f.wall_hit_y_meter IS NOT NULL
                  AND f.wall_hit_player_id IN (1, 2)
            ),
            quadrant_counts AS (
                SELECT
                    game_number,
                    player_id,
                    quadrant,
                    COUNT(*) as count
                FROM game_player_wall_quadrants
                GROUP BY game_number, player_id, quadrant
            ),
            player_totals AS (
                SELECT
                    game_number,
                    player_id,
                    SUM(count) as total
                FROM quadrant_counts
                GROUP BY game_number, player_id
            )
            SELECT
                qc.game_number,
                qc.player_id,
                qc.quadrant,
                qc.count,
                pt.total,
                CASE
                    WHEN pt.total > 0
                    THEN (qc.count::float / pt.total * 100)
                    ELSE 0
                END as percentage
            FROM quadrant_counts qc
            JOIN player_totals pt ON qc.game_number = pt.game_number
                AND qc.player_id = pt.player_id
            ORDER BY qc.game_number, qc.player_id, qc.quadrant
        """
        )

        results = self.db.execute(query, params).fetchall()

        # Build wall quadrant distribution
        def build_wall_quadrant_dist(rows):
            distribution = []
            total = 0
            for r in rows:
                quadrant = r.quadrant
                count = r.count
                percentage = r.percentage
                distribution.append(
                    DistributionItem(label=quadrant, count=count, percentage=percentage)
                )
                total += count
            return SingleDistribution(distribution=distribution, total=total)

        # Organize by game -> player
        game_player_rows = {}
        for row in results:
            game_num = row.game_number
            player_id = row.player_id
            if game_num not in game_player_rows:
                game_player_rows[game_num] = {1: [], 2: []}
            game_player_rows[game_num][player_id].append(row)

        # Build final response
        game_items = []
        for game_meta in games_metadata:
            game_num = game_meta["game_number"]
            player_data = game_player_rows.get(game_num, {1: [], 2: []})

            p1_dist = build_wall_quadrant_dist(player_data.get(1, []))
            p2_dist = build_wall_quadrant_dist(player_data.get(2, []))

            game_items.append(
                WallQuadrantPerGameItem(
                    game_number=game_num,
                    start_rally_id=game_meta["start_rally_id"],
                    end_rally_id=game_meta["end_rally_id"],
                    start_time=game_meta["start_time"],
                    end_time=game_meta["end_time"],
                    duration=game_meta["duration"],
                    rally_count=game_meta["rally_count"],
                    player_1=p1_dist,
                    player_2=p2_dist,
                )
            )

        return WallQuadrantPerGameResponse(
            video_id=video_id,
            filters=filters,
            data=game_items,
            total_games=len(game_items),
            quadrant_boundaries={"x_cut": X_CUT, "y_cut": Y_CUT},
        )

    def get_wall_quadrant_per_rally(
        self, video_id: str, filters: AnalyticsFilters
    ) -> WallQuadrantPerRallyResponse:
        """Get wall quadrant distribution per rally with both players' data - Pure SQL."""
        X_CUT = 3.2
        Y_CUT = 2.285

        where_sql, params = self._build_where_clause(video_id, filters)
        params["video_id"] = video_id

        query = text(
            f"""
            WITH rally_metadata AS (
                SELECT DISTINCT
                    rally_id,
                    MIN(timestamp) as rally_start_time,
                    MAX(timestamp) - MIN(timestamp) as rally_duration,
                    COUNT(CASE WHEN is_racket_hit = TRUE THEN 1 END) as shot_count,
                    MAX(point_winner) as point_winner
                FROM frame_data
                WHERE {where_sql} AND rally_id IS NOT NULL
                GROUP BY rally_id
            ),
            rally_game_mapping AS (
                SELECT f.rally_id, MIN(g.game_number) as game_number
                FROM frame_data f
                JOIN games g ON f.video_id = g.video_id
                    AND f.rally_id BETWEEN g.start_rally_id AND g.end_rally_id
                WHERE f.video_id = :video_id
                GROUP BY f.rally_id
            ),
            rally_player_wall_quadrants AS (
                SELECT
                    f.rally_id,
                    f.wall_hit_player_id as player_id,
                    CASE
                        WHEN wall_hit_y_meter < {Y_CUT} THEN
                            CASE WHEN wall_hit_x_meter < {X_CUT} THEN 'Bottom-Left' ELSE 'Bottom-Right' END
                        ELSE
                            CASE WHEN wall_hit_x_meter < {X_CUT} THEN 'Top-Left' ELSE 'Top-Right' END
                    END as quadrant
                FROM frame_data f
                WHERE {where_sql}
                  AND f.rally_id IS NOT NULL
                  AND f.is_wall_hit = TRUE
                  AND f.wall_hit_x_meter IS NOT NULL
                  AND f.wall_hit_y_meter IS NOT NULL
                  AND f.wall_hit_player_id IN (1, 2)
            ),
            quadrant_counts AS (
                SELECT
                    rally_id,
                    player_id,
                    quadrant,
                    COUNT(*) as count
                FROM rally_player_wall_quadrants
                GROUP BY rally_id, player_id, quadrant
            ),
            player_totals AS (
                SELECT
                    rally_id,
                    player_id,
                    SUM(count) as total
                FROM quadrant_counts
                GROUP BY rally_id, player_id
            )
            SELECT
                rm.rally_id,
                rm.rally_start_time,
                rm.rally_duration,
                rm.shot_count,
                rm.point_winner,
                rgm.game_number,
                qc.player_id,
                qc.quadrant,
                qc.count,
                pt.total,
                CASE
                    WHEN pt.total > 0
                    THEN (qc.count::float / pt.total * 100)
                    ELSE 0
                END as percentage
            FROM rally_metadata rm
            LEFT JOIN rally_game_mapping rgm ON rm.rally_id = rgm.rally_id
            LEFT JOIN quadrant_counts qc ON rm.rally_id = qc.rally_id
            LEFT JOIN player_totals pt ON qc.rally_id = pt.rally_id
                AND qc.player_id = pt.player_id
            WHERE qc.player_id IS NOT NULL
            ORDER BY rm.rally_id, qc.player_id, qc.quadrant
        """
        )

        results = self.db.execute(query, params).fetchall()

        # Build wall quadrant distribution
        def build_wall_quadrant_dist(rows):
            distribution = []
            total = 0
            for r in rows:
                quadrant = r.quadrant
                count = r.count
                percentage = r.percentage
                distribution.append(
                    DistributionItem(label=quadrant, count=count, percentage=percentage)
                )
                total += count
            return SingleDistribution(distribution=distribution, total=total)

        # Organize by rally -> player
        rally_player_rows = {}
        rally_metadata_map = {}
        for row in results:
            rally_id = row.rally_id
            player_id = row.player_id

            if rally_id not in rally_metadata_map:
                rally_metadata_map[rally_id] = {
                    "rally_start_time": row.rally_start_time,
                    "rally_duration": row.rally_duration,
                    "shot_count": row.shot_count,
                    "point_winner": row.point_winner,
                    "game_number": row.game_number,
                }

            if rally_id not in rally_player_rows:
                rally_player_rows[rally_id] = {1: [], 2: []}
            rally_player_rows[rally_id][player_id].append(row)

        # Build final response
        rally_items = []
        for rally_id, metadata in rally_metadata_map.items():
            player_data = rally_player_rows.get(rally_id, {1: [], 2: []})

            p1_dist = build_wall_quadrant_dist(player_data.get(1, []))
            p2_dist = build_wall_quadrant_dist(player_data.get(2, []))

            rally_items.append(
                WallQuadrantPerRallyItem(
                    rally_id=rally_id,
                    game_number=metadata["game_number"],
                    rally_start_time=metadata["rally_start_time"],
                    rally_duration=metadata["rally_duration"],
                    shot_count=metadata["shot_count"],
                    point_winner=metadata["point_winner"],
                    player_1=p1_dist,
                    player_2=p2_dist,
                )
            )

        return WallQuadrantPerRallyResponse(
            video_id=video_id,
            filters=filters,
            data=rally_items,
            total_rallies=len(rally_items),
            quadrant_boundaries={"x_cut": X_CUT, "y_cut": Y_CUT},
        )

    def get_movement_metrics_per_game(
        self, video_id: str, filters: AnalyticsFilters
    ) -> MovementMetricsPerGameResponse:
        """Get movement metrics per game with both players' data - Pure SQL."""
        games_metadata = self._get_games_metadata(video_id, filters)
        if not games_metadata:
            return MovementMetricsPerGameResponse(
                video_id=video_id, data=[], total_games=0
            )

        where_sql, params = self._build_where_clause(video_id, filters)
        params["video_id"] = video_id

        query = text(
            f"""
            WITH frame_distances AS (
                SELECT
                    g.game_number,
                    f.rally_id,
                    f.frame_number,
                    -- Player 1 distance from previous frame
                    SQRT(
                        POW(f.player_1_x_meter - LAG(f.player_1_x_meter) OVER (PARTITION BY f.rally_id ORDER BY f.frame_number), 2) +
                        POW(f.player_1_y_meter - LAG(f.player_1_y_meter) OVER (PARTITION BY f.rally_id ORDER BY f.frame_number), 2)
                    ) as p1_frame_distance,
                    -- Player 2 distance from previous frame
                    SQRT(
                        POW(f.player_2_x_meter - LAG(f.player_2_x_meter) OVER (PARTITION BY f.rally_id ORDER BY f.frame_number), 2) +
                        POW(f.player_2_y_meter - LAG(f.player_2_y_meter) OVER (PARTITION BY f.rally_id ORDER BY f.frame_number), 2)
                    ) as p2_frame_distance
                FROM games g
                JOIN frame_data f ON f.video_id = g.video_id
                    AND f.rally_id BETWEEN g.start_rally_id AND g.end_rally_id
                WHERE g.{where_sql}
                  AND f.is_rally_frame = TRUE
                  AND f.player_1_x_meter IS NOT NULL
                  AND f.player_1_y_meter IS NOT NULL
                  AND f.player_2_x_meter IS NOT NULL
                  AND f.player_2_y_meter IS NOT NULL
            ),
            rally_distances AS (
                SELECT
                    game_number,
                    rally_id,
                    SUM(p1_frame_distance) as p1_rally_distance,
                    SUM(p2_frame_distance) as p2_rally_distance
                FROM frame_distances
                GROUP BY game_number, rally_id
            ),
            game_shot_counts AS (
                SELECT
                    g.game_number,
                    COUNT(CASE WHEN f.is_racket_hit = TRUE THEN 1 END) as total_shots
                FROM games g
                JOIN frame_data f ON f.video_id = g.video_id
                    AND f.rally_id BETWEEN g.start_rally_id AND g.end_rally_id
                WHERE g.{where_sql}
                GROUP BY g.game_number
            ),
            game_distances AS (
                SELECT
                    rd.game_number,
                    SUM(rd.p1_rally_distance) as p1_total_distance,
                    AVG(rd.p1_rally_distance) as p1_avg_distance_per_rally,
                    SUM(rd.p2_rally_distance) as p2_total_distance,
                    AVG(rd.p2_rally_distance) as p2_avg_distance_per_rally,
                    COUNT(*) as rally_count,
                    gsc.total_shots
                FROM rally_distances rd
                LEFT JOIN game_shot_counts gsc ON rd.game_number = gsc.game_number
                GROUP BY rd.game_number, gsc.total_shots
            )
            SELECT * FROM game_distances
            ORDER BY game_number
        """
        )

        results = self.db.execute(query, params).fetchall()

        # Build movement metrics per game
        game_metrics = {}
        for row in results:
            game_metrics[row.game_number] = {
                "p1": SingleMovementMetrics(
                    total_distance=row.p1_total_distance or 0.0,
                    avg_distance_per_rally=row.p1_avg_distance_per_rally or 0.0,
                ),
                "p2": SingleMovementMetrics(
                    total_distance=row.p2_total_distance or 0.0,
                    avg_distance_per_rally=row.p2_avg_distance_per_rally or 0.0,
                ),
            }

        # Build final response
        game_items = []
        for game_meta in games_metadata:
            game_num = game_meta["game_number"]
            metrics = game_metrics.get(
                game_num,
                {
                    "p1": SingleMovementMetrics(
                        total_distance=0.0,
                        avg_distance_per_rally=0.0,
                    ),
                    "p2": SingleMovementMetrics(
                        total_distance=0.0,
                        avg_distance_per_rally=0.0,
                    ),
                },
            )

            game_items.append(
                MovementMetricsPerGameItem(
                    game_number=game_num,
                    start_rally_id=game_meta["start_rally_id"],
                    end_rally_id=game_meta["end_rally_id"],
                    start_time=game_meta["start_time"],
                    end_time=game_meta["end_time"],
                    duration=game_meta["duration"],
                    rally_count=game_meta["rally_count"],
                    player_1=metrics["p1"],
                    player_2=metrics["p2"],
                )
            )

        return MovementMetricsPerGameResponse(
            video_id=video_id, data=game_items, total_games=len(game_items)
        )

    def get_movement_metrics_per_rally(
        self, video_id: str, filters: AnalyticsFilters
    ) -> MovementMetricsPerRallyResponse:
        """Get movement metrics per rally with both players' data - Pure SQL."""
        where_sql, params = self._build_where_clause(video_id, filters)
        params["video_id"] = video_id

        query = text(
            f"""
            WITH rally_metadata AS (
                SELECT DISTINCT
                    rally_id,
                    MIN(timestamp) as rally_start_time,
                    MAX(timestamp) - MIN(timestamp) as rally_duration,
                    COUNT(CASE WHEN is_racket_hit = TRUE THEN 1 END) as shot_count,
                    MAX(point_winner) as point_winner
                FROM frame_data
                WHERE {where_sql} AND rally_id IS NOT NULL
                GROUP BY rally_id
            ),
            rally_game_mapping AS (
                SELECT f.rally_id, MIN(g.game_number) as game_number
                FROM frame_data f
                JOIN games g ON f.video_id = g.video_id
                    AND f.rally_id BETWEEN g.start_rally_id AND g.end_rally_id
                WHERE f.video_id = :video_id
                GROUP BY f.rally_id
            ),
            frame_distances AS (
                SELECT
                    f.rally_id,
                    -- Player 1 distance from previous frame
                    SQRT(
                        POW(f.player_1_x_meter - LAG(f.player_1_x_meter) OVER (PARTITION BY f.rally_id ORDER BY f.frame_number), 2) +
                        POW(f.player_1_y_meter - LAG(f.player_1_y_meter) OVER (PARTITION BY f.rally_id ORDER BY f.frame_number), 2)
                    ) as p1_frame_distance,
                    -- Player 2 distance from previous frame
                    SQRT(
                        POW(f.player_2_x_meter - LAG(f.player_2_x_meter) OVER (PARTITION BY f.rally_id ORDER BY f.frame_number), 2) +
                        POW(f.player_2_y_meter - LAG(f.player_2_y_meter) OVER (PARTITION BY f.rally_id ORDER BY f.frame_number), 2)
                    ) as p2_frame_distance
                FROM frame_data f
                WHERE {where_sql}
                  AND f.rally_id IS NOT NULL
                  AND f.is_rally_frame = TRUE
                  AND f.player_1_x_meter IS NOT NULL
                  AND f.player_1_y_meter IS NOT NULL
                  AND f.player_2_x_meter IS NOT NULL
                  AND f.player_2_y_meter IS NOT NULL
            ),
            rally_distances AS (
                SELECT
                    rally_id,
                    SUM(p1_frame_distance) as p1_total_distance,
                    SUM(p2_frame_distance) as p2_total_distance
                FROM frame_distances
                GROUP BY rally_id
            )
            SELECT
                rm.rally_id,
                rm.rally_start_time,
                rm.rally_duration,
                rm.shot_count,
                rm.point_winner,
                rgm.game_number,
                rd.p1_total_distance,
                rd.p2_total_distance
            FROM rally_metadata rm
            LEFT JOIN rally_game_mapping rgm ON rm.rally_id = rgm.rally_id
            LEFT JOIN rally_distances rd ON rm.rally_id = rd.rally_id
            ORDER BY rm.rally_id
        """
        )

        results = self.db.execute(query, params).fetchall()

        # Build final response
        rally_items = []
        for row in results:
            p1_metrics = SingleMovementMetrics(
                total_distance=row.p1_total_distance or 0.0,
                avg_distance_per_rally=row.p1_total_distance
                or 0.0,  # Same as total for per-rally
            )
            p2_metrics = SingleMovementMetrics(
                total_distance=row.p2_total_distance or 0.0,
                avg_distance_per_rally=row.p2_total_distance
                or 0.0,  # Same as total for per-rally
            )

            rally_items.append(
                MovementMetricsPerRallyItem(
                    rally_id=row.rally_id,
                    game_number=row.game_number,
                    rally_start_time=row.rally_start_time,
                    rally_duration=row.rally_duration,
                    shot_count=row.shot_count,
                    point_winner=row.point_winner,
                    player_1=p1_metrics,
                    player_2=p2_metrics,
                )
            )

        return MovementMetricsPerRallyResponse(
            video_id=video_id, data=rally_items, total_rallies=len(rally_items)
        )

    def get_t_zone_occupancy_per_game(
        self, video_id: str, filters: AnalyticsFilters
    ) -> TZoneOccupancyPerGameResponse:
        """Get T-zone occupancy per game with both players' data - Pure SQL."""
        games_metadata = self._get_games_metadata(video_id, filters)
        if not games_metadata:
            return TZoneOccupancyPerGameResponse(
                video_id=video_id, data=[], total_games=0
            )

        where_sql, params = self._build_where_clause(video_id, filters)
        params["video_id"] = video_id

        query = text(
            f"""
            WITH game_frames AS (
                SELECT
                    g.game_number,
                    f.player_1_in_t_zone,
                    f.player_2_in_t_zone,
                    f.player_1_time_to_t,
                    f.player_2_time_to_t,
                    f.is_racket_hit,
                    f.racket_hit_player_id
                FROM games g
                JOIN frame_data f ON f.video_id = g.video_id
                    AND f.rally_id BETWEEN g.start_rally_id AND g.end_rally_id
                WHERE g.{where_sql}
                  AND f.is_rally_frame = TRUE
            ),
            game_stats AS (
                SELECT
                    game_number,
                    -- Player 1 metrics
                    COUNT(*) FILTER (WHERE player_1_in_t_zone = TRUE) as p1_frames_in_t,
                    COUNT(*) as total_frames,
                    AVG(player_1_time_to_t) FILTER (WHERE player_1_time_to_t IS NOT NULL) as p1_avg_time_to_t,
                    MIN(player_1_time_to_t) FILTER (WHERE player_1_time_to_t IS NOT NULL) as p1_min_time_to_t,
                    MAX(player_1_time_to_t) FILTER (WHERE player_1_time_to_t IS NOT NULL) as p1_max_time_to_t,
                    VARIANCE(player_1_time_to_t) FILTER (WHERE player_1_time_to_t IS NOT NULL) as p1_time_to_t_var,
                    COUNT(*) FILTER (WHERE player_1_time_to_t IS NOT NULL) as p1_successful_returns,
                    COUNT(*) FILTER (WHERE is_racket_hit = TRUE AND racket_hit_player_id = 2) as p1_opponent_shots,
                    -- Player 2 metrics
                    COUNT(*) FILTER (WHERE player_2_in_t_zone = TRUE) as p2_frames_in_t,
                    AVG(player_2_time_to_t) FILTER (WHERE player_2_time_to_t IS NOT NULL) as p2_avg_time_to_t,
                    MIN(player_2_time_to_t) FILTER (WHERE player_2_time_to_t IS NOT NULL) as p2_min_time_to_t,
                    MAX(player_2_time_to_t) FILTER (WHERE player_2_time_to_t IS NOT NULL) as p2_max_time_to_t,
                    VARIANCE(player_2_time_to_t) FILTER (WHERE player_2_time_to_t IS NOT NULL) as p2_time_to_t_var,
                    COUNT(*) FILTER (WHERE player_2_time_to_t IS NOT NULL) as p2_successful_returns,
                    COUNT(*) FILTER (WHERE is_racket_hit = TRUE AND racket_hit_player_id = 1) as p2_opponent_shots
                FROM game_frames
                GROUP BY game_number
            )
            SELECT
                game_number,
                -- Player 1
                CASE WHEN total_frames > 0 THEN (p1_frames_in_t::float / total_frames * 100) ELSE 0 END as p1_pct_time_in_t,
                p1_avg_time_to_t,
                p1_min_time_to_t,
                p1_max_time_to_t,
                p1_time_to_t_var,
                CASE WHEN p1_opponent_shots > 0 THEN (p1_successful_returns::float / p1_opponent_shots * 100) ELSE NULL END as p1_success_rate,
                p1_opponent_shots,
                p1_successful_returns,
                -- Player 2
                CASE WHEN total_frames > 0 THEN (p2_frames_in_t::float / total_frames * 100) ELSE 0 END as p2_pct_time_in_t,
                p2_avg_time_to_t,
                p2_min_time_to_t,
                p2_max_time_to_t,
                p2_time_to_t_var,
                CASE WHEN p2_opponent_shots > 0 THEN (p2_successful_returns::float / p2_opponent_shots * 100) ELSE NULL END as p2_success_rate,
                p2_opponent_shots,
                p2_successful_returns
            FROM game_stats
            ORDER BY game_number
        """
        )

        results = self.db.execute(query, params).fetchall()

        # Build T-zone metrics per game
        game_metrics = {}
        for row in results:
            game_metrics[row.game_number] = {
                "p1": SingleTZoneMetrics(
                    pct_time_in_t=row.p1_pct_time_in_t,
                    avg_time_to_t=row.p1_avg_time_to_t,
                    min_time_to_t=row.p1_min_time_to_t,
                    max_time_to_t=row.p1_max_time_to_t,
                    time_to_t_variance=row.p1_time_to_t_var,
                    t_zone_success_rate=row.p1_success_rate,
                    total_shots_taken=row.p1_opponent_shots or 0,
                    successful_returns=row.p1_successful_returns or 0,
                ),
                "p2": SingleTZoneMetrics(
                    pct_time_in_t=row.p2_pct_time_in_t,
                    avg_time_to_t=row.p2_avg_time_to_t,
                    min_time_to_t=row.p2_min_time_to_t,
                    max_time_to_t=row.p2_max_time_to_t,
                    time_to_t_variance=row.p2_time_to_t_var,
                    t_zone_success_rate=row.p2_success_rate,
                    total_shots_taken=row.p2_opponent_shots or 0,
                    successful_returns=row.p2_successful_returns or 0,
                ),
            }

        # Build final response
        game_items = []
        for game_meta in games_metadata:
            game_num = game_meta["game_number"]
            metrics = game_metrics.get(
                game_num,
                {
                    "p1": SingleTZoneMetrics(
                        pct_time_in_t=0.0,
                        avg_time_to_t=None,
                        min_time_to_t=None,
                        max_time_to_t=None,
                        time_to_t_variance=None,
                        t_zone_success_rate=None,
                        total_shots_taken=0,
                        successful_returns=0,
                    ),
                    "p2": SingleTZoneMetrics(
                        pct_time_in_t=0.0,
                        avg_time_to_t=None,
                        min_time_to_t=None,
                        max_time_to_t=None,
                        time_to_t_variance=None,
                        t_zone_success_rate=None,
                        total_shots_taken=0,
                        successful_returns=0,
                    ),
                },
            )

            game_items.append(
                TZoneOccupancyPerGameItem(
                    game_number=game_num,
                    start_rally_id=game_meta["start_rally_id"],
                    end_rally_id=game_meta["end_rally_id"],
                    start_time=game_meta["start_time"],
                    end_time=game_meta["end_time"],
                    duration=game_meta["duration"],
                    rally_count=game_meta["rally_count"],
                    player_1=metrics["p1"],
                    player_2=metrics["p2"],
                )
            )

        return TZoneOccupancyPerGameResponse(
            video_id=video_id, data=game_items, total_games=len(game_items)
        )

    def get_t_zone_occupancy_per_rally(
        self, video_id: str, filters: AnalyticsFilters
    ) -> TZoneOccupancyPerRallyResponse:
        """Get T-zone occupancy per rally with both players' data - Pure SQL."""
        where_sql, params = self._build_where_clause(video_id, filters)
        params["video_id"] = video_id

        query = text(
            f"""
            WITH rally_metadata AS (
                SELECT DISTINCT
                    rally_id,
                    MIN(timestamp) as rally_start_time,
                    MAX(timestamp) - MIN(timestamp) as rally_duration,
                    COUNT(CASE WHEN is_racket_hit = TRUE THEN 1 END) as shot_count,
                    MAX(point_winner) as point_winner
                FROM frame_data
                WHERE {where_sql} AND rally_id IS NOT NULL
                GROUP BY rally_id
            ),
            rally_game_mapping AS (
                SELECT f.rally_id, MIN(g.game_number) as game_number
                FROM frame_data f
                JOIN games g ON f.video_id = g.video_id
                    AND f.rally_id BETWEEN g.start_rally_id AND g.end_rally_id
                WHERE f.video_id = :video_id
                GROUP BY f.rally_id
            ),
            rally_frames AS (
                SELECT
                    f.rally_id,
                    f.player_1_in_t_zone,
                    f.player_2_in_t_zone,
                    f.player_1_time_to_t,
                    f.player_2_time_to_t,
                    f.is_racket_hit,
                    f.racket_hit_player_id
                FROM frame_data f
                WHERE {where_sql}
                  AND f.rally_id IS NOT NULL
                  AND f.is_rally_frame = TRUE
            ),
            rally_stats AS (
                SELECT
                    rally_id,
                    -- Player 1 metrics
                    COUNT(*) FILTER (WHERE player_1_in_t_zone = TRUE) as p1_frames_in_t,
                    COUNT(*) as total_frames,
                    AVG(player_1_time_to_t) FILTER (WHERE player_1_time_to_t IS NOT NULL) as p1_avg_time_to_t,
                    MIN(player_1_time_to_t) FILTER (WHERE player_1_time_to_t IS NOT NULL) as p1_min_time_to_t,
                    MAX(player_1_time_to_t) FILTER (WHERE player_1_time_to_t IS NOT NULL) as p1_max_time_to_t,
                    VARIANCE(player_1_time_to_t) FILTER (WHERE player_1_time_to_t IS NOT NULL) as p1_time_to_t_var,
                    COUNT(*) FILTER (WHERE player_1_time_to_t IS NOT NULL) as p1_successful_returns,
                    COUNT(*) FILTER (WHERE is_racket_hit = TRUE AND racket_hit_player_id = 2) as p1_opponent_shots,
                    -- Player 2 metrics
                    COUNT(*) FILTER (WHERE player_2_in_t_zone = TRUE) as p2_frames_in_t,
                    AVG(player_2_time_to_t) FILTER (WHERE player_2_time_to_t IS NOT NULL) as p2_avg_time_to_t,
                    MIN(player_2_time_to_t) FILTER (WHERE player_2_time_to_t IS NOT NULL) as p2_min_time_to_t,
                    MAX(player_2_time_to_t) FILTER (WHERE player_2_time_to_t IS NOT NULL) as p2_max_time_to_t,
                    VARIANCE(player_2_time_to_t) FILTER (WHERE player_2_time_to_t IS NOT NULL) as p2_time_to_t_var,
                    COUNT(*) FILTER (WHERE player_2_time_to_t IS NOT NULL) as p2_successful_returns,
                    COUNT(*) FILTER (WHERE is_racket_hit = TRUE AND racket_hit_player_id = 1) as p2_opponent_shots
                FROM rally_frames
                GROUP BY rally_id
            )
            SELECT
                rm.rally_id,
                rm.rally_start_time,
                rm.rally_duration,
                rm.shot_count,
                rm.point_winner,
                rgm.game_number,
                -- Player 1
                CASE WHEN rs.total_frames > 0 THEN (rs.p1_frames_in_t::float / rs.total_frames * 100) ELSE 0 END as p1_pct_time_in_t,
                rs.p1_avg_time_to_t,
                rs.p1_min_time_to_t,
                rs.p1_max_time_to_t,
                rs.p1_time_to_t_var,
                CASE WHEN rs.p1_opponent_shots > 0 THEN (rs.p1_successful_returns::float / rs.p1_opponent_shots * 100) ELSE NULL END as p1_success_rate,
                rs.p1_opponent_shots,
                rs.p1_successful_returns,
                -- Player 2
                CASE WHEN rs.total_frames > 0 THEN (rs.p2_frames_in_t::float / rs.total_frames * 100) ELSE 0 END as p2_pct_time_in_t,
                rs.p2_avg_time_to_t,
                rs.p2_min_time_to_t,
                rs.p2_max_time_to_t,
                rs.p2_time_to_t_var,
                CASE WHEN rs.p2_opponent_shots > 0 THEN (rs.p2_successful_returns::float / rs.p2_opponent_shots * 100) ELSE NULL END as p2_success_rate,
                rs.p2_opponent_shots,
                rs.p2_successful_returns
            FROM rally_metadata rm
            LEFT JOIN rally_game_mapping rgm ON rm.rally_id = rgm.rally_id
            LEFT JOIN rally_stats rs ON rm.rally_id = rs.rally_id
            ORDER BY rm.rally_id
        """
        )

        results = self.db.execute(query, params).fetchall()

        # Build final response
        rally_items = []
        for row in results:
            p1_metrics = SingleTZoneMetrics(
                pct_time_in_t=row.p1_pct_time_in_t or 0.0,
                avg_time_to_t=row.p1_avg_time_to_t,
                min_time_to_t=row.p1_min_time_to_t,
                max_time_to_t=row.p1_max_time_to_t,
                time_to_t_variance=row.p1_time_to_t_var,
                t_zone_success_rate=row.p1_success_rate,
                total_shots_taken=row.p1_opponent_shots or 0,
                successful_returns=row.p1_successful_returns or 0,
            )
            p2_metrics = SingleTZoneMetrics(
                pct_time_in_t=row.p2_pct_time_in_t or 0.0,
                avg_time_to_t=row.p2_avg_time_to_t,
                min_time_to_t=row.p2_min_time_to_t,
                max_time_to_t=row.p2_max_time_to_t,
                time_to_t_variance=row.p2_time_to_t_var,
                t_zone_success_rate=row.p2_success_rate,
                total_shots_taken=row.p2_opponent_shots or 0,
                successful_returns=row.p2_successful_returns or 0,
            )

            rally_items.append(
                TZoneOccupancyPerRallyItem(
                    rally_id=row.rally_id,
                    game_number=row.game_number,
                    rally_start_time=row.rally_start_time,
                    rally_duration=row.rally_duration,
                    shot_count=row.shot_count,
                    point_winner=row.point_winner,
                    player_1=p1_metrics,
                    player_2=p2_metrics,
                )
            )

        return TZoneOccupancyPerRallyResponse(
            video_id=video_id, data=rally_items, total_rallies=len(rally_items)
        )

    # ============================================================================
    # SHOT EFFECTIVENESS PER-GAME AND PER-RALLY
    # ============================================================================

    def get_shot_effectiveness_per_game(
        self, video_id: str, filters: AnalyticsFilters
    ) -> ShotEffectivenessPerGameResponse:
        """Get shot effectiveness metrics per game with both players' data."""
        logger.info(f"Computing shot effectiveness per-game for video {video_id}")
        self._check_processed(video_id)

        # Get game metadata
        games_metadata = self._get_games_metadata(video_id, filters)

        if not games_metadata:
            return ShotEffectivenessPerGameResponse(
                video_id=video_id, filters=filters, data=[], total_games=0
            )

        # Build WHERE clause
        where_sql, params = self._build_where_clause(
            video_id, filters, include_player_id=False
        )
        params["video_id"] = video_id

        # Query per-game shot effectiveness grouped by game and player
        # Uses LATERAL JOIN like the original aggregate query to find opponent's next shot
        query = text(
            f"""
            WITH shot_frames AS (
                SELECT
                    g.game_number,
                    f.rally_id,
                    f.frame_number,
                    f.racket_hit_player_id,
                    f.player_1_x_meter,
                    f.player_1_y_meter,
                    f.player_2_x_meter,
                    f.player_2_y_meter,
                    CASE
                        WHEN f.racket_hit_player_id = 1 THEN
                            SQRT(POW(f.player_2_x_meter - 3.05, 2) + POW(f.player_2_y_meter - 5.44, 2))
                        WHEN f.racket_hit_player_id = 2 THEN
                            SQRT(POW(f.player_1_x_meter - 3.05, 2) + POW(f.player_1_y_meter - 5.44, 2))
                    END as opponent_distance_from_t,
                    f.shot_type
                FROM games g
                JOIN frame_data f ON f.video_id = g.video_id
                    AND f.rally_id BETWEEN g.start_rally_id AND g.end_rally_id
                WHERE g.{where_sql}
                  AND f.is_racket_hit = TRUE
                  AND f.player_1_x_meter IS NOT NULL
                  AND f.player_2_x_meter IS NOT NULL
            ),
            shot_with_response AS (
                SELECT
                    curr.game_number,
                    curr.racket_hit_player_id,
                    curr.player_1_x_meter,
                    curr.player_1_y_meter,
                    curr.player_2_x_meter,
                    curr.player_2_y_meter,
                    curr.opponent_distance_from_t,
                    curr.shot_type,
                    next_shot.opponent_distance_from_t as next_opponent_dist_from_t,
                    next_shot.player_1_x_meter as next_player_1_x,
                    next_shot.player_1_y_meter as next_player_1_y,
                    next_shot.player_2_x_meter as next_player_2_x,
                    next_shot.player_2_y_meter as next_player_2_y
                FROM shot_frames curr
                LEFT JOIN LATERAL (
                    SELECT opponent_distance_from_t, player_1_x_meter, player_1_y_meter, player_2_x_meter, player_2_y_meter
                    FROM shot_frames next
                    WHERE next.rally_id = curr.rally_id
                      AND next.frame_number > curr.frame_number
                      AND next.racket_hit_player_id != curr.racket_hit_player_id
                    ORDER BY next.frame_number ASC
                    LIMIT 1
                ) next_shot ON TRUE
            ),
            game_effectiveness AS (
                SELECT
                    game_number,
                    racket_hit_player_id,
                    -- Displacement from T: change in opponent's distance from T
                    AVG(next_opponent_dist_from_t - opponent_distance_from_t) as avg_displacement,
                    MAX(next_opponent_dist_from_t - opponent_distance_from_t) as max_displacement,
                    VARIANCE(next_opponent_dist_from_t - opponent_distance_from_t) as displacement_variance,
                    -- Opponent distance moved: actual distance opponent moved to return shot
                    AVG(CASE
                        WHEN racket_hit_player_id = 1 AND next_player_2_x IS NOT NULL AND next_player_2_y IS NOT NULL THEN
                            SQRT(POW(next_player_2_x - player_2_x_meter, 2) + POW(next_player_2_y - player_2_y_meter, 2))
                        WHEN racket_hit_player_id = 2 AND next_player_1_x IS NOT NULL AND next_player_1_y IS NOT NULL THEN
                            SQRT(POW(next_player_1_x - player_1_x_meter, 2) + POW(next_player_1_y - player_1_y_meter, 2))
                    END) as avg_opponent_dist_moved,
                    MAX(CASE
                        WHEN racket_hit_player_id = 1 AND next_player_2_x IS NOT NULL AND next_player_2_y IS NOT NULL THEN
                            SQRT(POW(next_player_2_x - player_2_x_meter, 2) + POW(next_player_2_y - player_2_y_meter, 2))
                        WHEN racket_hit_player_id = 2 AND next_player_1_x IS NOT NULL AND next_player_1_y IS NOT NULL THEN
                            SQRT(POW(next_player_1_x - player_1_x_meter, 2) + POW(next_player_1_y - player_1_y_meter, 2))
                    END) as max_opponent_dist_moved,
                    VARIANCE(CASE
                        WHEN racket_hit_player_id = 1 AND next_player_2_x IS NOT NULL AND next_player_2_y IS NOT NULL THEN
                            SQRT(POW(next_player_2_x - player_2_x_meter, 2) + POW(next_player_2_y - player_2_y_meter, 2))
                        WHEN racket_hit_player_id = 2 AND next_player_1_x IS NOT NULL AND next_player_1_y IS NOT NULL THEN
                            SQRT(POW(next_player_1_x - player_1_x_meter, 2) + POW(next_player_1_y - player_1_y_meter, 2))
                    END) as opponent_dist_moved_variance,
                    -- Depth dominance
                    AVG(CASE
                        WHEN racket_hit_player_id = 1 AND player_2_y_meter > player_1_y_meter THEN 1
                        WHEN racket_hit_player_id = 2 AND player_1_y_meter > player_2_y_meter THEN 1
                        ELSE 0
                    END) * 100 as depth_dominance_pct,
                    AVG(CASE
                        WHEN racket_hit_player_id = 1 THEN player_2_y_meter - player_1_y_meter
                        WHEN racket_hit_player_id = 2 THEN player_1_y_meter - player_2_y_meter
                    END) as avg_depth_diff,
                    MIN(CASE
                        WHEN racket_hit_player_id = 1 THEN player_2_y_meter - player_1_y_meter
                        WHEN racket_hit_player_id = 2 THEN player_1_y_meter - player_2_y_meter
                    END) as min_depth_diff,
                    MAX(CASE
                        WHEN racket_hit_player_id = 1 THEN player_2_y_meter - player_1_y_meter
                        WHEN racket_hit_player_id = 2 THEN player_1_y_meter - player_2_y_meter
                    END) as max_depth_diff,
                    -- Straight shot quality
                    COUNT(CASE WHEN shot_type IN ('straight_drive', 'straight_drop') THEN 1 END) as straight_shots,
                    COUNT(CASE
                        WHEN shot_type IN ('straight_drive', 'straight_drop')
                        AND (
                            (racket_hit_player_id = 1 AND LEAST(player_1_x_meter, 6.1 - player_1_x_meter) <= 1.2)
                            OR (racket_hit_player_id = 2 AND LEAST(player_2_x_meter, 6.1 - player_2_x_meter) <= 1.2)
                        )
                        THEN 1
                    END) as quality_straights
                FROM shot_with_response
                WHERE next_opponent_dist_from_t IS NOT NULL
                GROUP BY game_number, racket_hit_player_id
                ORDER BY game_number, racket_hit_player_id
            )
            SELECT * FROM game_effectiveness
        """
        )

        results = self.db.execute(query, params).fetchall()

        # Define metric builder function
        def build_shot_effectiveness(row):
            return SingleShotEffectivenessMetrics(
                avg_displacement_from_t=(
                    float(row.avg_displacement)
                    if row.avg_displacement is not None
                    else None
                ),
                max_displacement_from_t=(
                    float(row.max_displacement)
                    if row.max_displacement is not None
                    else None
                ),
                displacement_variance=(
                    float(row.displacement_variance)
                    if row.displacement_variance is not None
                    else None
                ),
                avg_opponent_distance_moved=(
                    float(row.avg_opponent_dist_moved)
                    if row.avg_opponent_dist_moved is not None
                    else None
                ),
                max_opponent_distance_moved=(
                    float(row.max_opponent_dist_moved)
                    if row.max_opponent_dist_moved is not None
                    else None
                ),
                opponent_distance_moved_variance=(
                    float(row.opponent_dist_moved_variance)
                    if row.opponent_dist_moved_variance is not None
                    else None
                ),
                depth_dominance_pct=(
                    float(row.depth_dominance_pct)
                    if row.depth_dominance_pct is not None
                    else None
                ),
                avg_depth_difference=(
                    float(row.avg_depth_diff) if row.avg_depth_diff is not None else None
                ),
                min_depth_difference=(
                    float(row.min_depth_diff) if row.min_depth_diff is not None else None
                ),
                max_depth_difference=(
                    float(row.max_depth_diff) if row.max_depth_diff is not None else None
                ),
                straight_shot_quality_pct=(
                    (float(row.quality_straights) / row.straight_shots * 100)
                    if row.straight_shots > 0
                    else None
                ),
                straight_shots_count=(
                    int(row.straight_shots) if row.straight_shots else 0
                ),
                shots_close_to_wall=(
                    int(row.quality_straights) if row.quality_straights else 0
                ),
            )

        # Pivot data
        game_items_data = self._pivot_by_game(
            results, games_metadata, build_shot_effectiveness
        )

        # Create empty metrics for missing players
        empty_shot_effectiveness = SingleShotEffectivenessMetrics(
            avg_displacement_from_t=None,
            max_displacement_from_t=None,
            displacement_variance=None,
            avg_opponent_distance_moved=None,
            max_opponent_distance_moved=None,
            opponent_distance_moved_variance=None,
            depth_dominance_pct=None,
            avg_depth_difference=None,
            min_depth_difference=None,
            max_depth_difference=None,
            straight_shot_quality_pct=None,
            straight_shots_count=0,
            shots_close_to_wall=0,
        )

        # Convert to response items, filling in empty metrics for missing players
        game_items = []
        for item in game_items_data:
            game_items.append(
                ShotEffectivenessPerGameItem(
                    game_number=item["game_number"],
                    start_rally_id=item["start_rally_id"],
                    end_rally_id=item["end_rally_id"],
                    start_time=item["start_time"],
                    end_time=item["end_time"],
                    duration=item["duration"],
                    rally_count=item["rally_count"],
                    player_1=(
                        item["player_1"]
                        if item["player_1"] is not None
                        else empty_shot_effectiveness
                    ),
                    player_2=(
                        item["player_2"]
                        if item["player_2"] is not None
                        else empty_shot_effectiveness
                    ),
                )
            )

        return ShotEffectivenessPerGameResponse(
            video_id=video_id,
            filters=filters,
            data=game_items,
            total_games=len(game_items),
        )

    def get_shot_effectiveness_per_rally(
        self, video_id: str, filters: AnalyticsFilters
    ) -> ShotEffectivenessPerRallyResponse:
        """Get shot effectiveness metrics per rally with both players' data."""
        logger.info(f"Computing shot effectiveness per-rally for video {video_id}")
        self._check_processed(video_id)

        # Build WHERE clause
        where_sql, params = self._build_where_clause(
            video_id, filters, include_player_id=False
        )
        params["video_id"] = video_id

        # Query per-rally shot effectiveness grouped by rally and player
        # Uses LATERAL JOIN like the original aggregate query to find opponent's next shot
        query = text(
            f"""
            WITH rally_metadata AS (
                SELECT DISTINCT
                    rally_id,
                    MIN(timestamp) as rally_start_time,
                    MAX(timestamp) - MIN(timestamp) as rally_duration,
                    COUNT(CASE WHEN is_racket_hit = TRUE THEN 1 END) as shot_count,
                    MAX(point_winner) as point_winner
                FROM frame_data
                WHERE {where_sql}
                  AND rally_id IS NOT NULL
                GROUP BY rally_id
            ),
            shot_frames AS (
                SELECT
                    rally_id,
                    frame_number,
                    racket_hit_player_id,
                    player_1_x_meter,
                    player_1_y_meter,
                    player_2_x_meter,
                    player_2_y_meter,
                    CASE
                        WHEN racket_hit_player_id = 1 THEN
                            SQRT(POW(player_2_x_meter - 3.05, 2) + POW(player_2_y_meter - 5.44, 2))
                        WHEN racket_hit_player_id = 2 THEN
                            SQRT(POW(player_1_x_meter - 3.05, 2) + POW(player_1_y_meter - 5.44, 2))
                    END as opponent_distance_from_t,
                    shot_type
                FROM frame_data
                WHERE {where_sql}
                  AND is_racket_hit = TRUE
                  AND rally_id IS NOT NULL
                  AND player_1_x_meter IS NOT NULL
                  AND player_2_x_meter IS NOT NULL
            ),
            shot_with_response AS (
                SELECT
                    curr.rally_id,
                    curr.racket_hit_player_id,
                    curr.player_1_x_meter,
                    curr.player_1_y_meter,
                    curr.player_2_x_meter,
                    curr.player_2_y_meter,
                    curr.opponent_distance_from_t,
                    curr.shot_type,
                    next_shot.opponent_distance_from_t as next_opponent_dist_from_t,
                    next_shot.player_1_x_meter as next_player_1_x,
                    next_shot.player_1_y_meter as next_player_1_y,
                    next_shot.player_2_x_meter as next_player_2_x,
                    next_shot.player_2_y_meter as next_player_2_y
                FROM shot_frames curr
                LEFT JOIN LATERAL (
                    SELECT opponent_distance_from_t, player_1_x_meter, player_1_y_meter, player_2_x_meter, player_2_y_meter
                    FROM shot_frames next
                    WHERE next.rally_id = curr.rally_id
                      AND next.frame_number > curr.frame_number
                      AND next.racket_hit_player_id != curr.racket_hit_player_id
                    ORDER BY next.frame_number ASC
                    LIMIT 1
                ) next_shot ON TRUE
            ),
            rally_effectiveness AS (
                SELECT
                    rally_id,
                    racket_hit_player_id,
                    -- Displacement from T: change in opponent's distance from T
                    AVG(next_opponent_dist_from_t - opponent_distance_from_t) as avg_displacement,
                    MAX(next_opponent_dist_from_t - opponent_distance_from_t) as max_displacement,
                    VARIANCE(next_opponent_dist_from_t - opponent_distance_from_t) as displacement_variance,
                    -- Opponent distance moved: actual distance opponent moved to return shot
                    AVG(CASE
                        WHEN racket_hit_player_id = 1 AND next_player_2_x IS NOT NULL AND next_player_2_y IS NOT NULL THEN
                            SQRT(POW(next_player_2_x - player_2_x_meter, 2) + POW(next_player_2_y - player_2_y_meter, 2))
                        WHEN racket_hit_player_id = 2 AND next_player_1_x IS NOT NULL AND next_player_1_y IS NOT NULL THEN
                            SQRT(POW(next_player_1_x - player_1_x_meter, 2) + POW(next_player_1_y - player_1_y_meter, 2))
                    END) as avg_opponent_dist_moved,
                    MAX(CASE
                        WHEN racket_hit_player_id = 1 AND next_player_2_x IS NOT NULL AND next_player_2_y IS NOT NULL THEN
                            SQRT(POW(next_player_2_x - player_2_x_meter, 2) + POW(next_player_2_y - player_2_y_meter, 2))
                        WHEN racket_hit_player_id = 2 AND next_player_1_x IS NOT NULL AND next_player_1_y IS NOT NULL THEN
                            SQRT(POW(next_player_1_x - player_1_x_meter, 2) + POW(next_player_1_y - player_1_y_meter, 2))
                    END) as max_opponent_dist_moved,
                    VARIANCE(CASE
                        WHEN racket_hit_player_id = 1 AND next_player_2_x IS NOT NULL AND next_player_2_y IS NOT NULL THEN
                            SQRT(POW(next_player_2_x - player_2_x_meter, 2) + POW(next_player_2_y - player_2_y_meter, 2))
                        WHEN racket_hit_player_id = 2 AND next_player_1_x IS NOT NULL AND next_player_1_y IS NOT NULL THEN
                            SQRT(POW(next_player_1_x - player_1_x_meter, 2) + POW(next_player_1_y - player_1_y_meter, 2))
                    END) as opponent_dist_moved_variance,
                    -- Depth dominance
                    AVG(CASE
                        WHEN racket_hit_player_id = 1 AND player_2_y_meter > player_1_y_meter THEN 1
                        WHEN racket_hit_player_id = 2 AND player_1_y_meter > player_2_y_meter THEN 1
                        ELSE 0
                    END) * 100 as depth_dominance_pct,
                    AVG(CASE
                        WHEN racket_hit_player_id = 1 THEN player_2_y_meter - player_1_y_meter
                        WHEN racket_hit_player_id = 2 THEN player_1_y_meter - player_2_y_meter
                    END) as avg_depth_diff,
                    MIN(CASE
                        WHEN racket_hit_player_id = 1 THEN player_2_y_meter - player_1_y_meter
                        WHEN racket_hit_player_id = 2 THEN player_1_y_meter - player_2_y_meter
                    END) as min_depth_diff,
                    MAX(CASE
                        WHEN racket_hit_player_id = 1 THEN player_2_y_meter - player_1_y_meter
                        WHEN racket_hit_player_id = 2 THEN player_1_y_meter - player_2_y_meter
                    END) as max_depth_diff,
                    -- Straight shot quality
                    COUNT(CASE WHEN shot_type IN ('straight_drive', 'straight_drop') THEN 1 END) as straight_shots,
                    COUNT(CASE
                        WHEN shot_type IN ('straight_drive', 'straight_drop')
                        AND (
                            (racket_hit_player_id = 1 AND LEAST(player_1_x_meter, 6.1 - player_1_x_meter) <= 1.2)
                            OR (racket_hit_player_id = 2 AND LEAST(player_2_x_meter, 6.1 - player_2_x_meter) <= 1.2)
                        )
                        THEN 1
                    END) as quality_straights
                FROM shot_with_response
                WHERE next_opponent_dist_from_t IS NOT NULL
                GROUP BY rally_id, racket_hit_player_id
            ),
            rally_game_mapping AS (
                SELECT f.rally_id, MIN(g.game_number) as game_number
                FROM frame_data f
                JOIN games g ON f.video_id = g.video_id
                    AND f.rally_id BETWEEN g.start_rally_id AND g.end_rally_id
                WHERE f.video_id = :video_id
                GROUP BY f.rally_id
            )
            SELECT
                rm.rally_id,
                rgm.game_number,
                rm.rally_start_time,
                rm.rally_duration,
                rm.shot_count,
                rm.point_winner,
                re.racket_hit_player_id,
                re.avg_displacement,
                re.max_displacement,
                re.displacement_variance,
                re.avg_opponent_dist_moved,
                re.max_opponent_dist_moved,
                re.opponent_dist_moved_variance,
                re.depth_dominance_pct,
                re.avg_depth_diff,
                re.min_depth_diff,
                re.max_depth_diff,
                re.straight_shots,
                re.quality_straights
            FROM rally_metadata rm
            LEFT JOIN rally_game_mapping rgm ON rm.rally_id = rgm.rally_id
            LEFT JOIN rally_effectiveness re ON rm.rally_id = re.rally_id
            ORDER BY rm.rally_id, re.racket_hit_player_id
        """
        )

        results = self.db.execute(query, params).fetchall()

        # Define metric builder function
        def build_shot_effectiveness(row):
            return SingleShotEffectivenessMetrics(
                avg_displacement_from_t=(
                    float(row.avg_displacement)
                    if row.avg_displacement is not None
                    else None
                ),
                max_displacement_from_t=(
                    float(row.max_displacement)
                    if row.max_displacement is not None
                    else None
                ),
                displacement_variance=(
                    float(row.displacement_variance)
                    if row.displacement_variance is not None
                    else None
                ),
                avg_opponent_distance_moved=(
                    float(row.avg_opponent_dist_moved)
                    if row.avg_opponent_dist_moved is not None
                    else None
                ),
                max_opponent_distance_moved=(
                    float(row.max_opponent_dist_moved)
                    if row.max_opponent_dist_moved is not None
                    else None
                ),
                opponent_distance_moved_variance=(
                    float(row.opponent_dist_moved_variance)
                    if row.opponent_dist_moved_variance is not None
                    else None
                ),
                depth_dominance_pct=(
                    float(row.depth_dominance_pct)
                    if row.depth_dominance_pct is not None
                    else None
                ),
                avg_depth_difference=(
                    float(row.avg_depth_diff) if row.avg_depth_diff is not None else None
                ),
                min_depth_difference=(
                    float(row.min_depth_diff) if row.min_depth_diff is not None else None
                ),
                max_depth_difference=(
                    float(row.max_depth_diff) if row.max_depth_diff is not None else None
                ),
                straight_shot_quality_pct=(
                    (float(row.quality_straights) / row.straight_shots * 100)
                    if row.straight_shots > 0
                    else None
                ),
                straight_shots_count=(
                    int(row.straight_shots) if row.straight_shots else 0
                ),
                shots_close_to_wall=(
                    int(row.quality_straights) if row.quality_straights else 0
                ),
            )

        # Pivot data
        rally_items_data = self._pivot_by_rally(results, build_shot_effectiveness)

        # Create empty metrics for missing players
        empty_shot_effectiveness = SingleShotEffectivenessMetrics(
            avg_displacement_from_t=None,
            max_displacement_from_t=None,
            displacement_variance=None,
            avg_opponent_distance_moved=None,
            max_opponent_distance_moved=None,
            opponent_distance_moved_variance=None,
            depth_dominance_pct=None,
            avg_depth_difference=None,
            min_depth_difference=None,
            max_depth_difference=None,
            straight_shot_quality_pct=None,
            straight_shots_count=0,
            shots_close_to_wall=0,
        )

        # Convert to response items, filling in empty metrics for missing players
        rally_items = []
        for item in rally_items_data:
            # Convert -1 point_winner to None (indicates unknown/not set)
            point_winner = (
                item["point_winner"] if item["point_winner"] not in [-1, None] else None
            )

            rally_items.append(
                ShotEffectivenessPerRallyItem(
                    rally_id=item["rally_id"],
                    game_number=item["game_number"],
                    rally_start_time=item["rally_start_time"],
                    rally_duration=item["rally_duration"],
                    shot_count=item["shot_count"],
                    point_winner=point_winner,
                    player_1=(
                        item["player_1"]
                        if item["player_1"] is not None
                        else empty_shot_effectiveness
                    ),
                    player_2=(
                        item["player_2"]
                        if item["player_2"] is not None
                        else empty_shot_effectiveness
                    ),
                )
            )

        return ShotEffectivenessPerRallyResponse(
            video_id=video_id,
            filters=filters,
            data=rally_items,
            total_rallies=len(rally_items),
        )

    # ============================================================================
    # WINNING EFFICIENCY PER-GAME AND PER-RALLY
    # ============================================================================

    def get_winning_efficiency_per_game(
        self, video_id: str, filters: AnalyticsFilters
    ) -> WinningEfficiencyPerGameResponse:
        """Get winning efficiency metrics per game with both players' data."""
        logger.info(f"Computing winning efficiency per-game for video {video_id}")
        self._check_processed(video_id)

        # Get game metadata
        games_metadata = self._get_games_metadata(video_id, filters)

        if not games_metadata:
            return WinningEfficiencyPerGameResponse(
                video_id=video_id, filters=filters, data=[], total_games=0
            )

        # Build WHERE clause
        where_sql, params = self._build_where_clause(
            video_id, filters, include_player_id=False
        )
        params["video_id"] = video_id

        # Query per-game winning efficiency grouped by game and player
        query = text(
            f"""
            WITH game_rally_stats AS (
                SELECT
                    g.game_number,
                    f.rally_id,
                    SUM(CASE WHEN f.is_racket_hit = TRUE AND f.racket_hit_player_id = 1 THEN 1 ELSE 0 END) as p1_shots,
                    SUM(CASE WHEN f.is_racket_hit = TRUE AND f.racket_hit_player_id = 2 THEN 1 ELSE 0 END) as p2_shots,
                    MAX(CASE WHEN f.point_winner = 1 THEN 1 ELSE 0 END) as p1_won,
                    MAX(CASE WHEN f.point_winner = 2 THEN 1 ELSE 0 END) as p2_won
                FROM games g
                JOIN frame_data f ON f.video_id = g.video_id
                    AND f.rally_id BETWEEN g.start_rally_id AND g.end_rally_id
                WHERE g.{where_sql}
                  AND f.rally_id IS NOT NULL
                GROUP BY g.game_number, f.rally_id
            )
            SELECT
                game_number,
                1 as racket_hit_player_id,
                SUM(p1_shots) as total_shots,
                SUM(p1_won) as points_won,
                COUNT(*) as rallies_played,
                SUM(CASE WHEN p1_won = 1 THEN p1_shots ELSE 0 END) as shots_in_won_rallies
            FROM game_rally_stats
            GROUP BY game_number
            UNION ALL
            SELECT
                game_number,
                2 as racket_hit_player_id,
                SUM(p2_shots) as total_shots,
                SUM(p2_won) as points_won,
                COUNT(*) as rallies_played,
                SUM(CASE WHEN p2_won = 1 THEN p2_shots ELSE 0 END) as shots_in_won_rallies
            FROM game_rally_stats
            GROUP BY game_number
            ORDER BY game_number, racket_hit_player_id
        """
        )

        results = self.db.execute(query, params).fetchall()

        # Define metric builder function
        def build_winning_efficiency(row):
            shots_per_point = (
                (row.shots_in_won_rallies / row.points_won)
                if row.points_won > 0
                else 0.0
            )
            win_rate = (
                (row.points_won / row.rallies_played * 100)
                if row.rallies_played > 0
                else 0.0
            )

            return WinningEfficiencyData(
                shots_per_point_won=float(shots_per_point),
                points_won=int(row.points_won) if row.points_won else 0,
                total_shots=int(row.total_shots) if row.total_shots else 0,
                win_rate=float(win_rate),
                rallies_played=int(row.rallies_played) if row.rallies_played else 0,
            )

        # Pivot data
        game_items_data = self._pivot_by_game(
            results, games_metadata, build_winning_efficiency
        )

        # Create empty metrics for missing players
        empty_winning_efficiency = WinningEfficiencyData(
            shots_per_point_won=0.0,
            points_won=0,
            total_shots=0,
            win_rate=0.0,
            rallies_played=0,
        )

        # Convert to response items, filling in empty metrics for missing players
        game_items = []
        for item in game_items_data:
            game_items.append(
                WinningEfficiencyPerGameItem(
                    game_number=item["game_number"],
                    start_rally_id=item["start_rally_id"],
                    end_rally_id=item["end_rally_id"],
                    start_time=item["start_time"],
                    end_time=item["end_time"],
                    duration=item["duration"],
                    rally_count=item["rally_count"],
                    player_1=(
                        item["player_1"]
                        if item["player_1"] is not None
                        else empty_winning_efficiency
                    ),
                    player_2=(
                        item["player_2"]
                        if item["player_2"] is not None
                        else empty_winning_efficiency
                    ),
                )
            )

        return WinningEfficiencyPerGameResponse(
            video_id=video_id,
            filters=filters,
            data=game_items,
            total_games=len(game_items),
        )

    def get_winning_efficiency_per_rally(
        self, video_id: str, filters: AnalyticsFilters
    ) -> WinningEfficiencyPerRallyResponse:
        """Get winning efficiency metrics per rally with both players' data."""
        logger.info(f"Computing winning efficiency per-rally for video {video_id}")
        self._check_processed(video_id)

        # Build WHERE clause
        where_sql, params = self._build_where_clause(
            video_id, filters, include_player_id=False
        )
        params["video_id"] = video_id

        # Query per-rally winning efficiency grouped by rally and player
        query = text(
            f"""
            WITH rally_stats AS (
                SELECT DISTINCT
                    f.rally_id,
                    MIN(f.timestamp) as rally_start_time,
                    MAX(f.timestamp) - MIN(f.timestamp) as rally_duration,
                    COUNT(CASE WHEN f.is_racket_hit = TRUE THEN 1 END) as shot_count,
                    MAX(f.point_winner) as point_winner,
                    SUM(CASE WHEN f.is_racket_hit = TRUE AND f.racket_hit_player_id = 1 THEN 1 ELSE 0 END) as p1_shots,
                    SUM(CASE WHEN f.is_racket_hit = TRUE AND f.racket_hit_player_id = 2 THEN 1 ELSE 0 END) as p2_shots,
                    MAX(CASE WHEN f.point_winner = 1 THEN 1 ELSE 0 END) as p1_won,
                    MAX(CASE WHEN f.point_winner = 2 THEN 1 ELSE 0 END) as p2_won
                FROM frame_data f
                WHERE {where_sql}
                  AND f.rally_id IS NOT NULL
                GROUP BY f.rally_id
            ),
            rally_game_mapping AS (
                SELECT f.rally_id, MIN(g.game_number) as game_number
                FROM frame_data f
                JOIN games g ON f.video_id = g.video_id
                    AND f.rally_id BETWEEN g.start_rally_id AND g.end_rally_id
                WHERE f.video_id = :video_id
                GROUP BY f.rally_id
            )
            SELECT
                rs.rally_id,
                rs.rally_start_time,
                rs.rally_duration,
                rs.shot_count,
                rs.point_winner,
                rgm.game_number,
                1 as racket_hit_player_id,
                rs.p1_shots as player_shots,
                rs.p1_won as won_point
            FROM rally_stats rs
            LEFT JOIN rally_game_mapping rgm ON rs.rally_id = rgm.rally_id
            UNION ALL
            SELECT
                rs.rally_id,
                rs.rally_start_time,
                rs.rally_duration,
                rs.shot_count,
                rs.point_winner,
                rgm.game_number,
                2 as racket_hit_player_id,
                rs.p2_shots as player_shots,
                rs.p2_won as won_point
            FROM rally_stats rs
            LEFT JOIN rally_game_mapping rgm ON rs.rally_id = rgm.rally_id
            ORDER BY rally_id, racket_hit_player_id
        """
        )

        results = self.db.execute(query, params).fetchall()

        # Define metric builder function
        def build_winning_efficiency(row):
            shots_per_point = float(row.player_shots) if row.won_point == 1 else 0.0
            win_rate = 100.0 if row.won_point == 1 else 0.0

            return WinningEfficiencyData(
                shots_per_point_won=shots_per_point,
                points_won=int(row.won_point) if row.won_point else 0,
                total_shots=int(row.player_shots) if row.player_shots else 0,
                win_rate=win_rate,
                rallies_played=1,  # This is per-rally, so always 1
            )

        # Pivot data
        rally_items_data = self._pivot_by_rally(results, build_winning_efficiency)

        # Create empty metrics for missing players
        empty_winning_efficiency = WinningEfficiencyData(
            shots_per_point_won=0.0,
            points_won=0,
            total_shots=0,
            win_rate=0.0,
            rallies_played=0,
        )

        # Convert to response items, filling in empty metrics for missing players
        rally_items = []
        for item in rally_items_data:
            # Convert -1 point_winner to None (indicates unknown/not set)
            point_winner = (
                item["point_winner"] if item["point_winner"] not in [-1, None] else None
            )

            rally_items.append(
                WinningEfficiencyPerRallyItem(
                    rally_id=item["rally_id"],
                    game_number=item["game_number"],
                    rally_start_time=item["rally_start_time"],
                    rally_duration=item["rally_duration"],
                    shot_count=item["shot_count"],
                    point_winner=point_winner,
                    player_1=(
                        item["player_1"]
                        if item["player_1"] is not None
                        else empty_winning_efficiency
                    ),
                    player_2=(
                        item["player_2"]
                        if item["player_2"] is not None
                        else empty_winning_efficiency
                    ),
                )
            )

        return WinningEfficiencyPerRallyResponse(
            video_id=video_id,
            filters=filters,
            data=rally_items,
            total_rallies=len(rally_items),
        )

    # ============================================================================
    # RALLY INTENSITY PER-GAME AND PER-RALLY
    # ============================================================================

    def get_rally_intensity_per_game(
        self, video_id: str, filters: AnalyticsFilters
    ) -> RallyIntensityPerGameResponse:
        """Get rally intensity metrics per game (not player-specific)."""
        logger.info(f"Computing rally intensity per-game for video {video_id}")
        self._check_processed(video_id)

        # Get game metadata
        games_metadata = self._get_games_metadata(video_id, filters)

        if not games_metadata:
            return RallyIntensityPerGameResponse(
                video_id=video_id, filters=filters, data=[], total_games=0
            )

        # Build WHERE clause
        where_sql, params = self._build_where_clause(
            video_id, filters, include_player_id=False
        )

        # Query per-game rally intensity
        query = text(
            f"""
            WITH rally_intensity AS (
                SELECT
                    g.game_number,
                    f.rally_id,
                    (MAX(f.timestamp) - MIN(f.timestamp)) / NULLIF(COUNT(CASE WHEN f.is_racket_hit = TRUE THEN 1 END), 0) as seconds_per_shot
                FROM games g
                JOIN frame_data f ON f.video_id = g.video_id
                    AND f.rally_id BETWEEN g.start_rally_id AND g.end_rally_id
                WHERE g.{where_sql}
                  AND f.rally_id IS NOT NULL
                GROUP BY g.game_number, f.rally_id
            )
            SELECT
                game_number,
                AVG(seconds_per_shot) as avg_seconds_per_shot,
                MIN(seconds_per_shot) as min_seconds_per_shot,
                MAX(seconds_per_shot) as max_seconds_per_shot,
                STDDEV_POP(seconds_per_shot) as std_dev,
                COUNT(*) as rally_count
            FROM rally_intensity
            WHERE seconds_per_shot IS NOT NULL
            GROUP BY game_number
            ORDER BY game_number
        """
        )

        results = self.db.execute(query, params).fetchall()

        # Build response items
        game_items = []
        for game in games_metadata:
            game_number = game["game_number"]
            game_row = next((r for r in results if r.game_number == game_number), None)

            if game_row:
                intensity_data = RallyIntensityData(
                    avg_seconds_per_shot=(
                        float(game_row.avg_seconds_per_shot)
                        if game_row.avg_seconds_per_shot
                        else 0.0
                    ),
                    min_seconds_per_shot=(
                        float(game_row.min_seconds_per_shot)
                        if game_row.min_seconds_per_shot
                        else 0.0
                    ),
                    max_seconds_per_shot=(
                        float(game_row.max_seconds_per_shot)
                        if game_row.max_seconds_per_shot
                        else 0.0
                    ),
                    std_dev=float(game_row.std_dev) if game_row.std_dev else 0.0,
                    rally_count=(
                        int(game_row.rally_count) if game_row.rally_count else 0
                    ),
                )
            else:
                intensity_data = RallyIntensityData(
                    avg_seconds_per_shot=0.0,
                    min_seconds_per_shot=0.0,
                    max_seconds_per_shot=0.0,
                    std_dev=0.0,
                    rally_count=0,
                )

            game_items.append(
                RallyIntensityPerGameItem(
                    game_number=game_number,
                    start_rally_id=game["start_rally_id"],
                    end_rally_id=game["end_rally_id"],
                    start_time=game["start_time"],
                    end_time=game["end_time"],
                    duration=game["duration"],
                    rally_count=game["rally_count"],
                    data=intensity_data,
                )
            )

        return RallyIntensityPerGameResponse(
            video_id=video_id,
            filters=filters,
            data=game_items,
            total_games=len(game_items),
        )

    def get_rally_intensity_per_rally(
        self, video_id: str, filters: AnalyticsFilters
    ) -> RallyIntensityPerRallyResponse:
        """Get rally intensity metrics per rally (not player-specific)."""
        logger.info(f"Computing rally intensity per-rally for video {video_id}")
        self._check_processed(video_id)

        # Build WHERE clause
        where_sql, params = self._build_where_clause(
            video_id, filters, include_player_id=False
        )
        params["video_id"] = video_id

        # Query per-rally intensity
        query = text(
            f"""
            WITH rally_intensity AS (
                SELECT
                    f.rally_id,
                    MIN(f.timestamp) as rally_start_time,
                    MAX(f.timestamp) - MIN(f.timestamp) as rally_duration,
                    COUNT(CASE WHEN f.is_racket_hit = TRUE THEN 1 END) as shot_count,
                    MAX(f.point_winner) as point_winner,
                    (MAX(f.timestamp) - MIN(f.timestamp)) / NULLIF(COUNT(CASE WHEN f.is_racket_hit = TRUE THEN 1 END), 0) as seconds_per_shot
                FROM frame_data f
                WHERE {where_sql}
                  AND f.rally_id IS NOT NULL
                GROUP BY f.rally_id
            ),
            rally_game_mapping AS (
                SELECT f.rally_id, MIN(g.game_number) as game_number
                FROM frame_data f
                JOIN games g ON f.video_id = g.video_id
                    AND f.rally_id BETWEEN g.start_rally_id AND g.end_rally_id
                WHERE f.video_id = :video_id
                GROUP BY f.rally_id
            )
            SELECT ri.*, rgm.game_number
            FROM rally_intensity ri
            LEFT JOIN rally_game_mapping rgm ON ri.rally_id = rgm.rally_id
            ORDER BY ri.rally_id
        """
        )

        results = self.db.execute(query, params).fetchall()

        # Build response items
        rally_items = []
        for row in results:
            # Convert -1 point_winner to None (indicates unknown/not set)
            point_winner = (
                row.point_winner if row.point_winner not in [-1, None] else None
            )

            rally_items.append(
                RallyIntensityPerRallyItem(
                    rally_id=row.rally_id,
                    game_number=row.game_number,
                    rally_start_time=row.rally_start_time,
                    rally_duration=row.rally_duration,
                    shot_count=row.shot_count,
                    point_winner=point_winner,
                    seconds_per_shot=(
                        float(row.seconds_per_shot) if row.seconds_per_shot else None
                    ),
                )
            )

        return RallyIntensityPerRallyResponse(
            video_id=video_id,
            filters=filters,
            data=rally_items,
            total_rallies=len(rally_items),
        )

    # ============================================================================
    # MATCH HIGHLIGHTS
    # ============================================================================

    def get_longest_rally(
        self, video_id: str, filters: AnalyticsFilters
    ) -> LongestRallyResponse:
        """Get the longest rally in the match by duration and shot count."""
        where_sql, params = self._build_where_clause(video_id, filters)
        params["video_id"] = video_id

        query = text(
            f"""
            WITH rally_stats AS (
                SELECT
                    rally_id,
                    MIN(timestamp) as rally_start_time,
                    MAX(timestamp) - MIN(timestamp) as rally_duration,
                    COUNT(CASE WHEN is_racket_hit = TRUE THEN 1 END) as shot_count,
                    MAX(point_winner) as point_winner
                FROM frame_data
                WHERE {where_sql} AND rally_id IS NOT NULL
                GROUP BY rally_id
            ),
            rally_game_mapping AS (
                SELECT f.rally_id, MIN(g.game_number) as game_number
                FROM frame_data f
                JOIN games g ON f.video_id = g.video_id
                    AND f.rally_id BETWEEN g.start_rally_id AND g.end_rally_id
                WHERE f.video_id = :video_id
                GROUP BY f.rally_id
            ),
            longest_rally AS (
                SELECT rally_id, rally_start_time, rally_duration, shot_count, point_winner
                FROM rally_stats
                ORDER BY rally_duration DESC, shot_count DESC
                LIMIT 1
            )
            SELECT lr.*, rgm.game_number
            FROM longest_rally lr
            LEFT JOIN rally_game_mapping rgm ON lr.rally_id = rgm.rally_id
        """
        )

        result = self.db.execute(query, params).fetchone()

        if not result:
            raise ValueError(f"No rallies found for video {video_id}")

        # Convert -1 point_winner to None
        point_winner = result.point_winner if result.point_winner not in [-1, None] else None

        data = LongestRallyData(
            rally_id=result.rally_id,
            game_number=result.game_number,
            rally_start_time=float(result.rally_start_time),
            rally_duration=float(result.rally_duration),
            shot_count=result.shot_count,
            point_winner=point_winner,
        )

        return LongestRallyResponse(video_id=video_id, filters=filters, data=data)

    def get_fastest_shot(
        self, video_id: str, filters: AnalyticsFilters
    ) -> FastestShotResponse:
        """Get the fastest shot in the match."""
        where_sql, params = self._build_where_clause(video_id, filters)
        params["video_id"] = video_id

        query = text(
            f"""
            WITH fastest_shot AS (
                SELECT
                    frame_number,
                    timestamp,
                    rally_id,
                    racket_hit_player_id as player_id,
                    ball_speed,
                    stroke_type,
                    shot_type
                FROM frame_data
                WHERE {where_sql}
                  AND is_racket_hit = TRUE
                  AND ball_speed IS NOT NULL
                  AND racket_hit_player_id IN (1, 2)
                ORDER BY ball_speed DESC
                LIMIT 1
            ),
            rally_game_mapping AS (
                SELECT f.rally_id, MIN(g.game_number) as game_number
                FROM frame_data f
                JOIN games g ON f.video_id = g.video_id
                    AND f.rally_id BETWEEN g.start_rally_id AND g.end_rally_id
                WHERE f.video_id = :video_id
                GROUP BY f.rally_id
            )
            SELECT fs.*, rgm.game_number
            FROM fastest_shot fs
            LEFT JOIN rally_game_mapping rgm ON fs.rally_id = rgm.rally_id
        """
        )

        result = self.db.execute(query, params).fetchone()

        if not result:
            raise ValueError(f"No shots with ball speed found for video {video_id}")

        data = FastestShotData(
            frame_number=result.frame_number,
            timestamp=float(result.timestamp),
            rally_id=result.rally_id,
            game_number=result.game_number,
            player_id=result.player_id,
            ball_speed=float(result.ball_speed),
            stroke_type=result.stroke_type,
            shot_type=result.shot_type,
        )

        return FastestShotResponse(video_id=video_id, filters=filters, data=data)
