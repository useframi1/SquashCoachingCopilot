"""Analysis service for querying processed video data - PostgreSQL optimized."""

import logging
from typing import Dict, List, Tuple

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.models.frame_data import FrameData
from backend.models.video import Video
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
    ShotPlacementData,
    WinningStatsData,
    RallyIntensityData,
    # Pattern 3: Spatial
    HeatmapGrid,
    SpatialData,
    # Pattern 5: Extended
    SingleMovementMetrics,
    SingleTZoneMetrics,
    SingleShotEffectivenessMetrics,
    # Response schemas
    StrokeDistributionResponse,
    ShotTypeDistributionResponse,
    BallSpeedResponse,
    RhythmDisruptionResponse,
    PlayerPositionHeatmapResponse,
    ShotPlacementResponse,
    CourtQuadrantResponse,
    WallHitHeatmapResponse,
    WallQuadrantResponse,
    WinningStatsResponse,
    # Extended analytics schemas
    MovementMetricsResponse,
    TZoneOccupancyResponse,
    ShotEffectivenessResponse,
    RallyIntensityResponse,
)

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

        if filters.rally_id is not None:
            where_clauses.append("rally_id = :rally_id")
            params["rally_id"] = filters.rally_id

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
            ball_speed_variance=float(result.ball_speed_variance) if result.ball_speed_variance else 0.0,
            wall_hit_height_cv=float(result.wall_hit_height_cv) if result.wall_hit_height_cv else 0.0,
            wall_hit_height_variance=float(result.wall_hit_height_variance) if result.wall_hit_height_variance else 0.0,
            shot_count=int(result.shot_count) if result.shot_count else 0,
        )

        return RhythmDisruptionResponse(
            video_id=video_id, filters=filters, data=data
        )

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

    def get_shot_placement_effectiveness(
        self, video_id: str, player_id: int, filters: AnalyticsFilters
    ) -> ShotPlacementResponse:
        """Analyze shot placement effectiveness - PostgreSQL optimized.

        Measures average distance opponent had to move after each shot.
        Greater distance indicates more effective shot placement.
        """
        logger.info(f"Computing shot placement effectiveness for player {player_id}")
        self._check_processed(video_id)

        where_sql, params = self._build_where_clause(video_id, filters)
        params["player_id"] = player_id

        # Use PostgreSQL window functions to calculate opponent distance moved
        opponent_id = 2 if player_id == 1 else 1
        opp_x = f"player_{opponent_id}_x_meter"
        opp_y = f"player_{opponent_id}_y_meter"

        query = text(
            f"""
            WITH shot_positions AS (
                SELECT
                    {opp_x} as opp_x_before,
                    {opp_y} as opp_y_before,
                    LEAD({opp_x}) OVER (ORDER BY frame_number) as opp_x_after,
                    LEAD({opp_y}) OVER (ORDER BY frame_number) as opp_y_after
                FROM frame_data
                WHERE {where_sql}
                  AND is_racket_hit = TRUE
                  AND racket_hit_player_id = :player_id
            ),
            distances AS (
                SELECT
                    SQRT(
                        POW(opp_x_after - opp_x_before, 2) +
                        POW(opp_y_after - opp_y_before, 2)
                    ) as distance_moved
                FROM shot_positions
                WHERE opp_x_before IS NOT NULL
                  AND opp_y_before IS NOT NULL
                  AND opp_x_after IS NOT NULL
                  AND opp_y_after IS NOT NULL
            )
            SELECT
                AVG(distance_moved) as avg_distance,
                MIN(distance_moved) as min_distance,
                MAX(distance_moved) as max_distance,
                STDDEV_POP(distance_moved) as std_dev,
                COUNT(*) as count
            FROM distances
        """
        )

        result = self.db.execute(query, params).fetchone()

        data = ShotPlacementData(
            avg_opponent_distance_moved=float(result.avg_distance) if result.avg_distance else 0.0,
            min_opponent_distance_moved=float(result.min_distance) if result.min_distance else 0.0,
            max_opponent_distance_moved=float(result.max_distance) if result.max_distance else 0.0,
            std_dev=float(result.std_dev) if result.std_dev else 0.0,
            shot_count=int(result.count) if result.count else 0,
        )

        return ShotPlacementResponse(video_id=video_id, filters=filters, data=data)

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

    def get_winning_stats(
        self, video_id: str, player_id: int, filters: AnalyticsFilters
    ) -> WinningStatsResponse:
        """Calculate winning statistics for a specific player - PostgreSQL optimized.

        Returns aggregate winning efficiency metrics (points won per shot ratio).
        """
        logger.info(f"Computing winning stats for player {player_id}")
        self._check_processed(video_id)

        where_sql, params = self._build_where_clause(video_id, filters)
        params["player_id"] = player_id

        # Aggregate query for a single player
        query = text(
            f"""
            SELECT
                SUM(CASE WHEN is_racket_hit = TRUE AND racket_hit_player_id = :player_id THEN 1 ELSE 0 END) as total_shots,
                SUM(CASE WHEN point_winner = :player_id THEN 1 ELSE 0 END) as points_won,
                COUNT(DISTINCT rally_id) as rallies_played
            FROM frame_data
            WHERE {where_sql}
              AND rally_id IS NOT NULL
        """
        )

        result = self.db.execute(query, params).fetchone()

        total_shots = int(result.total_shots) if result.total_shots else 0
        points_won = int(result.points_won) if result.points_won else 0
        rallies_played = int(result.rallies_played) if result.rallies_played else 0

        efficiency = (points_won / total_shots) if total_shots > 0 else 0.0
        points_per_rally = (points_won / rallies_played) if rallies_played > 0 else 0.0

        data = WinningStatsData(
            efficiency=efficiency,
            points_won=points_won,
            total_shots=total_shots,
            points_per_rally=points_per_rally,
            rallies_played=rallies_played,
        )

        return WinningStatsResponse(video_id=video_id, filters=filters, data=data)

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
                        {player_col_x},
                        {player_col_y},
                        {opponent_col_x},
                        {opponent_col_y},
                        is_racket_hit,
                        racket_hit_player_id,
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
                shot_frames AS (
                    SELECT
                        rally_id,
                        frame_number,
                        racket_hit_player_id,
                        {player_col_x} as player_x,
                        {player_col_y} as player_y,
                        {opponent_col_x} as opponent_x,
                        {opponent_col_y} as opponent_y
                    FROM frame_distances
                    WHERE is_racket_hit = TRUE
                ),
                distance_to_ball AS (
                    SELECT
                        curr.rally_id,
                        curr.frame_number,
                        SQRT(
                            POW(curr.player_x - prev.opponent_x, 2) +
                            POW(curr.player_y - prev.opponent_y, 2)
                        ) as distance_to_ball
                    FROM shot_frames curr
                    LEFT JOIN LATERAL (
                        SELECT opponent_x, opponent_y
                        FROM shot_frames prev
                        WHERE prev.rally_id = curr.rally_id
                          AND prev.frame_number < curr.frame_number
                          AND prev.racket_hit_player_id != curr.racket_hit_player_id
                        ORDER BY prev.frame_number DESC
                        LIMIT 1
                    ) prev ON TRUE
                    WHERE curr.racket_hit_player_id = :player_id_param
                ),
                rally_totals AS (
                    SELECT
                        rally_id,
                        SUM(frame_distance) as rally_distance
                    FROM frame_distances
                    GROUP BY rally_id
                )
                SELECT
                    SUM(rt.rally_distance) as total_distance,
                    AVG(rt.rally_distance) as avg_distance_per_rally,
                    AVG(dtb.distance_to_ball) as avg_distance_to_ball,
                    MIN(dtb.distance_to_ball) as min_distance_to_ball,
                    MAX(dtb.distance_to_ball) as max_distance_to_ball,
                    COUNT(dtb.distance_to_ball) as shot_count
                FROM rally_totals rt
                LEFT JOIN distance_to_ball dtb ON rt.rally_id = dtb.rally_id
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
                        player_1_x_meter,
                        player_1_y_meter,
                        player_2_x_meter,
                        player_2_y_meter,
                        is_racket_hit,
                        racket_hit_player_id,
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
                shot_frames AS (
                    SELECT
                        rally_id,
                        frame_number,
                        racket_hit_player_id,
                        player_1_x_meter,
                        player_1_y_meter,
                        player_2_x_meter,
                        player_2_y_meter
                    FROM frame_distances
                    WHERE is_racket_hit = TRUE
                ),
                distance_to_ball AS (
                    SELECT
                        curr.rally_id,
                        curr.frame_number,
                        curr.racket_hit_player_id,
                        -- P1 distance to ball
                        CASE
                            WHEN curr.racket_hit_player_id = 1 AND prev.player_2_x_meter IS NOT NULL THEN
                                SQRT(
                                    POW(curr.player_1_x_meter - prev.player_2_x_meter, 2) +
                                    POW(curr.player_1_y_meter - prev.player_2_y_meter, 2)
                                )
                            ELSE NULL
                        END as p1_distance_to_ball,
                        -- P2 distance to ball
                        CASE
                            WHEN curr.racket_hit_player_id = 2 AND prev.player_1_x_meter IS NOT NULL THEN
                                SQRT(
                                    POW(curr.player_2_x_meter - prev.player_1_x_meter, 2) +
                                    POW(curr.player_2_y_meter - prev.player_1_y_meter, 2)
                                )
                            ELSE NULL
                        END as p2_distance_to_ball
                    FROM shot_frames curr
                    LEFT JOIN LATERAL (
                        SELECT player_1_x_meter, player_1_y_meter, player_2_x_meter, player_2_y_meter
                        FROM shot_frames prev
                        WHERE prev.rally_id = curr.rally_id
                          AND prev.frame_number < curr.frame_number
                          AND prev.racket_hit_player_id != curr.racket_hit_player_id
                        ORDER BY prev.frame_number DESC
                        LIMIT 1
                    ) prev ON TRUE
                ),
                rally_totals AS (
                    SELECT
                        rally_id,
                        SUM(p1_frame_distance) + SUM(p2_frame_distance) as combined_rally_distance
                    FROM frame_distances
                    GROUP BY rally_id
                ),
                all_distances_to_ball AS (
                    SELECT distance_to_ball
                    FROM (
                        SELECT p1_distance_to_ball as distance_to_ball FROM distance_to_ball WHERE p1_distance_to_ball IS NOT NULL
                        UNION ALL
                        SELECT p2_distance_to_ball as distance_to_ball FROM distance_to_ball WHERE p2_distance_to_ball IS NOT NULL
                    ) combined
                )
                SELECT
                    SUM(rt.combined_rally_distance) as total_distance,
                    AVG(rt.combined_rally_distance) as avg_distance_per_rally,
                    AVG(dtb.distance_to_ball) as avg_distance_to_ball,
                    MIN(dtb.distance_to_ball) as min_distance_to_ball,
                    MAX(dtb.distance_to_ball) as max_distance_to_ball,
                    COUNT(dtb.distance_to_ball) as shot_count
                FROM rally_totals rt
                CROSS JOIN all_distances_to_ball dtb
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
            avg_distance_to_ball=(
                float(result.avg_distance_to_ball)
                if result and result.avg_distance_to_ball
                else 0.0
            ),
            min_distance_to_ball=(
                float(result.min_distance_to_ball)
                if result and result.min_distance_to_ball
                else None
            ),
            max_distance_to_ball=(
                float(result.max_distance_to_ball)
                if result and result.max_distance_to_ball
                else None
            ),
            shot_count=int(result.shot_count) if result and result.shot_count else 0,
        )

        return MovementMetricsResponse(
            video_id=video_id, filters=filters, data=movement_metrics
        )

    def get_t_zone_occupancy(
        self, video_id: str, filters: AnalyticsFilters
    ) -> TZoneOccupancyResponse:
        """Calculate T-zone occupancy metrics - PostgreSQL optimized.

        Returns aggregated totals if player_id filter is not specified,
        otherwise returns data for the specified player only.

        Note: This method requires complex state tracking (T-zone entry/exit logic)
        that's difficult to express in pure SQL, so minimal Python processing is used.
        """
        logger.info(f"Computing T-zone occupancy for video {video_id}")
        self._check_processed(video_id)

        # Build WHERE clause WITHOUT player_id filter (we handle it separately)
        where_sql, params = self._build_where_clause(
            video_id, filters, include_player_id=False
        )

        # Define T-zone parameters (standard squash court)
        T_X = 3.05  # meters (half of 6.1m court width)
        T_Y = 5.44  # meters
        T_RADIUS = 1.2  # meters

        # Compute T-zone occupancy in SQL
        query = text(
            f"""
            SELECT
                (SQRT(POW(player_1_x_meter - {T_X}, 2) + POW(player_1_y_meter - {T_Y}, 2)) <= {T_RADIUS}) as p1_in_t_zone,
                (SQRT(POW(player_2_x_meter - {T_X}, 2) + POW(player_2_y_meter - {T_Y}, 2)) <= {T_RADIUS}) as p2_in_t_zone,
                is_racket_hit,
                racket_hit_player_id,
                timestamp,
                rally_id
            FROM frame_data
            WHERE {where_sql}
              AND is_rally_frame = TRUE
              AND player_1_x_meter IS NOT NULL
              AND player_1_y_meter IS NOT NULL
              AND player_2_x_meter IS NOT NULL
              AND player_2_y_meter IS NOT NULL
            ORDER BY frame_number
        """
        )

        results = self.db.execute(query, params).fetchall()

        # Calculate % time in T
        total_frames = len(results)
        p1_frames_in_t = sum(1 for r in results if r.p1_in_t_zone)
        p2_frames_in_t = sum(1 for r in results if r.p2_in_t_zone)

        p1_pct_time_in_t = (
            (p1_frames_in_t / total_frames * 100) if total_frames > 0 else 0.0
        )
        p2_pct_time_in_t = (
            (p2_frames_in_t / total_frames * 100) if total_frames > 0 else 0.0
        )

        # Calculate time-to-T metrics (requires state tracking)
        # Time-to-T measures from player's own shot to when they return to T
        p1_time_to_t = []
        p2_time_to_t = []
        p1_total_shots = 0
        p2_total_shots = 0
        p1_successful_returns = 0
        p2_successful_returns = 0

        last_p1_shot_time = None
        last_p2_shot_time = None
        p1_entered_t_after_shot = False
        p2_entered_t_after_shot = False

        for frame in results:
            if frame.is_racket_hit:
                if frame.racket_hit_player_id == 1:
                    last_p1_shot_time = frame.timestamp
                    p1_total_shots += 1
                    p1_entered_t_after_shot = (
                        False  # Reset P1's own tracking after P1 hits
                    )
                elif frame.racket_hit_player_id == 2:
                    last_p2_shot_time = frame.timestamp
                    p2_total_shots += 1
                    p2_entered_t_after_shot = (
                        False  # Reset P2's own tracking after P2 hits
                    )

            # P1 returns to T after P1's own shot
            if (
                last_p1_shot_time is not None
                and not p1_entered_t_after_shot
                and frame.p1_in_t_zone
            ):
                time_to_t = frame.timestamp - last_p1_shot_time
                p1_time_to_t.append(time_to_t)
                p1_entered_t_after_shot = True
                p1_successful_returns += 1

            # P2 returns to T after P2's own shot
            if (
                last_p2_shot_time is not None
                and not p2_entered_t_after_shot
                and frame.p2_in_t_zone
            ):
                time_to_t = frame.timestamp - last_p2_shot_time
                p2_time_to_t.append(time_to_t)
                p2_entered_t_after_shot = True
                p2_successful_returns += 1

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
                next_shot.opponent_distance_from_t as next_opponent_dist_from_t
            FROM shot_frames curr
            LEFT JOIN LATERAL (
                SELECT opponent_distance_from_t
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
        depth = np.array(depth_diffs) if depth_diffs else np.array([])

        metrics = SingleShotEffectivenessMetrics(
            avg_displacement_from_t=float(displ.mean()) if len(displ) > 0 else None,
            max_displacement_from_t=float(displ.max()) if len(displ) > 0 else None,
            displacement_variance=float(displ.var()) if len(displ) > 0 else None,
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
            avg_seconds_per_shot=float(result.avg_sec_per_shot) if result.avg_sec_per_shot else 0.0,
            min_seconds_per_shot=float(result.min_sec_per_shot) if result.min_sec_per_shot else 0.0,
            max_seconds_per_shot=float(result.max_sec_per_shot) if result.max_sec_per_shot else 0.0,
            std_dev=float(result.std_dev) if result.std_dev else 0.0,
            rally_count=int(result.rally_count) if result.rally_count else 0,
        )

        return RallyIntensityResponse(
            video_id=video_id, filters=filters, data=data
        )
