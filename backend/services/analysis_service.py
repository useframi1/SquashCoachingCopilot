"""Analysis service for querying processed video data."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import case, func, literal
from sqlalchemy.orm import Session, Query

from backend.models.frame_data import FrameData
from backend.models.video import Video
from backend.schemas.analysis import (
    # Base schemas
    AnalyticsFilters,
    # Pattern 1: Distribution
    DistributionItem,
    PlayerDistribution,
    DistributionData,
    # Pattern 2: Aggregate
    PlayerAggregates,
    AggregateData,
    # Pattern 3: Spatial
    HeatmapGrid,
    SpatialData,
    # Pattern 4: Comparative
    RallyItem,
    RallyStatsData,
    RhythmRallyItem,
    RhythmDisruptionData,
    WinningRallyItem,
    WinningStatsData,
    ShotPlacementItem,
    ShotPlacementData,
    # Response schemas
    StrokeDistributionResponse,
    ShotTypeDistributionResponse,
    BallSpeedResponse,
    RhythmDisruptionResponse,
    PlayerPositionHeatmapResponse,
    ShotPlacementResponse,
    CourtQuadrantResponse,
    RallyStatsResponse,
    WallHitHeatmapResponse,
    WallQuadrantResponse,
    WinningStatsResponse,
)

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service for querying and analyzing processed video data."""

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

    # ========================================================================
    # ANALYTICS METHODS
    # ========================================================================

    def _get_base_query(self, video_id: str, filters: AnalyticsFilters) -> Query:
        """Get base filtered query for analytics."""
        self._check_processed(video_id)
        query = self.db.query(FrameData).filter(FrameData.video_id == video_id)

        if filters.rally_id is not None:
            query = query.filter(FrameData.rally_id == filters.rally_id)
        if filters.start_time is not None:
            query = query.filter(FrameData.timestamp >= filters.start_time)
        if filters.end_time is not None:
            query = query.filter(FrameData.timestamp <= filters.end_time)

        return query

    # ========================================================================
    # HELPER METHODS FOR NEW SCHEMA PATTERNS
    # ========================================================================

    def _build_distribution_data(
        self,
        player_1_data: Dict[str, int],
        player_2_data: Dict[str, int],
    ) -> DistributionData:
        """Build distribution data for both players (always both present)."""

        def make_distribution(
            data: Dict[str, int], player_id: int
        ) -> PlayerDistribution:
            total = sum(data.values())
            items = [
                DistributionItem(
                    label=label,
                    count=count,
                    percentage=(count / total * 100) if total > 0 else 0.0,
                )
                for label, count in sorted(data.items())
            ]
            return PlayerDistribution(
                player_id=player_id, distribution=items, total=total
            )

        return DistributionData(
            player_1=make_distribution(player_1_data, 1),
            player_2=make_distribution(player_2_data, 2),
        )

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

    def _build_aggregate_data(
        self, player_1_stats: Tuple, player_2_stats: Tuple
    ) -> AggregateData:
        """Build aggregate data from SQL aggregation results."""
        # player_stats tuple: (player_id, mean, min, max, std_dev, count)

        def make_aggregates(stats: Optional[Tuple]) -> PlayerAggregates:
            if stats and len(stats) >= 6:
                return PlayerAggregates(
                    player_id=stats[0],
                    mean=stats[1] or 0.0,
                    min=stats[2] or 0.0,
                    max=stats[3] or 0.0,
                    std_dev=stats[4] or 0.0,
                    count=stats[5] or 0,
                )
            # Return empty aggregates if no data
            return PlayerAggregates(
                player_id=stats[0] if stats else 1,
                mean=0.0,
                min=0.0,
                max=0.0,
                std_dev=0.0,
                count=0,
            )

        return AggregateData(
            player_1=make_aggregates(player_1_stats),
            player_2=make_aggregates(player_2_stats),
        )

    # ========================================================================
    # ANALYTICS ENDPOINT METHODS
    # ========================================================================

    def get_stroke_distribution(
        self, video_id: str, filters: AnalyticsFilters
    ) -> StrokeDistributionResponse:
        """Get stroke distribution analytics (forehand vs backhand)."""
        logger.info(f"Computing stroke distribution for video {video_id}")

        query = self._get_base_query(video_id, filters)
        query = query.filter(FrameData.is_racket_hit == True)

        # Don't apply player filter to query - we always return both players
        # If player_id filter is specified, other player will have zero values

        # Count strokes per player
        results = (
            query.with_entities(
                FrameData.racket_hit_player_id,
                FrameData.stroke_type,
                func.count(FrameData.id).label("count"),
            )
            .filter(FrameData.stroke_type.isnot(None))
            .group_by(FrameData.racket_hit_player_id, FrameData.stroke_type)
            .all()
        )

        # Initialize stats for both players
        player_1_data = {"forehand": 0, "backhand": 0}
        player_2_data = {"forehand": 0, "backhand": 0}

        # Populate from query results
        for player_id, stroke_type, count in results:
            # Apply filter if specified
            if filters.player_id is not None and player_id != filters.player_id:
                continue

            if player_id == 1:
                player_1_data[stroke_type] = count
            elif player_id == 2:
                player_2_data[stroke_type] = count

        # Build distribution data using helper
        distribution_data = self._build_distribution_data(player_1_data, player_2_data)

        return StrokeDistributionResponse(
            video_id=video_id, filters=filters, data=distribution_data
        )

    def get_shot_type_distribution(
        self, video_id: str, filters: AnalyticsFilters
    ) -> ShotTypeDistributionResponse:
        """Get shot type distribution analytics."""
        logger.info(f"Computing shot type distribution for video {video_id}")

        query = self._get_base_query(video_id, filters)
        query = query.filter(
            FrameData.is_racket_hit == True, FrameData.shot_type.isnot(None)
        )

        # Don't apply player filter to query - always return both players

        results = (
            query.with_entities(
                FrameData.racket_hit_player_id,
                FrameData.shot_type,
                func.count(FrameData.id).label("count"),
            )
            .group_by(FrameData.racket_hit_player_id, FrameData.shot_type)
            .all()
        )

        player_1_data: Dict[str, int] = {}
        player_2_data: Dict[str, int] = {}
        all_shot_types = set()

        for player_id, shot_type, count in results:
            # Apply filter if specified
            if filters.player_id is not None and player_id != filters.player_id:
                continue

            all_shot_types.add(shot_type)
            if player_id == 1:
                player_1_data[shot_type] = count
            elif player_id == 2:
                player_2_data[shot_type] = count

        # Build distribution data using helper
        distribution_data = self._build_distribution_data(player_1_data, player_2_data)

        return ShotTypeDistributionResponse(
            video_id=video_id,
            filters=filters,
            data=distribution_data,
            all_shot_types=sorted(list(all_shot_types)),
        )

    def get_ball_speed_analytics(
        self, video_id: str, filters: AnalyticsFilters
    ) -> BallSpeedResponse:
        """Get ball speed aggregate statistics (no time series)."""
        logger.info(f"Computing ball speed analytics for video {video_id}")

        # Base query using precomputed ball_speed field
        query = self._get_base_query(video_id, filters).filter(
            FrameData.is_racket_hit == True,
            FrameData.ball_speed.isnot(None),
        )

        # Don't apply player filter - always return both players

        # SQL aggregation for player statistics (including std_dev)
        # Note: SQLite doesn't have built-in stddev, but we can use variance functions
        from sqlalchemy.sql import expression

        player_stats = (
            query.with_entities(
                FrameData.racket_hit_player_id,
                func.avg(FrameData.ball_speed).label("mean"),
                func.min(FrameData.ball_speed).label("min"),
                func.max(FrameData.ball_speed).label("max"),
                # For std_dev, we'll compute it in Python from variance
                func.count(FrameData.id).label("count"),
            )
            .group_by(FrameData.racket_hit_player_id)
            .all()
        )

        # Extract statistics and compute std_dev separately
        player_1_stats_tuple = None
        player_2_stats_tuple = None

        for row in player_stats:
            player_id = row.racket_hit_player_id

            # Get all speeds for this player to compute std_dev (vectorized)
            speeds = np.array(
                [
                    s[0]
                    for s in query.filter(FrameData.racket_hit_player_id == player_id)
                    .with_entities(FrameData.ball_speed)
                    .all()
                ]
            )

            # Compute std_dev using NumPy (vectorized)
            if len(speeds) > 1:
                std_dev = float(np.std(speeds))
            else:
                std_dev = 0.0

            stats_tuple = (
                player_id,
                row.mean,
                row.min,
                row.max,
                std_dev,
                row.count,
            )

            # Apply filter if specified
            if filters.player_id is not None and player_id != filters.player_id:
                continue

            if player_id == 1:
                player_1_stats_tuple = stats_tuple
            elif player_id == 2:
                player_2_stats_tuple = stats_tuple

        # Build aggregate data using helper
        aggregate_data = self._build_aggregate_data(
            player_1_stats_tuple, player_2_stats_tuple
        )

        return BallSpeedResponse(
            video_id=video_id, filters=filters, data=aggregate_data
        )

    def get_rhythm_disruption(
        self, video_id: str, filters: AnalyticsFilters
    ) -> RhythmDisruptionResponse:
        """Calculate rhythm disruption metrics using precomputed ball speeds and SQL."""
        logger.info(f"Computing rhythm disruption for video {video_id}")

        query = self._get_base_query(video_id, filters)

        # Fetch all racket hits in a single query (don't apply player filter to query)
        hit_query = query.filter(
            FrameData.is_racket_hit == True,
            FrameData.rally_id.isnot(None),
            FrameData.ball_speed.isnot(None),
        )

        all_hits = (
            hit_query.with_entities(
                FrameData.rally_id,
                FrameData.racket_hit_player_id,
                FrameData.ball_speed,
                FrameData.wall_hit_height,
            )
            .order_by(FrameData.rally_id, FrameData.frame_number)
            .all()
        )

        # Group hits by rally and player
        from collections import defaultdict

        rally_player_data = defaultdict(lambda: {"speeds": [], "heights": []})

        for hit in all_hits:
            # Apply filter if specified
            if (
                filters.player_id is not None
                and hit.racket_hit_player_id != filters.player_id
            ):
                continue

            key = (hit.rally_id, hit.racket_hit_player_id)
            rally_player_data[key]["speeds"].append(hit.ball_speed)
            if hit.wall_hit_height is not None:
                rally_player_data[key]["heights"].append(hit.wall_hit_height)

        # Calculate variance and CV for each rally-player combination
        rally_items = []
        player_1_cvs_speed = []
        player_2_cvs_speed = []
        player_1_cvs_height = []
        player_2_cvs_height = []

        for (rally_id, player_id), data in rally_player_data.items():
            speeds = np.array(data["speeds"])
            heights = np.array(data["heights"]) if data["heights"] else np.array([])

            # Calculate variance and CV for speeds (vectorized)
            speed_var = None
            speed_cv = None
            if len(speeds) >= 2:
                speed_mean = speeds.mean()
                speed_var = float(speeds.var())
                speed_cv = float(speeds.std() / speed_mean) if speed_mean > 0 else None
                if speed_cv is not None:
                    (
                        player_1_cvs_speed if player_id == 1 else player_2_cvs_speed
                    ).append(speed_cv)

            # Calculate variance and CV for heights (vectorized)
            height_var = None
            height_cv = None
            if len(heights) >= 2:
                height_mean = heights.mean()
                height_var = float(heights.var())
                height_cv = (
                    float(heights.std() / height_mean) if height_mean > 0 else None
                )
                if height_cv is not None:
                    (
                        player_1_cvs_height if player_id == 1 else player_2_cvs_height
                    ).append(height_cv)

            rally_items.append(
                RhythmRallyItem(
                    rally_id=rally_id,
                    player_id=player_id,
                    ball_speed_variance=speed_var,
                    ball_speed_cv=speed_cv,
                    wall_hit_height_variance=height_var,
                    wall_hit_height_cv=height_cv,
                    shot_count=len(speeds),
                )
            )

        # Build player summaries (always return both)
        player_1_summary = {
            "avg_ball_speed_cv": (
                sum(player_1_cvs_speed) / len(player_1_cvs_speed)
                if player_1_cvs_speed
                else None
            ),
            "avg_height_cv": (
                sum(player_1_cvs_height) / len(player_1_cvs_height)
                if player_1_cvs_height
                else None
            ),
        }

        player_2_summary = {
            "avg_ball_speed_cv": (
                sum(player_2_cvs_speed) / len(player_2_cvs_speed)
                if player_2_cvs_speed
                else None
            ),
            "avg_height_cv": (
                sum(player_2_cvs_height) / len(player_2_cvs_height)
                if player_2_cvs_height
                else None
            ),
        }

        rhythm_data = RhythmDisruptionData(
            items=rally_items,
            player_1_summary=player_1_summary,
            player_2_summary=player_2_summary,
        )

        return RhythmDisruptionResponse(
            video_id=video_id, filters=filters, data=rhythm_data
        )

    def get_player_position_heatmap(
        self, video_id: str, player_id: int, filters: AnalyticsFilters
    ) -> PlayerPositionHeatmapResponse:
        """Get player position heatmap for visualization using Spatial pattern."""
        logger.info(f"Computing position heatmap for player {player_id}")

        query = self._get_base_query(video_id, filters)
        query = query.filter(FrameData.is_rally_frame == True)

        # Dynamically select the player columns
        if player_id == 1:
            x_col = FrameData.player_1_x_meter
            y_col = FrameData.player_1_y_meter
        else:
            x_col = FrameData.player_2_x_meter
            y_col = FrameData.player_2_y_meter

        # Fetch only the needed columns with SQL
        results = (
            query.filter(x_col.isnot(None), y_col.isnot(None))
            .with_entities(
                x_col.label("x"),
                y_col.label("y"),
            )
            .all()
        )

        # Extract position points for heatmap computation
        points = [(row.x, row.y) for row in results]

        # Court boundaries for squash court
        court_bounds = {"x_min": 0, "x_max": 6.4, "y_min": 0, "y_max": 9.75}

        # Compute heatmap grid using helper method
        heatmap_grid = self._compute_heatmap_grid(points, court_bounds)

        # Build spatial data
        spatial_data = SpatialData(player_id=player_id, heatmap_grid=heatmap_grid)

        return PlayerPositionHeatmapResponse(
            video_id=video_id, filters=filters, data=spatial_data
        )

    def get_shot_placement_effectiveness(
        self, video_id: str, player_id: int, filters: AnalyticsFilters
    ) -> ShotPlacementResponse:
        """Analyze shot placement effectiveness using precomputed opponent distances and SQL."""
        logger.info(f"Computing shot placement effectiveness for player {player_id}")

        query = self._get_base_query(video_id, filters)

        # Determine which columns to select based on player_id
        if player_id == 1:
            player_x_col = FrameData.player_1_x_meter
            player_y_col = FrameData.player_1_y_meter
            opp_x_col = FrameData.player_2_x_meter
            opp_y_col = FrameData.player_2_y_meter
        else:
            player_x_col = FrameData.player_2_x_meter
            player_y_col = FrameData.player_2_y_meter
            opp_x_col = FrameData.player_1_x_meter
            opp_y_col = FrameData.player_1_y_meter

        # Fetch only needed columns with SQL
        player_hits = (
            query.filter(
                FrameData.is_racket_hit == True,
                FrameData.racket_hit_player_id == player_id,
            )
            .with_entities(
                FrameData.frame_number,
                FrameData.timestamp,
                player_x_col.label("player_x"),
                player_y_col.label("player_y"),
                opp_x_col.label("opp_x_before"),
                opp_y_col.label("opp_y_before"),
                FrameData.opponent_distance_moved,
                FrameData.next_opponent_x,
                FrameData.next_opponent_y,
            )
            .order_by(FrameData.frame_number)
            .all()
        )

        # Use SQL aggregation for average and max distance
        stats_result = (
            query.filter(
                FrameData.is_racket_hit == True,
                FrameData.racket_hit_player_id == player_id,
                FrameData.opponent_distance_moved.isnot(None),
            )
            .with_entities(
                func.avg(FrameData.opponent_distance_moved).label("avg_distance"),
                func.max(FrameData.opponent_distance_moved).label("max_distance"),
            )
            .first()
        )

        # Build placements list from SQL results
        shot_items = []
        for hit in player_hits:
            if None in [hit.opp_x_before, hit.opp_y_before, hit.player_x, hit.player_y]:
                continue

            shot_items.append(
                ShotPlacementItem(
                    frame_number=hit.frame_number,
                    timestamp=hit.timestamp,
                    player_id=player_id,
                    player_x=hit.player_x,
                    player_y=hit.player_y,
                    opponent_x_before=hit.opp_x_before,
                    opponent_y_before=hit.opp_y_before,
                    opponent_x_after=hit.next_opponent_x,
                    opponent_y_after=hit.next_opponent_y,
                    distance_moved=hit.opponent_distance_moved,
                )
            )

        # Build summary
        summary = {
            "avg_distance_moved": stats_result.avg_distance if stats_result else None,
            "max_distance_moved": stats_result.max_distance if stats_result else None,
        }

        # Build shot placement data
        shot_data = ShotPlacementData(
            player_id=player_id, items=shot_items, summary=summary
        )

        return ShotPlacementResponse(video_id=video_id, filters=filters, data=shot_data)

    def get_court_quadrant_distribution(
        self, video_id: str, filters: AnalyticsFilters
    ) -> CourtQuadrantResponse:
        """Calculate time spent in each court quadrant using Distribution pattern."""
        logger.info(f"Computing court quadrant distribution for video {video_id}")

        # Standard squash court quadrant boundaries
        X_CUT = 3.2  # meters (half court width)
        Y_CUT = 5.44  # meters (roughly half court length)

        query = self._get_base_query(video_id, filters)
        query = query.filter(FrameData.is_rally_frame == True)

        # SQL aggregation for Player 1 quadrants (always query both players)
        p1_results = (
            query.filter(
                FrameData.player_1_x_meter.isnot(None),
                FrameData.player_1_y_meter.isnot(None),
            )
            .with_entities(
                case(
                    (
                        FrameData.player_1_y_meter < Y_CUT,
                        case(
                            (FrameData.player_1_x_meter < X_CUT, "Front-Left"),
                            else_="Front-Right",
                        ),
                    ),
                    else_=case(
                        (FrameData.player_1_x_meter < X_CUT, "Back-Left"),
                        else_="Back-Right",
                    ),
                ).label("quadrant"),
                func.count().label("count"),
            )
            .group_by("quadrant")
            .all()
        )

        # SQL aggregation for Player 2 quadrants
        p2_results = (
            query.filter(
                FrameData.player_2_x_meter.isnot(None),
                FrameData.player_2_y_meter.isnot(None),
            )
            .with_entities(
                case(
                    (
                        FrameData.player_2_y_meter < Y_CUT,
                        case(
                            (FrameData.player_2_x_meter < X_CUT, "Front-Left"),
                            else_="Front-Right",
                        ),
                    ),
                    else_=case(
                        (FrameData.player_2_x_meter < X_CUT, "Back-Left"),
                        else_="Back-Right",
                    ),
                ).label("quadrant"),
                func.count().label("count"),
            )
            .group_by("quadrant")
            .all()
        )

        # Convert SQL results to dictionaries
        player_1_quadrants = {row.quadrant: row.count for row in p1_results}
        player_2_quadrants = {row.quadrant: row.count for row in p2_results}

        # Ensure all quadrants are present in results
        all_quadrants = ["Front-Left", "Front-Right", "Back-Left", "Back-Right"]
        for quadrant in all_quadrants:
            player_1_quadrants.setdefault(quadrant, 0)
            player_2_quadrants.setdefault(quadrant, 0)

        # Apply player filter if specified (zero out non-filtered player)
        if filters.player_id == 1:
            player_2_quadrants = {q: 0 for q in all_quadrants}
        elif filters.player_id == 2:
            player_1_quadrants = {q: 0 for q in all_quadrants}

        # Build distribution data using helper
        distribution_data = self._build_distribution_data(
            player_1_quadrants, player_2_quadrants
        )

        return CourtQuadrantResponse(
            video_id=video_id,
            filters=filters,
            data=distribution_data,
            quadrant_boundaries={"x_cut": X_CUT, "y_cut": Y_CUT},
        )

    def get_rally_stats(
        self, video_id: str, filters: AnalyticsFilters
    ) -> RallyStatsResponse:
        """Get rally duration and stroke count statistics using Comparative pattern."""
        logger.info(f"Computing rally stats for video {video_id}")

        query = self._get_base_query(video_id, filters)

        # Single SQL query with GROUP BY instead of N queries
        results = (
            query.filter(FrameData.rally_id.isnot(None))
            .with_entities(
                FrameData.rally_id,
                func.min(FrameData.timestamp).label("start_time"),
                func.max(FrameData.timestamp).label("end_time"),
                func.sum(case((FrameData.is_racket_hit == True, 1), else_=0)).label(
                    "total_shots"
                ),
                func.sum(
                    case(
                        (
                            (FrameData.is_racket_hit == True)
                            & (FrameData.racket_hit_player_id == 1),
                            1,
                        ),
                        else_=0,
                    )
                ).label("p1_shots"),
                func.sum(
                    case(
                        (
                            (FrameData.is_racket_hit == True)
                            & (FrameData.racket_hit_player_id == 2),
                            1,
                        ),
                        else_=0,
                    )
                ).label("p2_shots"),
            )
            .group_by(FrameData.rally_id)
            .all()
        )

        rally_items = []
        total_duration = 0
        total_strokes = 0
        player_1_total_shots = 0
        player_2_total_shots = 0

        for rally_id, start_time, end_time, stroke_count, p1_shots, p2_shots in results:
            duration = end_time - start_time

            rally_items.append(
                RallyItem(
                    rally_id=rally_id,
                    duration_seconds=duration,
                    stroke_count=stroke_count,
                    player_1_shots=p1_shots,
                    player_2_shots=p2_shots,
                    start_timestamp=start_time,
                    end_timestamp=end_time,
                )
            )

            total_duration += duration
            total_strokes += stroke_count
            player_1_total_shots += p1_shots
            player_2_total_shots += p2_shots

        # Build summaries (always return both players)
        total_rallies = len(rally_items)
        summary = {
            "total_rallies": total_rallies,
            "avg_duration": total_duration / total_rallies if total_rallies > 0 else 0,
            "avg_stroke_count": (
                total_strokes / total_rallies if total_rallies > 0 else 0
            ),
        }

        player_1_summary = {
            "total_shots": player_1_total_shots,
            "avg_shots_per_rally": (
                player_1_total_shots / total_rallies if total_rallies > 0 else 0
            ),
        }

        player_2_summary = {
            "total_shots": player_2_total_shots,
            "avg_shots_per_rally": (
                player_2_total_shots / total_rallies if total_rallies > 0 else 0
            ),
        }

        rally_data = RallyStatsData(
            items=rally_items,
            summary=summary,
            player_1_summary=player_1_summary,
            player_2_summary=player_2_summary,
        )

        return RallyStatsResponse(video_id=video_id, filters=filters, data=rally_data)

    def get_wall_hit_heatmap(
        self, video_id: str, filters: AnalyticsFilters
    ) -> WallHitHeatmapResponse:
        """Get wall hit distribution heatmap using Spatial pattern."""
        logger.info(f"Computing wall hit distribution for video {video_id}")

        query = self._get_base_query(video_id, filters)
        query = query.filter(
            FrameData.is_wall_hit == True,
            FrameData.wall_hit_x_meter.isnot(None),
            FrameData.wall_hit_y_meter.isnot(None),
        )

        # Apply player filter using wall_hit_player_id
        if filters.player_id is not None:
            query = query.filter(FrameData.wall_hit_player_id == filters.player_id)

        # Fetch only needed columns with SQL
        wall_hits_frames = query.with_entities(
            FrameData.wall_hit_x_meter,
            FrameData.wall_hit_y_meter,
        ).all()

        # Extract points for heatmap computation
        points = [(f.wall_hit_x_meter, f.wall_hit_y_meter) for f in wall_hits_frames]

        # Wall boundaries for squash court
        wall_bounds = {"x_min": 0, "x_max": 6.4, "y_min": 0, "y_max": 4.57}

        # Compute heatmap grid using helper method
        heatmap_grid = self._compute_heatmap_grid(points, wall_bounds)

        # Build spatial data (use player_id from filter, or 0 if not specified)
        player_id = filters.player_id if filters.player_id is not None else 0
        spatial_data = SpatialData(player_id=player_id, heatmap_grid=heatmap_grid)

        return WallHitHeatmapResponse(
            video_id=video_id, filters=filters, data=spatial_data
        )

    def get_wall_quadrant_distribution(
        self, video_id: str, filters: AnalyticsFilters
    ) -> WallQuadrantResponse:
        """Calculate wall hit distribution across quadrants using Distribution pattern."""
        logger.info(f"Computing wall quadrant distribution for video {video_id}")

        # Front wall quadrant boundaries
        X_CUT = 3.2  # meters (half wall width, same as court width)
        Y_CUT = 2.285  # meters (half wall height of 4.57m)

        query = self._get_base_query(video_id, filters)

        # SQL aggregation for Player 1 wall hits (always query both players)
        p1_results = (
            query.filter(
                FrameData.is_wall_hit == True,
                FrameData.wall_hit_player_id == 1,
                FrameData.wall_hit_x_meter.isnot(None),
                FrameData.wall_hit_y_meter.isnot(None),
            )
            .with_entities(
                case(
                    (
                        FrameData.wall_hit_y_meter < Y_CUT,
                        case(
                            (FrameData.wall_hit_x_meter < X_CUT, "Bottom-Left"),
                            else_="Bottom-Right",
                        ),
                    ),
                    else_=case(
                        (FrameData.wall_hit_x_meter < X_CUT, "Top-Left"),
                        else_="Top-Right",
                    ),
                ).label("quadrant"),
                func.count().label("count"),
            )
            .group_by("quadrant")
            .all()
        )

        # SQL aggregation for Player 2 wall hits
        p2_results = (
            query.filter(
                FrameData.is_wall_hit == True,
                FrameData.wall_hit_player_id == 2,
                FrameData.wall_hit_x_meter.isnot(None),
                FrameData.wall_hit_y_meter.isnot(None),
            )
            .with_entities(
                case(
                    (
                        FrameData.wall_hit_y_meter < Y_CUT,
                        case(
                            (FrameData.wall_hit_x_meter < X_CUT, "Bottom-Left"),
                            else_="Bottom-Right",
                        ),
                    ),
                    else_=case(
                        (FrameData.wall_hit_x_meter < X_CUT, "Top-Left"),
                        else_="Top-Right",
                    ),
                ).label("quadrant"),
                func.count().label("count"),
            )
            .group_by("quadrant")
            .all()
        )

        # Convert SQL results to dictionaries
        player_1_quadrants = {row.quadrant: row.count for row in p1_results}
        player_2_quadrants = {row.quadrant: row.count for row in p2_results}

        # Ensure all quadrants are present in results
        all_quadrants = ["Bottom-Left", "Bottom-Right", "Top-Left", "Top-Right"]
        for quadrant in all_quadrants:
            player_1_quadrants.setdefault(quadrant, 0)
            player_2_quadrants.setdefault(quadrant, 0)

        # Apply player filter if specified (zero out non-filtered player)
        if filters.player_id == 1:
            player_2_quadrants = {q: 0 for q in all_quadrants}
        elif filters.player_id == 2:
            player_1_quadrants = {q: 0 for q in all_quadrants}

        # Build distribution data using helper
        distribution_data = self._build_distribution_data(
            player_1_quadrants, player_2_quadrants
        )

        return WallQuadrantResponse(
            video_id=video_id,
            filters=filters,
            data=distribution_data,
            quadrant_boundaries={"x_cut": X_CUT, "y_cut": Y_CUT},
        )

    def get_winning_stats(
        self, video_id: str, filters: AnalyticsFilters
    ) -> WinningStatsResponse:
        """Calculate winning statistics and points per shot ratios using Comparative pattern."""
        logger.info(f"Computing winning stats for video {video_id}")

        query = self._get_base_query(video_id, filters)

        # Single SQL query with GROUP BY rally_id, racket_hit_player_id
        # Use UNION to get stats for both players per rally in one query
        from sqlalchemy import union_all

        # Always query both players (don't apply player filter to subqueries)
        subqueries = []

        # Subquery for player 1
        player1_stats = (
            query.filter(FrameData.rally_id.isnot(None))
            .with_entities(
                FrameData.rally_id,
                func.sum(
                    case(
                        (
                            (FrameData.is_racket_hit == True)
                            & (FrameData.racket_hit_player_id == 1),
                            1,
                        ),
                        else_=0,
                    )
                ).label("total_shots"),
                func.sum(case((FrameData.point_winner == 1, 1), else_=0)).label(
                    "points_won"
                ),
                literal(1).label("player_id"),
            )
            .group_by(FrameData.rally_id)
        )
        subqueries.append(player1_stats)

        # Subquery for player 2
        player2_stats = (
            query.filter(FrameData.rally_id.isnot(None))
            .with_entities(
                FrameData.rally_id,
                func.sum(
                    case(
                        (
                            (FrameData.is_racket_hit == True)
                            & (FrameData.racket_hit_player_id == 2),
                            1,
                        ),
                        else_=0,
                    )
                ).label("total_shots"),
                func.sum(case((FrameData.point_winner == 2, 1), else_=0)).label(
                    "points_won"
                ),
                literal(2).label("player_id"),
            )
            .group_by(FrameData.rally_id)
        )
        subqueries.append(player2_stats)

        # Combine player stats
        combined_query = union_all(subqueries[0], subqueries[1])
        results = self.db.execute(combined_query).all()

        # Build rally stats and calculate totals
        winning_items = []
        player_1_total_points = 0
        player_1_total_shots = 0
        player_2_total_points = 0
        player_2_total_shots = 0

        for row in results:
            rally_id, total_shots, points_won, player_id = row

            # Apply filter if specified
            if filters.player_id is not None and player_id != filters.player_id:
                continue

            # Only include if player has shots in this rally
            if total_shots > 0:
                winning_items.append(
                    WinningRallyItem(
                        rally_id=rally_id,
                        player_id=player_id,
                        total_shots=total_shots,
                        points_won=points_won,
                        points_per_shot=points_won / total_shots,
                    )
                )

                if player_id == 1:
                    player_1_total_points += points_won
                    player_1_total_shots += total_shots
                else:
                    player_2_total_points += points_won
                    player_2_total_shots += total_shots

        # Build player summaries (always return both)
        player_1_summary = {
            "total_points": player_1_total_points,
            "total_shots": player_1_total_shots,
            "efficiency": (
                player_1_total_points / player_1_total_shots
                if player_1_total_shots > 0
                else 0
            ),
        }

        player_2_summary = {
            "total_points": player_2_total_points,
            "total_shots": player_2_total_shots,
            "efficiency": (
                player_2_total_points / player_2_total_shots
                if player_2_total_shots > 0
                else 0
            ),
        }

        winning_data = WinningStatsData(
            items=winning_items,
            player_1_summary=player_1_summary,
            player_2_summary=player_2_summary,
        )

        return WinningStatsResponse(
            video_id=video_id, filters=filters, data=winning_data
        )
