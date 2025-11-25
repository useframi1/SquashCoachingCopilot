"""Analysis service for querying processed video data."""

import logging
import math
from typing import Optional

from sqlalchemy import case, func, literal
from sqlalchemy.orm import Session, Query

from backend.models.frame_data import FrameData
from backend.models.video import Video
from backend.schemas.analysis import (
    # Analytics schemas
    AnalyticsFilters,
    StrokeDistributionResponse,
    PlayerStrokeStats,
    ShotTypeDistributionResponse,
    ShotTypeStats,
    BallSpeedAnalyticsResponse,
    BallSpeedDataPoint,
    RhythmDisruptionResponse,
    RallyRhythmStats,
    PlayerPositionHeatmapResponse,
    PositionPoint,
    ShotPlacementResponse,
    ShotPlacementDetail,
    CourtQuadrantResponse,
    QuadrantStats,
    RallyStatsResponse,
    RallyStatsDetail,
    WallHitDistributionResponse,
    WallHitPoint,
    WinningStatsResponse,
    WinningStatsDetail,
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

    def get_stroke_distribution(
        self, video_id: str, filters: AnalyticsFilters
    ) -> StrokeDistributionResponse:
        """Get stroke distribution analytics (forehand vs backhand)."""
        logger.info(f"Computing stroke distribution for video {video_id}")

        query = self._get_base_query(video_id, filters)
        query = query.filter(FrameData.is_racket_hit == True)

        # Apply player filter if specified
        if filters.player_id is not None:
            query = query.filter(FrameData.racket_hit_player_id == filters.player_id)

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

        # Initialize stats
        player_1_stats = {"forehand": 0, "backhand": 0}
        player_2_stats = {"forehand": 0, "backhand": 0}

        for player_id, stroke_type, count in results:
            if player_id == 1:
                if stroke_type == "forehand":
                    player_1_stats["forehand"] = count
                elif stroke_type == "backhand":
                    player_1_stats["backhand"] = count
            elif player_id == 2:
                if stroke_type == "forehand":
                    player_2_stats["forehand"] = count
                elif stroke_type == "backhand":
                    player_2_stats["backhand"] = count

        total_data_points = sum(player_1_stats.values()) + sum(player_2_stats.values())

        # Build response based on player filter
        player_1_obj = None
        player_2_obj = None

        if filters.player_id is None or filters.player_id == 1:
            player_1_obj = PlayerStrokeStats(
                player_id=1,
                forehand_count=player_1_stats["forehand"],
                backhand_count=player_1_stats["backhand"],
                total_shots=player_1_stats["forehand"] + player_1_stats["backhand"],
            )

        if filters.player_id is None or filters.player_id == 2:
            player_2_obj = PlayerStrokeStats(
                player_id=2,
                forehand_count=player_2_stats["forehand"],
                backhand_count=player_2_stats["backhand"],
                total_shots=player_2_stats["forehand"] + player_2_stats["backhand"],
            )

        return StrokeDistributionResponse(
            video_id=video_id,
            filters_applied=filters,
            data_points=total_data_points,
            player_1=player_1_obj,
            player_2=player_2_obj,
        )

    def get_shot_type_distribution(
        self, video_id: str, filters: AnalyticsFilters
    ) -> ShotTypeDistributionResponse:
        """Get shot type distribution analytics."""
        logger.info(f"Computing shot type distribution for video {video_id}")

        query = self._get_base_query(video_id, filters)
        query = query.filter(FrameData.is_racket_hit == True, FrameData.shot_type.isnot(None))

        if filters.player_id is not None:
            query = query.filter(FrameData.racket_hit_player_id == filters.player_id)

        results = (
            query.with_entities(
                FrameData.racket_hit_player_id,
                FrameData.shot_type,
                func.count(FrameData.id).label("count"),
            )
            .group_by(FrameData.racket_hit_player_id, FrameData.shot_type)
            .all()
        )

        player_1_shots = {}
        player_2_shots = {}
        all_shot_types = set()

        for player_id, shot_type, count in results:
            all_shot_types.add(shot_type)
            if player_id == 1:
                player_1_shots[shot_type] = count
            elif player_id == 2:
                player_2_shots[shot_type] = count

        # Build response based on player filter
        player_1_obj = None
        player_2_obj = None

        if filters.player_id is None or filters.player_id == 1:
            player_1_obj = ShotTypeStats(
                player_id=1,
                shot_counts=player_1_shots,
                total_shots=sum(player_1_shots.values()),
            )

        if filters.player_id is None or filters.player_id == 2:
            player_2_obj = ShotTypeStats(
                player_id=2,
                shot_counts=player_2_shots,
                total_shots=sum(player_2_shots.values()),
            )

        return ShotTypeDistributionResponse(
            video_id=video_id,
            filters_applied=filters,
            data_points=sum(player_1_shots.values()) + sum(player_2_shots.values()),
            player_1=player_1_obj,
            player_2=player_2_obj,
            all_shot_types=sorted(list(all_shot_types)),
        )

    def get_ball_speed_analytics(
        self, video_id: str, filters: AnalyticsFilters
    ) -> BallSpeedAnalyticsResponse:
        """Get ball speed analytics using precomputed values and SQL aggregation."""
        logger.info(f"Computing ball speed analytics for video {video_id}")

        # Base query using precomputed ball_speed field
        query = self._get_base_query(video_id, filters).filter(
            FrameData.is_racket_hit == True,
            FrameData.ball_speed.isnot(None),
        )

        if filters.player_id is not None:
            query = query.filter(FrameData.racket_hit_player_id == filters.player_id)

        # Fetch time series data with only needed columns
        results = (
            query.with_entities(
                FrameData.timestamp,
                FrameData.frame_number,
                FrameData.racket_hit_player_id,
                FrameData.ball_speed,
            )
            .order_by(FrameData.frame_number)
            .all()
        )

        # Use SQL aggregation for player statistics
        player_stats = (
            query.with_entities(
                FrameData.racket_hit_player_id,
                func.avg(FrameData.ball_speed).label('avg_speed'),
                func.max(FrameData.ball_speed).label('max_speed'),
                func.min(FrameData.ball_speed).label('min_speed'),
            )
            .group_by(FrameData.racket_hit_player_id)
            .all()
        )

        # Build time series from SQL results
        time_series = [
            BallSpeedDataPoint(
                timestamp=row.timestamp,
                player_id=row.racket_hit_player_id,
                speed=row.ball_speed,
                frame_number=row.frame_number,
            )
            for row in results
        ]

        # Extract player statistics from SQL aggregation
        player_1_avg = player_1_max = player_1_min = None
        player_2_avg = player_2_max = player_2_min = None

        for row in player_stats:
            if row.racket_hit_player_id == 1:
                player_1_avg = row.avg_speed
                player_1_max = row.max_speed
                player_1_min = row.min_speed
            elif row.racket_hit_player_id == 2:
                player_2_avg = row.avg_speed
                player_2_max = row.max_speed
                player_2_min = row.min_speed

        return BallSpeedAnalyticsResponse(
            video_id=video_id,
            filters_applied=filters,
            data_points=len(time_series),
            time_series=time_series,
            player_1_avg_speed=player_1_avg,
            player_1_max_speed=player_1_max,
            player_1_min_speed=player_1_min,
            player_2_avg_speed=player_2_avg,
            player_2_max_speed=player_2_max,
            player_2_min_speed=player_2_min,
        )

    def get_rhythm_disruption(
        self, video_id: str, filters: AnalyticsFilters
    ) -> RhythmDisruptionResponse:
        """Calculate rhythm disruption metrics using precomputed ball speeds and SQL."""
        logger.info(f"Computing rhythm disruption for video {video_id}")

        query = self._get_base_query(video_id, filters)

        # Fetch all racket hits in a single query
        hit_query = query.filter(
            FrameData.is_racket_hit == True,
            FrameData.rally_id.isnot(None),
            FrameData.ball_speed.isnot(None),
        )

        # Apply player filter if specified
        if filters.player_id is not None:
            hit_query = hit_query.filter(FrameData.racket_hit_player_id == filters.player_id)

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
            key = (hit.rally_id, hit.racket_hit_player_id)
            rally_player_data[key]["speeds"].append(hit.ball_speed)
            if hit.wall_hit_height is not None:
                rally_player_data[key]["heights"].append(hit.wall_hit_height)

        # Calculate variance and CV for each rally-player combination
        rally_stats = []
        player_1_cvs_speed = []
        player_2_cvs_speed = []
        player_1_cvs_height = []
        player_2_cvs_height = []

        for (rally_id, player_id), data in rally_player_data.items():
            speeds = data["speeds"]
            heights = data["heights"]

            # Calculate variance and CV for speeds
            speed_var = None
            speed_cv = None
            if len(speeds) >= 2:
                speed_mean = sum(speeds) / len(speeds)
                speed_var = sum((s - speed_mean) ** 2 for s in speeds) / len(speeds)
                speed_cv = math.sqrt(speed_var) / speed_mean if speed_mean > 0 else None
                if speed_cv is not None:
                    (player_1_cvs_speed if player_id == 1 else player_2_cvs_speed).append(speed_cv)

            # Calculate variance and CV for heights
            height_var = None
            height_cv = None
            if len(heights) >= 2:
                height_mean = sum(heights) / len(heights)
                height_var = sum((h - height_mean) ** 2 for h in heights) / len(heights)
                height_cv = math.sqrt(height_var) / height_mean if height_mean > 0 else None
                if height_cv is not None:
                    (player_1_cvs_height if player_id == 1 else player_2_cvs_height).append(height_cv)

            rally_stats.append(
                RallyRhythmStats(
                    rally_id=rally_id,
                    player_id=player_id,
                    ball_speed_variance=speed_var,
                    ball_speed_cv=speed_cv,
                    wall_hit_height_variance=height_var,
                    wall_hit_height_cv=height_cv,
                    shot_count=len(speeds),
                )
            )

        return RhythmDisruptionResponse(
            video_id=video_id,
            filters_applied=filters,
            data_points=len(rally_stats),
            rallies=rally_stats,
            player_1_avg_ball_speed_cv=sum(player_1_cvs_speed) / len(player_1_cvs_speed) if player_1_cvs_speed else None,
            player_2_avg_ball_speed_cv=sum(player_2_cvs_speed) / len(player_2_cvs_speed) if player_2_cvs_speed else None,
            player_1_avg_height_cv=sum(player_1_cvs_height) / len(player_1_cvs_height) if player_1_cvs_height else None,
            player_2_avg_height_cv=sum(player_2_cvs_height) / len(player_2_cvs_height) if player_2_cvs_height else None,
        )

    def get_player_position_heatmap(
        self, video_id: str, player_id: int, filters: AnalyticsFilters
    ) -> PlayerPositionHeatmapResponse:
        """Get player position data for heatmap visualization using optimized SQL."""
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
                x_col.label('x'),
                y_col.label('y'),
                FrameData.timestamp,
                FrameData.frame_number
            )
            .all()
        )

        # Use SQL aggregation for bounds
        bounds_result = (
            query.filter(x_col.isnot(None), y_col.isnot(None))
            .with_entities(
                func.min(x_col).label('x_min'),
                func.max(x_col).label('x_max'),
                func.min(y_col).label('y_min'),
                func.max(y_col).label('y_max')
            )
            .first()
        )

        # Build position list from SQL results
        positions = [
            PositionPoint(
                x=row.x,
                y=row.y,
                timestamp=row.timestamp,
                frame_number=row.frame_number,
            )
            for row in results
        ]

        court_bounds = {
            "x_min": bounds_result.x_min if bounds_result.x_min is not None else 0,
            "x_max": bounds_result.x_max if bounds_result.x_max is not None else 6.4,
            "y_min": bounds_result.y_min if bounds_result.y_min is not None else 0,
            "y_max": bounds_result.y_max if bounds_result.y_max is not None else 9.75,
        }

        return PlayerPositionHeatmapResponse(
            video_id=video_id,
            filters_applied=filters,
            data_points=len(positions),
            player_id=player_id,
            positions=positions,
            court_bounds=court_bounds,
        )

    def get_shot_placement_effectiveness(
        self, video_id: str, player_id: int, filters: AnalyticsFilters
    ) -> ShotPlacementResponse:
        """Analyze shot placement effectiveness using precomputed opponent distances and SQL."""
        logger.info(f"Computing shot placement effectiveness for player {player_id}")

        opponent_id = 2 if player_id == 1 else 1
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
                player_x_col.label('player_x'),
                player_y_col.label('player_y'),
                opp_x_col.label('opp_x_before'),
                opp_y_col.label('opp_y_before'),
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
                FrameData.opponent_distance_moved.isnot(None)
            )
            .with_entities(
                func.avg(FrameData.opponent_distance_moved).label('avg_distance'),
                func.max(FrameData.opponent_distance_moved).label('max_distance')
            )
            .first()
        )

        # Build placements list from SQL results
        placements = []
        for hit in player_hits:
            if None in [hit.opp_x_before, hit.opp_y_before, hit.player_x, hit.player_y]:
                continue

            placements.append(
                ShotPlacementDetail(
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

        return ShotPlacementResponse(
            video_id=video_id,
            filters_applied=filters,
            data_points=len(placements),
            player_id=player_id,
            placements=placements,
            avg_distance_moved=stats_result.avg_distance if stats_result else None,
            max_distance_moved=stats_result.max_distance if stats_result else None,
        )

    def get_court_quadrant_distribution(
        self, video_id: str, filters: AnalyticsFilters
    ) -> CourtQuadrantResponse:
        """Calculate time spent in each court quadrant using SQL aggregation."""
        logger.info(f"Computing court quadrant distribution for video {video_id}")

        # Standard squash court quadrant boundaries
        X_CUT = 3.2  # meters (half court width)
        Y_CUT = 5.44  # meters (roughly half court length)

        query = self._get_base_query(video_id, filters)
        query = query.filter(FrameData.is_rally_frame == True)

        # Get video FPS for time calculation
        video = self._get_video_or_404(video_id)
        fps = video.fps or 30

        # Determine which players to query based on filter
        query_player_1 = filters.player_id is None or filters.player_id == 1
        query_player_2 = filters.player_id is None or filters.player_id == 2

        # SQL aggregation for Player 1 quadrants
        p1_results = []
        if query_player_1:
            p1_results = (
                query.filter(
                    FrameData.player_1_x_meter.isnot(None),
                    FrameData.player_1_y_meter.isnot(None)
                )
                .with_entities(
                    case(
                        (FrameData.player_1_y_meter < Y_CUT,
                         case((FrameData.player_1_x_meter < X_CUT, "Front-Left"), else_="Front-Right")),
                        else_=case((FrameData.player_1_x_meter < X_CUT, "Back-Left"), else_="Back-Right")
                    ).label('quadrant'),
                    func.count().label('count')
                )
                .group_by('quadrant')
                .all()
            )

        # SQL aggregation for Player 2 quadrants
        p2_results = []
        if query_player_2:
            p2_results = (
                query.filter(
                    FrameData.player_2_x_meter.isnot(None),
                    FrameData.player_2_y_meter.isnot(None)
                )
                .with_entities(
                    case(
                        (FrameData.player_2_y_meter < Y_CUT,
                         case((FrameData.player_2_x_meter < X_CUT, "Front-Left"), else_="Front-Right")),
                        else_=case((FrameData.player_2_x_meter < X_CUT, "Back-Left"), else_="Back-Right")
                    ).label('quadrant'),
                    func.count().label('count')
                )
                .group_by('quadrant')
                .all()
            )

        # Convert SQL results to dictionaries
        player_1_quadrants = {row.quadrant: row.count for row in p1_results}
        player_2_quadrants = {row.quadrant: row.count for row in p2_results}

        player_1_total = sum(player_1_quadrants.values())
        player_2_total = sum(player_2_quadrants.values())

        # Ensure all quadrants are present in results
        all_quadrants = ["Front-Left", "Front-Right", "Back-Left", "Back-Right"]
        for quadrant in all_quadrants:
            player_1_quadrants.setdefault(quadrant, 0)
            player_2_quadrants.setdefault(quadrant, 0)

        # Convert to stats objects
        def make_stats(quadrants, total):
            return [
                QuadrantStats(
                    quadrant=q,
                    frame_count=quadrants[q],
                    percentage=(quadrants[q] / total * 100) if total > 0 else 0,
                    avg_time_seconds=(quadrants[q] / fps) if fps > 0 else 0,
                )
                for q in all_quadrants
            ]

        return CourtQuadrantResponse(
            video_id=video_id,
            filters_applied=filters,
            data_points=player_1_total + player_2_total,
            player_1_quadrants=make_stats(player_1_quadrants, player_1_total),
            player_2_quadrants=make_stats(player_2_quadrants, player_2_total),
            quadrant_boundaries={"x_cut": X_CUT, "y_cut": Y_CUT},
        )

    def get_rally_stats(
        self, video_id: str, filters: AnalyticsFilters
    ) -> RallyStatsResponse:
        """Get rally duration and stroke count statistics using SQL aggregation."""
        logger.info(f"Computing rally stats for video {video_id}")

        query = self._get_base_query(video_id, filters)

        # Single SQL query with GROUP BY instead of N queries
        results = (
            query.filter(FrameData.rally_id.isnot(None))
            .with_entities(
                FrameData.rally_id,
                func.min(FrameData.timestamp).label('start_time'),
                func.max(FrameData.timestamp).label('end_time'),
                func.sum(case((FrameData.is_racket_hit == True, 1), else_=0)).label('total_shots'),
                func.sum(case(((FrameData.is_racket_hit == True) & (FrameData.racket_hit_player_id == 1), 1), else_=0)).label('p1_shots'),
                func.sum(case(((FrameData.is_racket_hit == True) & (FrameData.racket_hit_player_id == 2), 1), else_=0)).label('p2_shots'),
            )
            .group_by(FrameData.rally_id)
            .all()
        )

        rallies = []
        total_duration = 0
        total_strokes = 0

        for rally_id, start_time, end_time, stroke_count, p1_shots, p2_shots in results:
            duration = end_time - start_time

            rallies.append(
                RallyStatsDetail(
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

        return RallyStatsResponse(
            video_id=video_id,
            filters_applied=filters,
            data_points=len(rallies),
            rallies=rallies,
            avg_rally_duration=total_duration / len(rallies) if rallies else 0,
            avg_stroke_count=total_strokes / len(rallies) if rallies else 0,
            total_rallies=len(rallies),
        )

    def get_wall_hit_distribution(
        self, video_id: str, filters: AnalyticsFilters, quadrant: Optional[str] = None
    ) -> WallHitDistributionResponse:
        """Get wall hit positions for shot placement heatmaps."""
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

        # Apply quadrant filter if specified
        if quadrant:
            X_CUT = 3.2
            Y_CUT = 5.44

            # Validate quadrant
            valid_quadrants = {"Front-Left", "Front-Right", "Back-Left", "Back-Right"}
            if quadrant not in valid_quadrants:
                raise ValueError(f"quadrant must be one of {valid_quadrants}")

            # Parse quadrant (e.g., "Front-Left")
            parts = quadrant.split("-")
            if len(parts) == 2:
                front_back, left_right = parts
                if front_back == "Front":
                    query = query.filter(FrameData.wall_hit_y_meter < Y_CUT)
                else:
                    query = query.filter(FrameData.wall_hit_y_meter >= Y_CUT)

                if left_right == "Left":
                    query = query.filter(FrameData.wall_hit_x_meter < X_CUT)
                else:
                    query = query.filter(FrameData.wall_hit_x_meter >= X_CUT)

        # Fetch only needed columns with SQL
        wall_hits_frames = (
            query.with_entities(
                FrameData.wall_hit_x_meter,
                FrameData.wall_hit_y_meter,
                FrameData.timestamp,
                FrameData.frame_number,
                FrameData.wall_hit_player_id,
            )
            .all()
        )

        wall_hits = [
            WallHitPoint(
                x=f.wall_hit_x_meter,
                y=f.wall_hit_y_meter,
                timestamp=f.timestamp,
                frame_number=f.frame_number,
                player_id=f.wall_hit_player_id or 0,
            )
            for f in wall_hits_frames
        ]

        return WallHitDistributionResponse(
            video_id=video_id,
            filters_applied=filters,
            data_points=len(wall_hits),
            wall_hits=wall_hits,
            player_id=filters.player_id,
            wall_bounds={"x_min": 0, "x_max": 6.4, "y_min": 0, "y_max": 9.75},
        )

    def get_winning_stats(
        self, video_id: str, filters: AnalyticsFilters
    ) -> WinningStatsResponse:
        """Calculate winning statistics and points per shot ratios using SQL aggregation."""
        logger.info(f"Computing winning stats for video {video_id}")

        query = self._get_base_query(video_id, filters)

        # Single SQL query with GROUP BY rally_id, racket_hit_player_id
        # Use UNION to get stats for both players per rally in one query
        from sqlalchemy import union_all

        # Determine which players to query based on filter
        query_player_1 = filters.player_id is None or filters.player_id == 1
        query_player_2 = filters.player_id is None or filters.player_id == 2

        subqueries = []

        # Subquery for player 1
        if query_player_1:
            player1_stats = (
                query.filter(FrameData.rally_id.isnot(None))
                .with_entities(
                    FrameData.rally_id,
                    func.sum(case(((FrameData.is_racket_hit == True) & (FrameData.racket_hit_player_id == 1), 1), else_=0)).label('total_shots'),
                    func.sum(case((FrameData.point_winner == 1, 1), else_=0)).label('points_won'),
                    literal(1).label('player_id'),
                )
                .group_by(FrameData.rally_id)
            )
            subqueries.append(player1_stats)

        # Subquery for player 2
        if query_player_2:
            player2_stats = (
                query.filter(FrameData.rally_id.isnot(None))
                .with_entities(
                    FrameData.rally_id,
                    func.sum(case(((FrameData.is_racket_hit == True) & (FrameData.racket_hit_player_id == 2), 1), else_=0)).label('total_shots'),
                    func.sum(case((FrameData.point_winner == 2, 1), else_=0)).label('points_won'),
                    literal(2).label('player_id'),
                )
                .group_by(FrameData.rally_id)
            )
            subqueries.append(player2_stats)

        # Combine player stats
        if len(subqueries) == 0:
            results = []
        elif len(subqueries) == 1:
            results = self.db.execute(subqueries[0]).all()
        else:
            combined_query = union_all(subqueries[0], subqueries[1])
            results = self.db.execute(combined_query).all()

        # Build rally stats and calculate totals
        rally_stats = []
        player_1_total_points = 0
        player_1_total_shots = 0
        player_2_total_points = 0
        player_2_total_shots = 0

        for row in results:
            rally_id, total_shots, points_won, player_id = row

            # Only include if player has shots in this rally
            if total_shots > 0:
                rally_stats.append(
                    WinningStatsDetail(
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

        return WinningStatsResponse(
            video_id=video_id,
            filters_applied=filters,
            data_points=len(rally_stats),
            rally_stats=rally_stats,
            player_1_total_points=player_1_total_points,
            player_1_total_shots=player_1_total_shots,
            player_1_efficiency=player_1_total_points / player_1_total_shots if player_1_total_shots > 0 else 0,
            player_2_total_points=player_2_total_points,
            player_2_total_shots=player_2_total_shots,
            player_2_efficiency=player_2_total_points / player_2_total_shots if player_2_total_shots > 0 else 0,
        )
