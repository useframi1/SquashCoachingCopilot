"""Analysis service for querying processed video data."""

from collections import Counter
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.models.frame_data import FrameData
from backend.models.video import Video
from backend.schemas.analysis import (
    FrameDataResponse,
    RallySummary,
    RallyDetailResponse,
    ShotDetail,
    RallyPlayerStats,
    MatchSummaryResponse,
    ShotAnalysisResponse,
    HeatmapPoint,
    HeatmapDataResponse,
    PlayerStatsResponse,
)


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

    def get_match_summary(self, video_id: str) -> MatchSummaryResponse:
        """Get overall match statistics."""
        video = self._get_video_or_404(video_id)
        self._check_processed(video_id)

        # Base query for all frames
        frames = self.db.query(FrameData).filter(FrameData.video_id == video_id)

        # Rally frames only
        rally_frames = frames.filter(FrameData.is_rally_frame == True)

        # Count rallies
        rally_ids = (
            self.db.query(FrameData.rally_id)
            .filter(FrameData.video_id == video_id, FrameData.rally_id.isnot(None))
            .distinct()
            .all()
        )
        total_rallies = len([r for r in rally_ids if r[0] is not None])

        # Count shots (racket hits)
        total_shots = frames.filter(FrameData.is_racket_hit == True).count()
        total_wall_hits = frames.filter(FrameData.is_wall_hit == True).count()

        # Player shots
        player_1_shots = frames.filter(
            FrameData.is_racket_hit == True, FrameData.racket_hit_player_id == 1
        ).count()
        player_2_shots = frames.filter(
            FrameData.is_racket_hit == True, FrameData.racket_hit_player_id == 2
        ).count()

        # Stroke distribution
        stroke_counts = (
            self.db.query(FrameData.stroke_type, func.count(FrameData.id))
            .filter(
                FrameData.video_id == video_id,
                FrameData.stroke_type.isnot(None),
                FrameData.stroke_type != "",
            )
            .group_by(FrameData.stroke_type)
            .all()
        )
        stroke_distribution = {s: c for s, c in stroke_counts}

        # Shot type distribution
        shot_counts = (
            self.db.query(FrameData.shot_type, func.count(FrameData.id))
            .filter(
                FrameData.video_id == video_id,
                FrameData.shot_type.isnot(None),
                FrameData.shot_type != "",
            )
            .group_by(FrameData.shot_type)
            .all()
        )
        shot_type_distribution = {s: c for s, c in shot_counts}

        # Calculate average rally duration
        avg_rally_duration = 0.0
        longest_rally_id = None
        longest_rally_shots = 0

        if total_rallies > 0:
            rally_durations = []
            rally_shot_counts = []

            for (rally_id,) in rally_ids:
                if rally_id is None:
                    continue
                rally_data = frames.filter(FrameData.rally_id == rally_id)
                first_frame = rally_data.order_by(FrameData.frame_number).first()
                last_frame = rally_data.order_by(FrameData.frame_number.desc()).first()

                if first_frame and last_frame:
                    duration = last_frame.timestamp - first_frame.timestamp
                    rally_durations.append(duration)

                    shots_in_rally = rally_data.filter(FrameData.is_racket_hit == True).count()
                    rally_shot_counts.append((rally_id, shots_in_rally))

            if rally_durations:
                avg_rally_duration = sum(rally_durations) / len(rally_durations)

            if rally_shot_counts:
                longest_rally_id, longest_rally_shots = max(rally_shot_counts, key=lambda x: x[1])

        return MatchSummaryResponse(
            video_id=video_id,
            total_frames=video.total_frames or 0,
            duration_seconds=video.duration_seconds or 0.0,
            total_rallies=total_rallies,
            total_shots=total_shots,
            total_wall_hits=total_wall_hits,
            avg_rally_duration=avg_rally_duration,
            longest_rally_id=longest_rally_id,
            longest_rally_shots=longest_rally_shots,
            player_1_total_shots=player_1_shots,
            player_2_total_shots=player_2_shots,
            stroke_distribution=stroke_distribution,
            shot_type_distribution=shot_type_distribution,
        )

    def get_rallies(self, video_id: str) -> list[RallySummary]:
        """Get summary of all rallies in a video."""
        self._check_processed(video_id)

        # Get distinct rally IDs
        rally_ids = (
            self.db.query(FrameData.rally_id)
            .filter(FrameData.video_id == video_id, FrameData.rally_id.isnot(None))
            .distinct()
            .order_by(FrameData.rally_id)
            .all()
        )

        rallies = []
        for (rally_id,) in rally_ids:
            if rally_id is None:
                continue

            rally_frames = (
                self.db.query(FrameData)
                .filter(FrameData.video_id == video_id, FrameData.rally_id == rally_id)
                .order_by(FrameData.frame_number)
            )

            first_frame = rally_frames.first()
            last_frame = rally_frames.order_by(FrameData.frame_number.desc()).first()

            if not first_frame or not last_frame:
                continue

            # Count shots
            total_shots = rally_frames.filter(FrameData.is_racket_hit == True).count()
            wall_hits = rally_frames.filter(FrameData.is_wall_hit == True).count()
            player_1_shots = rally_frames.filter(
                FrameData.is_racket_hit == True, FrameData.racket_hit_player_id == 1
            ).count()
            player_2_shots = rally_frames.filter(
                FrameData.is_racket_hit == True, FrameData.racket_hit_player_id == 2
            ).count()

            rallies.append(
                RallySummary(
                    rally_id=rally_id,
                    start_frame=first_frame.frame_number,
                    end_frame=last_frame.frame_number,
                    start_timestamp=first_frame.timestamp,
                    end_timestamp=last_frame.timestamp,
                    duration_seconds=last_frame.timestamp - first_frame.timestamp,
                    total_shots=total_shots,
                    wall_hits=wall_hits,
                    player_1_shots=player_1_shots,
                    player_2_shots=player_2_shots,
                )
            )

        return rallies

    def get_rally_detail(self, video_id: str, rally_id: int) -> RallyDetailResponse:
        """Get detailed analysis of a specific rally."""
        self._check_processed(video_id)

        rally_frames = (
            self.db.query(FrameData)
            .filter(FrameData.video_id == video_id, FrameData.rally_id == rally_id)
            .order_by(FrameData.frame_number)
            .all()
        )

        if not rally_frames:
            raise ValueError(f"Rally {rally_id} not found in video {video_id}")

        first_frame = rally_frames[0]
        last_frame = rally_frames[-1]

        # Get all shots in rally
        shots = []
        player_1_strokes = {"forehand": 0, "backhand": 0}
        player_2_strokes = {"forehand": 0, "backhand": 0}
        player_1_shot_types: Counter = Counter()
        player_2_shot_types: Counter = Counter()

        for frame in rally_frames:
            if frame.is_racket_hit:
                shot = ShotDetail(
                    frame_number=frame.frame_number,
                    timestamp=frame.timestamp,
                    player_id=frame.racket_hit_player_id or 0,
                    stroke_type=frame.stroke_type,
                    shot_type=frame.shot_type,
                    shot_direction=frame.shot_direction,
                    shot_depth=frame.shot_depth,
                )
                shots.append(shot)

                # Count by player
                if frame.racket_hit_player_id == 1:
                    if frame.stroke_type == "forehand":
                        player_1_strokes["forehand"] += 1
                    elif frame.stroke_type == "backhand":
                        player_1_strokes["backhand"] += 1
                    if frame.shot_type:
                        player_1_shot_types[frame.shot_type] += 1
                elif frame.racket_hit_player_id == 2:
                    if frame.stroke_type == "forehand":
                        player_2_strokes["forehand"] += 1
                    elif frame.stroke_type == "backhand":
                        player_2_strokes["backhand"] += 1
                    if frame.shot_type:
                        player_2_shot_types[frame.shot_type] += 1

        return RallyDetailResponse(
            rally_id=rally_id,
            start_frame=first_frame.frame_number,
            end_frame=last_frame.frame_number,
            duration_seconds=last_frame.timestamp - first_frame.timestamp,
            shots=shots,
            player_1_stats=RallyPlayerStats(
                total_shots=player_1_strokes["forehand"] + player_1_strokes["backhand"],
                forehand_count=player_1_strokes["forehand"],
                backhand_count=player_1_strokes["backhand"],
                shot_types=dict(player_1_shot_types),
            ),
            player_2_stats=RallyPlayerStats(
                total_shots=player_2_strokes["forehand"] + player_2_strokes["backhand"],
                forehand_count=player_2_strokes["forehand"],
                backhand_count=player_2_strokes["backhand"],
                shot_types=dict(player_2_shot_types),
            ),
        )

    def get_frames(
        self,
        video_id: str,
        page: int = 1,
        page_size: int = 100,
        rally_id: Optional[int] = None,
    ) -> tuple[list[FrameDataResponse], int]:
        """Get frame-by-frame data with pagination."""
        self._check_processed(video_id)

        query = self.db.query(FrameData).filter(FrameData.video_id == video_id)

        if rally_id is not None:
            query = query.filter(FrameData.rally_id == rally_id)

        total = query.count()
        frames = (
            query.order_by(FrameData.frame_number)
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )

        return [FrameDataResponse.model_validate(f) for f in frames], total

    def get_shots(self, video_id: str) -> ShotAnalysisResponse:
        """Get all shots in a video with analysis."""
        self._check_processed(video_id)

        shot_frames = (
            self.db.query(FrameData)
            .filter(FrameData.video_id == video_id, FrameData.is_racket_hit == True)
            .order_by(FrameData.frame_number)
            .all()
        )

        shots = []
        stroke_counter: Counter = Counter()
        shot_type_counter: Counter = Counter()
        direction_counter: Counter = Counter()
        depth_counter: Counter = Counter()

        for frame in shot_frames:
            shots.append(
                ShotDetail(
                    frame_number=frame.frame_number,
                    timestamp=frame.timestamp,
                    player_id=frame.racket_hit_player_id or 0,
                    stroke_type=frame.stroke_type,
                    shot_type=frame.shot_type,
                    shot_direction=frame.shot_direction,
                    shot_depth=frame.shot_depth,
                )
            )

            if frame.stroke_type:
                stroke_counter[frame.stroke_type] += 1
            if frame.shot_type:
                shot_type_counter[frame.shot_type] += 1
            if frame.shot_direction:
                direction_counter[frame.shot_direction] += 1
            if frame.shot_depth:
                depth_counter[frame.shot_depth] += 1

        return ShotAnalysisResponse(
            shots=shots,
            total=len(shots),
            stroke_distribution=dict(stroke_counter),
            shot_type_distribution=dict(shot_type_counter),
            direction_distribution=dict(direction_counter),
            depth_distribution=dict(depth_counter),
        )

    def get_heatmap(self, video_id: str, player_id: int) -> HeatmapDataResponse:
        """Get player position heatmap data."""
        self._check_processed(video_id)

        x_col = f"player_{player_id}_x_meter"
        y_col = f"player_{player_id}_y_meter"

        frames = (
            self.db.query(FrameData)
            .filter(
                FrameData.video_id == video_id,
                FrameData.is_rally_frame == True,
            )
            .all()
        )

        # Aggregate positions into grid cells
        grid_size = 0.5  # meters
        position_counts: Counter = Counter()
        min_x, max_x = float("inf"), float("-inf")
        min_y, max_y = float("inf"), float("-inf")

        for frame in frames:
            x = getattr(frame, x_col)
            y = getattr(frame, y_col)

            if x is not None and y is not None:
                # Round to grid cell
                grid_x = round(x / grid_size) * grid_size
                grid_y = round(y / grid_size) * grid_size
                position_counts[(grid_x, grid_y)] += 1

                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

        points = [
            HeatmapPoint(x=pos[0], y=pos[1], count=count)
            for pos, count in position_counts.items()
        ]

        return HeatmapDataResponse(
            video_id=video_id,
            player_id=player_id,
            points=points,
            bounds={
                "min_x": min_x if min_x != float("inf") else 0,
                "max_x": max_x if max_x != float("-inf") else 0,
                "min_y": min_y if min_y != float("inf") else 0,
                "max_y": max_y if max_y != float("-inf") else 0,
            },
        )

    def get_player_stats(self, video_id: str, player_id: int) -> PlayerStatsResponse:
        """Get per-player statistics."""
        self._check_processed(video_id)

        x_col = f"player_{player_id}_x_meter"
        y_col = f"player_{player_id}_y_meter"

        # Get shots for this player
        shots = (
            self.db.query(FrameData)
            .filter(
                FrameData.video_id == video_id,
                FrameData.is_racket_hit == True,
                FrameData.racket_hit_player_id == player_id,
            )
            .all()
        )

        forehand_count = sum(1 for s in shots if s.stroke_type == "forehand")
        backhand_count = sum(1 for s in shots if s.stroke_type == "backhand")
        total_shots = len(shots)

        shot_types: Counter = Counter()
        for shot in shots:
            if shot.shot_type:
                shot_types[shot.shot_type] += 1

        # Calculate position stats
        frames = (
            self.db.query(FrameData)
            .filter(
                FrameData.video_id == video_id,
                FrameData.is_rally_frame == True,
            )
            .all()
        )

        positions_x = []
        positions_y = []

        for frame in frames:
            x = getattr(frame, x_col)
            y = getattr(frame, y_col)
            if x is not None and y is not None:
                positions_x.append(x)
                positions_y.append(y)

        avg_x = sum(positions_x) / len(positions_x) if positions_x else 0
        avg_y = sum(positions_y) / len(positions_y) if positions_y else 0

        # Approximate court coverage (convex hull area approximation)
        if len(positions_x) >= 3:
            x_range = max(positions_x) - min(positions_x)
            y_range = max(positions_y) - min(positions_y)
            coverage = x_range * y_range * 0.7  # Rough approximation
        else:
            coverage = 0

        return PlayerStatsResponse(
            video_id=video_id,
            player_id=player_id,
            total_shots=total_shots,
            forehand_count=forehand_count,
            backhand_count=backhand_count,
            forehand_percentage=(forehand_count / total_shots * 100) if total_shots > 0 else 0,
            backhand_percentage=(backhand_count / total_shots * 100) if total_shots > 0 else 0,
            shot_types=dict(shot_types),
            avg_position_x=avg_x,
            avg_position_y=avg_y,
            court_coverage_area=coverage,
        )
