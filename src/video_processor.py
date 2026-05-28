"""Video reading, inference loop, rendering, and output writing."""

from __future__ import annotations

import time
from pathlib import Path
from typing import NamedTuple

import cv2

from src.config import (
    DEFAULT_DRAW_ROAD_MASK,
    DEFAULT_IGNORE_BOTTOM_RATIO,
    DEFAULT_OUTPUT_FPS,
    DEFAULT_RED_LINE_RATIO,
    DEFAULT_DRAW_ROI,
    DEFAULT_ONCOMING_ONLY,
    DEFAULT_ROAD_MODEL,
    DEFAULT_ROAD_SEG_INTERVAL,
    DEFAULT_ROAD_SEGMENTATION,
    DEFAULT_ROAD_LOCAL_FILES_ONLY,
    DEFAULT_SAVE_VIDEO,
    DEFAULT_TRAFFIC_SIDE,
    DEFAULT_YELLOW_LINE_RATIO,
    WINDOW_NAME,
)
from src.detector import VehicleDetector
from src.oncoming_lane_logic import NormalizedPoint, OncomingLaneLogic
from src.road_area_logic import RoadAreaSegmenter


class ProcessingStats(NamedTuple):
    """Summary returned after processing a video."""

    save_path: Path | None
    average_fps: float
    processed_frames: int


class VideoProcessor:
    """Processes a video file frame by frame with a vehicle detector."""

    def __init__(
        self,
        detector: VehicleDetector,
        source: str | Path,
        save_path: str | Path,
        save_video: bool = DEFAULT_SAVE_VIDEO,
        show: bool = True,
        ignore_bottom_ratio: float = DEFAULT_IGNORE_BOTTOM_RATIO,
        yellow_line_ratio: float = DEFAULT_YELLOW_LINE_RATIO,
        red_line_ratio: float = DEFAULT_RED_LINE_RATIO,
        traffic_side: str = DEFAULT_TRAFFIC_SIDE,
        draw_roi: bool = DEFAULT_DRAW_ROI,
        oncoming_only: bool = DEFAULT_ONCOMING_ONLY,
        oncoming_roi_points: list[NormalizedPoint] | None = None,
        road_segmentation: bool = DEFAULT_ROAD_SEGMENTATION,
        road_model: str = DEFAULT_ROAD_MODEL,
        road_seg_interval: int = DEFAULT_ROAD_SEG_INTERVAL,
        draw_road_mask: bool = DEFAULT_DRAW_ROAD_MASK,
        road_local_files_only: bool = DEFAULT_ROAD_LOCAL_FILES_ONLY,
        device: str | None = None,
    ) -> None:
        self.detector = detector
        self.source = Path(source)
        self.save_path = Path(save_path)
        self.save_video = save_video
        self.show = show
        self.ignore_bottom_ratio = ignore_bottom_ratio
        self.yellow_line_ratio = yellow_line_ratio
        self.red_line_ratio = red_line_ratio
        self.draw_roi = draw_roi
        self.oncoming_only = oncoming_only
        self.traffic_side = traffic_side
        self.draw_road_mask = draw_road_mask
        self.oncoming_lane = OncomingLaneLogic(
            traffic_side=traffic_side,
            roi_points=oncoming_roi_points,
        )
        self.road_segmenter = (
            RoadAreaSegmenter(
                model_name=road_model,
                device=device,
                update_interval=road_seg_interval,
                local_files_only=road_local_files_only,
                bottom_ignore_ratio=ignore_bottom_ratio,
                traffic_side=traffic_side,
            )
            if road_segmentation
            else None
        )
        self._validate_line_ratios()

    def process(self) -> ProcessingStats:
        cap = cv2.VideoCapture(str(self.source))
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {self.source}")

        if self.save_video:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = DEFAULT_OUTPUT_FPS

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError(f"Не удалось определить размер видео: {self.source}")

        processed_frames = 0
        total_processing_time = 0.0
        writer: cv2.VideoWriter | None = None

        try:
            if self.save_video:
                writer = self._create_writer(self.save_path, fps, width, height)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_start = time.perf_counter()
                results = self.detector.detect(frame)
                self._filter_ego_vehicle_zone(results, frame.shape[:2])
                road_analysis = None
                if self.road_segmenter is not None:
                    road_analysis = self.road_segmenter.analyze(frame, processed_frames)
                    oncoming_mask = self.road_segmenter.get_oncoming_mask(
                        results[0].boxes,
                        road_analysis,
                        self.traffic_side,
                    )
                else:
                    oncoming_mask = self.oncoming_lane.get_oncoming_mask(
                        results[0].boxes,
                        frame.shape[:2],
                    )

                line_status = self._get_line_status(
                    results,
                    frame.shape[:2],
                    detection_mask=oncoming_mask,
                )
                if self.oncoming_only:
                    self._filter_by_detection_mask(results, oncoming_mask)
                annotated_frame = results[0].plot()
                frame_time = time.perf_counter() - frame_start

                processed_frames += 1
                total_processing_time += frame_time
                current_fps = 1.0 / frame_time if frame_time > 0 else 0.0

                # Future extension point: pass detections and frame metadata to
                # oncoming_lane_logic.py for ROI/lane checks and oncoming alerts.
                if self.road_segmenter is not None and road_analysis is not None:
                    if self.draw_road_mask:
                        self.road_segmenter.draw(
                            annotated_frame,
                            road_analysis,
                            self.traffic_side,
                        )
                elif self.draw_roi:
                    self.oncoming_lane.draw_roi(annotated_frame)
                self._draw_control_lines(annotated_frame, line_status)
                self._draw_fps(annotated_frame, current_fps)

                if writer is not None:
                    writer.write(annotated_frame)

                if self.show:
                    cv2.imshow(WINDOW_NAME, annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if self.show:
                cv2.destroyAllWindows()

        average_fps = (
            processed_frames / total_processing_time if total_processing_time > 0 else 0.0
        )

        return ProcessingStats(
            save_path=self.save_path if self.save_video else None,
            average_fps=average_fps,
            processed_frames=processed_frames,
        )

    @staticmethod
    def _create_writer(save_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Не удалось создать файл для записи видео: {save_path}")
        return writer

    @staticmethod
    def _draw_fps(frame: cv2.typing.MatLike, fps: float) -> None:
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    def _filter_ego_vehicle_zone(self, results: list, frame_shape: tuple[int, int]) -> None:
        """Remove detections that belong to the bottom ego-vehicle area."""
        if self.ignore_bottom_ratio <= 0:
            return

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return

        frame_height, frame_width = frame_shape
        cutoff_y = frame_height * (1.0 - self.ignore_bottom_ratio)
        xyxy = boxes.xyxy
        box_width = xyxy[:, 2] - xyxy[:, 0]
        box_height = xyxy[:, 3] - xyxy[:, 1]
        box_area = box_width * box_height
        center_y = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
        bottom_y = xyxy[:, 3]

        width_ratio = box_width / max(frame_width, 1)
        area_ratio = box_area / max(frame_width * frame_height, 1)
        touches_ego_zone = bottom_y >= cutoff_y
        center_in_ego_zone = center_y >= cutoff_y
        unusually_large = (width_ratio >= 0.60) | (area_ratio >= 0.25)

        remove_mask = center_in_ego_zone | (touches_ego_zone & unusually_large)
        keep_mask = ~remove_mask

        # Future extension point: replace this coarse ego-vehicle mask with
        # lane-aware ROI logic in oncoming_lane_logic.py.
        result.update(boxes=boxes.data[keep_mask])

    @staticmethod
    def _filter_by_detection_mask(results: list, detection_mask) -> None:
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return

        if detection_mask is None:
            result.update(boxes=boxes.data[:0])
            return

        result.update(boxes=boxes.data[detection_mask])

    def _validate_line_ratios(self) -> None:
        if not 0.0 <= self.yellow_line_ratio < 1.0:
            raise ValueError("yellow_line_ratio должен быть числом от 0.0 до 1.0.")
        if not 0.0 <= self.red_line_ratio < 1.0:
            raise ValueError("red_line_ratio должен быть числом от 0.0 до 1.0.")
        if self.yellow_line_ratio >= self.red_line_ratio:
            raise ValueError("yellow_line_ratio должен быть меньше red_line_ratio.")

    def _get_line_status(
        self,
        results: list,
        frame_shape: tuple[int, int],
        detection_mask=None,
    ) -> str | None:
        """Return the most severe line crossed by any current vehicle detection."""
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None

        frame_height, _ = frame_shape
        yellow_y = frame_height * self.yellow_line_ratio
        red_y = frame_height * self.red_line_ratio
        bottom_y = boxes.xyxy[:, 3]
        if detection_mask is not None:
            bottom_y = bottom_y[detection_mask]

        if len(bottom_y) == 0:
            return None

        if (bottom_y >= red_y).any().item():
            return "red"
        if (bottom_y >= yellow_y).any().item():
            return "yellow"
        return None

    def _draw_control_lines(self, frame: cv2.typing.MatLike, status: str | None) -> None:
        height, width = frame.shape[:2]
        yellow_y = int(height * self.yellow_line_ratio)
        red_y = int(height * self.red_line_ratio)

        yellow_color = (0, 255, 255)
        red_color = (0, 0, 255)

        cv2.line(frame, (0, yellow_y), (width, yellow_y), yellow_color, 3, cv2.LINE_AA)
        cv2.line(frame, (0, red_y), (width, red_y), red_color, 3, cv2.LINE_AA)

        self._draw_label(frame, "CLOSE", (20, max(24, yellow_y - 10)), yellow_color)
        self._draw_label(frame, "NO OVERTAKING", (20, max(24, red_y - 10)), red_color)

        if status == "red":
            self._draw_alert(frame, "NO OVERTAKING", red_color)
        elif status == "yellow":
            self._draw_alert(frame, "VEHICLE CLOSE", yellow_color)

    @staticmethod
    def _draw_label(
        frame: cv2.typing.MatLike,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = origin
        cv2.rectangle(
            frame,
            (x - 6, y - text_height - 6),
            (x + text_width + 6, y + baseline + 6),
            (0, 0, 0),
            -1,
        )
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _draw_alert(
        frame: cv2.typing.MatLike,
        text: str,
        color: tuple[int, int, int],
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        height, width = frame.shape[:2]
        x = max(20, width - text_width - 30)
        y = 42
        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (x - 12, y - text_height - 12),
            (width - 18, y + baseline + 12),
            color,
            -1,
        )
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
