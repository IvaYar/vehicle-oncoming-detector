"""Road area segmentation and oncoming-side classification."""

from __future__ import annotations

from dataclasses import dataclass
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class RoadAnalysis:
    """Road mask and center curve for a video frame."""

    road_mask: np.ndarray
    road_corridor_mask: np.ndarray
    center_x_by_y: np.ndarray
    center_valid_by_y: np.ndarray


class RoadAreaSegmenter:
    """Segments the road area and splits it into own/oncoming sides."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        update_interval: int = 10,
        local_files_only: bool = False,
        bottom_ignore_ratio: float = 0.25,
        traffic_side: str = "right",
    ) -> None:
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.update_interval = max(1, update_interval)
        self.bottom_ignore_ratio = bottom_ignore_ratio
        self.traffic_side = traffic_side
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

        try:
            from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        except ImportError as exc:
            raise RuntimeError(
                "Для --road-seg true нужны зависимости transformers и pillow. "
                "Установите их командой: pip install -r requirements.txt"
            ) from exc

        try:
            self.image_processor = AutoImageProcessor.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                use_safetensors=False,
            )
        except OSError as exc:
            raise RuntimeError(
                "Не удалось загрузить модель сегментации дороги. "
                "Для первого запуска нужен интернет или локальный путь в --road-model. "
                "Если модель уже была скачана, попробуйте --road-local-files-only true."
            ) from exc

        self.model.to(self.device)
        self.model.eval()

        self.road_label_ids = self._find_road_label_ids()
        self._cached_analysis: RoadAnalysis | None = None
        self._cached_shape: tuple[int, int] | None = None
        self._missed_separator_frames = 0
        self._max_missed_separator_frames = 18

    def analyze(self, frame: cv2.typing.MatLike, frame_index: int) -> RoadAnalysis:
        """Return road analysis, reusing the heavy road mask between updates."""
        frame_shape = frame.shape[:2]
        can_reuse_mask = self._cached_analysis is not None and self._cached_shape == frame_shape
        should_update_mask = not can_reuse_mask or frame_index % self.update_interval == 0

        if should_update_mask:
            road_mask = self._segment_road(frame)
        else:
            road_mask = self._cached_analysis.road_mask

        road_corridor_mask, center_x_by_y, center_valid_by_y = self._build_road_corridor(
            frame,
            road_mask,
        )
        analysis = RoadAnalysis(
            road_mask=road_mask,
            road_corridor_mask=road_corridor_mask,
            center_x_by_y=center_x_by_y,
            center_valid_by_y=center_valid_by_y,
        )
        analysis = self._stabilize_analysis(analysis)
        self._cached_analysis = analysis
        self._cached_shape = frame_shape
        return analysis

    def get_oncoming_mask(
        self,
        boxes,
        analysis: RoadAnalysis,
        traffic_side: str,
    ) -> torch.Tensor | None:
        """Return mask for detections whose anchor is on the oncoming road side."""
        if boxes is None or len(boxes) == 0:
            return None

        height, width = analysis.road_mask.shape
        xyxy = boxes.xyxy.detach().cpu().numpy()
        mask_values = []

        for x1, _y1, x2, y2 in xyxy:
            anchor_x = int(np.clip((x1 + x2) / 2.0, 0, width - 1))
            anchor_y = int(np.clip(y2, 0, height - 1))
            is_on_road = bool(analysis.road_corridor_mask[anchor_y, anchor_x])
            has_center = bool(analysis.center_valid_by_y[anchor_y])
            center_x = float(analysis.center_x_by_y[anchor_y])

            if traffic_side == "right":
                is_oncoming_side = anchor_x < center_x
            else:
                is_oncoming_side = anchor_x > center_x

            mask_values.append(is_on_road and has_center and is_oncoming_side)

        return torch.tensor(mask_values, dtype=torch.bool, device=boxes.data.device)

    def draw(
        self,
        frame: cv2.typing.MatLike,
        analysis: RoadAnalysis,
        traffic_side: str,
    ) -> None:
        """Draw road mask and road center curve."""
        overlay = frame.copy()
        road_color = (80, 180, 80)
        overlay[analysis.road_corridor_mask] = road_color
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        points = []
        valid_rows = np.flatnonzero(analysis.center_valid_by_y)
        for y in valid_rows[::12]:
            center_x = int(analysis.center_x_by_y[y])
            points.append((center_x, y))

        if len(points) >= 2:
            cv2.polylines(
                frame,
                [np.array(points, dtype=np.int32)],
                False,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

        side_label = "ONCOMING: LEFT" if traffic_side == "right" else "ONCOMING: RIGHT"
        cv2.putText(
            frame,
            side_label,
            (20, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _stabilize_analysis(self, analysis: RoadAnalysis) -> RoadAnalysis:
        """Reuse or smooth the center curve to avoid one-frame jumps."""
        previous = self._cached_analysis
        if previous is None or previous.road_mask.shape != analysis.road_mask.shape:
            return analysis

        if not self._has_usable_center(analysis):
            self._missed_separator_frames += 1
            if self._has_usable_center(previous):
                if self._missed_separator_frames <= self._max_missed_separator_frames:
                    return previous
            return analysis

        if not self._has_usable_center(previous):
            self._missed_separator_frames = 0
            return analysis

        height, width = analysis.road_mask.shape
        stable_rows = np.flatnonzero(
            analysis.center_valid_by_y & previous.center_valid_by_y
        )
        stable_rows = stable_rows[
            (stable_rows >= int(height * 0.48))
            & (stable_rows <= int(height * (1.0 - self.bottom_ignore_ratio)))
        ]
        if len(stable_rows) < 25:
            return analysis

        center_diff = np.abs(
            analysis.center_x_by_y[stable_rows] - previous.center_x_by_y[stable_rows]
        )
        median_diff = float(np.median(center_diff))

        if median_diff > width * 0.12:
            self._missed_separator_frames += 1
            if self._missed_separator_frames <= self._max_missed_separator_frames:
                return previous
            self._missed_separator_frames = 0
            return analysis

        self._missed_separator_frames = 0

        center_x_by_y = analysis.center_x_by_y.copy()
        center_valid_by_y = analysis.center_valid_by_y.copy()
        common_rows = analysis.center_valid_by_y & previous.center_valid_by_y
        previous_only_rows = previous.center_valid_by_y & ~analysis.center_valid_by_y

        center_x_by_y[common_rows] = (
            0.60 * previous.center_x_by_y[common_rows]
            + 0.40 * analysis.center_x_by_y[common_rows]
        )
        center_x_by_y[previous_only_rows] = previous.center_x_by_y[previous_only_rows]
        center_valid_by_y[previous_only_rows] = True

        return RoadAnalysis(
            road_mask=analysis.road_mask,
            road_corridor_mask=analysis.road_corridor_mask,
            center_x_by_y=center_x_by_y.astype(np.float32),
            center_valid_by_y=center_valid_by_y,
        )

    def _segment_road(self, frame: cv2.typing.MatLike) -> np.ndarray:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.image_processor(images=rgb_frame, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = F.interpolate(
                outputs.logits,
                size=frame.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
            predicted = logits.argmax(dim=1)[0].detach().cpu().numpy()

        road_mask = np.isin(predicted, self.road_label_ids)
        return self._clean_road_mask(road_mask)

    def _clean_road_mask(self, road_mask: np.ndarray) -> np.ndarray:
        mask = road_mask.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        components_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
        if components_count <= 1:
            return mask.astype(bool)

        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return labels == largest_label

    def _build_road_corridor(
        self,
        frame: cv2.typing.MatLike,
        road_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build a stable main-road corridor and its center curve."""
        height, width = road_mask.shape
        cutoff_y = int(height * (1.0 - self.bottom_ignore_ratio))
        cutoff_y = int(np.clip(cutoff_y, 1, height))
        separator_center = self._detect_separator_curve(frame, road_mask, cutoff_y)
        if separator_center is not None:
            center_x_by_y, center_valid_by_y = separator_center
            corridor_mask = road_mask.copy()
            corridor_mask[cutoff_y:, :] = False
            return corridor_mask, center_x_by_y, center_valid_by_y

        # If no painted separator is visible, avoid inventing a center line from
        # a noisy road mask. Temporal stabilization can reuse the last good line.
        corridor_mask = road_mask.copy()
        corridor_mask[cutoff_y:, :] = False
        center_x_by_y = np.full(height, width / 2.0, dtype=np.float32)
        center_valid_by_y = np.zeros(height, dtype=bool)
        return corridor_mask, center_x_by_y, center_valid_by_y

    def _detect_separator_curve(
        self,
        frame: cv2.typing.MatLike,
        road_mask: np.ndarray,
        cutoff_y: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Detect a painted road separator before falling back to mask geometry."""
        hough_center = self._detect_hough_separator_curve(frame, road_mask, cutoff_y)
        if hough_center is not None:
            return hough_center
        return self._detect_yellow_separator_curve(frame, road_mask, cutoff_y)

    def _detect_hough_separator_curve(
        self,
        frame: cv2.typing.MatLike,
        road_mask: np.ndarray,
        cutoff_y: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Find the central road separator from diagonal lane-marking edges."""
        height, width = road_mask.shape
        separator_cutoff_y = int(np.clip(cutoff_y - int(height * 0.03), 1, height))
        roi_top_y = int(height * 0.42)
        if separator_cutoff_y <= roi_top_y:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)

        road_support = cv2.dilate(
            road_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        )
        roi = np.zeros_like(road_support, dtype=np.uint8)
        polygon = np.array(
            [
                [
                    (int(width * 0.06), separator_cutoff_y),
                    (int(width * 0.36), roi_top_y),
                    (int(width * 0.64), roi_top_y),
                    (int(width * 0.94), separator_cutoff_y),
                ]
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(roi, polygon, 1)
        edges = cv2.bitwise_and(edges, edges, mask=(roi & road_support))

        min_line_length = max(50, int(width * 0.035))
        max_line_gap = max(20, int(width * 0.015))
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=45,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )
        if lines is None:
            return None

        candidates = []
        expected_sign = -1.0 if self.traffic_side == "right" else 1.0
        preferred_bottom_x = 0.40 if self.traffic_side == "right" else 0.60
        preferred_top_x = 0.52 if self.traffic_side == "right" else 0.48

        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(value) for value in line]
            dx = x2 - x1
            dy = y2 - y1
            if dy == 0:
                continue

            slope = dx / dy
            slope_abs = abs(slope)
            if np.sign(slope) != expected_sign:
                continue
            if slope_abs < 0.35 or slope_abs > 6.0:
                continue

            length = float(np.hypot(dx, dy))
            if length < min_line_length:
                continue

            if y1 >= y2:
                bottom_x, bottom_y = x1, y1
                top_x, top_y = x2, y2
            else:
                bottom_x, bottom_y = x2, y2
                top_x, top_y = x1, y1

            if bottom_y < height * 0.50 or top_y > separator_cutoff_y:
                continue

            bottom_ratio = bottom_x / width
            top_ratio = top_x / width
            if self.traffic_side == "right":
                if not 0.28 <= bottom_ratio <= 0.68:
                    continue
            else:
                if not 0.32 <= bottom_ratio <= 0.72:
                    continue
            if self.traffic_side == "right":
                if not 0.34 <= top_ratio <= 0.62:
                    continue
            else:
                if not 0.38 <= top_ratio <= 0.70:
                    continue

            position_score = 360.0 * abs(bottom_ratio - preferred_bottom_x)
            vanish_score = 180.0 * abs(top_ratio - preferred_top_x)
            slope_score = 18.0 * abs(slope_abs - 1.6)
            temporal_score = self._get_temporal_line_penalty(
                bottom_x,
                bottom_y,
                top_x,
                top_y,
                width,
            )
            score = length - position_score - vanish_score - slope_score - temporal_score
            candidates.append(
                {
                    "score": score,
                    "line": (x1, y1, x2, y2),
                    "slope": slope,
                    "bottom_ratio": bottom_ratio,
                    "top_ratio": top_ratio,
                }
            )

        if not candidates:
            return None

        best = max(candidates, key=lambda item: item["score"])
        selected = []
        max_distance = max(55.0, width * 0.045)
        for candidate in candidates:
            x1, y1, x2, y2 = candidate["line"]
            middle_x = (x1 + x2) / 2.0
            middle_y = int((y1 + y2) / 2.0)
            if y1 >= y2:
                bottom_x, bottom_y = x1, y1
            else:
                bottom_x, bottom_y = x2, y2

            middle_distance = abs(middle_x - self._x_on_line_at_y(best["line"], middle_y))
            bottom_distance = abs(bottom_x - self._x_on_line_at_y(best["line"], bottom_y))
            if middle_distance <= max_distance and bottom_distance <= max_distance * 1.35:
                selected.append(candidate)

        if not selected:
            selected = [best]

        raw_center_by_y = np.full(height, np.nan, dtype=np.float32)
        sampled_by_y: dict[int, list[float]] = {}
        for candidate in selected:
            x1, y1, x2, y2 = candidate["line"]
            y_start = max(min(y1, y2), roi_top_y)
            y_end = min(max(y1, y2), separator_cutoff_y - 1)
            if y_end <= y_start:
                continue

            for y in range(y_start, y_end + 1):
                x = self._x_on_line_at_y((x1, y1, x2, y2), y)
                if 0 <= x < width:
                    sampled_by_y.setdefault(y, []).append(x)

        for y, xs in sampled_by_y.items():
            raw_center_by_y[y] = float(np.median(xs))

        valid_rows = np.flatnonzero(~np.isnan(raw_center_by_y))
        if len(valid_rows) < 25:
            return None

        y_min = int(valid_rows.min())
        y_max = int(valid_rows.max())
        if y_max - y_min < int(height * 0.08):
            return None

        y_start = max(roi_top_y, y_min - int(height * 0.03))
        y_end = min(separator_cutoff_y, y_max + int(height * 0.05))
        rows = np.arange(y_start, y_end)
        center_interp = self._fit_separator_curve(
            valid_rows,
            raw_center_by_y[valid_rows],
            rows,
        )
        center_interp = self._smooth_curve(center_interp, window_size=35)
        center_interp = np.clip(center_interp, width * 0.06, width * 0.94)

        center_x_by_y = np.full(height, np.nan, dtype=np.float32)
        center_x_by_y[:] = np.interp(np.arange(height), rows, center_interp).astype(np.float32)
        center_valid_by_y = np.zeros(height, dtype=bool)
        center_valid_by_y[rows] = True
        return center_x_by_y, center_valid_by_y

    def _detect_yellow_separator_curve(
        self,
        frame: cv2.typing.MatLike,
        road_mask: np.ndarray,
        cutoff_y: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Detect a yellow center separator inside the segmented road mask."""
        height, width = road_mask.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, (12, 45, 80), (42, 255, 255)).astype(bool)
        yellow_mask &= road_mask
        separator_cutoff_y = max(1, cutoff_y)
        yellow_mask[separator_cutoff_y:, :] = False
        yellow_mask[: int(height * 0.30), :] = False

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        yellow_mask_uint8 = yellow_mask.astype(np.uint8)
        yellow_mask_uint8 = cv2.morphologyEx(yellow_mask_uint8, cv2.MORPH_OPEN, kernel)
        yellow_mask_uint8 = cv2.morphologyEx(yellow_mask_uint8, cv2.MORPH_CLOSE, kernel)
        yellow_mask = yellow_mask_uint8.astype(bool)

        raw_center_by_y = np.full(height, np.nan, dtype=np.float32)
        min_pixels_per_row = max(4, int(width * 0.002))

        for y in range(height):
            yellow_x = np.flatnonzero(yellow_mask[y])
            if len(yellow_x) >= min_pixels_per_row:
                raw_center_by_y[y] = float(np.median(yellow_x))

        valid_rows = np.flatnonzero(~np.isnan(raw_center_by_y))
        if len(valid_rows) < 25:
            return None

        y_min = int(valid_rows.min())
        y_max = int(valid_rows.max())
        if y_max - y_min < int(height * 0.12):
            return None

        center_x_by_y = np.full(height, np.nan, dtype=np.float32)
        rows = np.arange(y_min, separator_cutoff_y)
        center_interp = np.interp(rows, valid_rows, raw_center_by_y[valid_rows]).astype(np.float32)
        center_interp = self._smooth_curve(center_interp, window_size=45)
        center_interp = self._fit_center_curve(rows, center_interp)

        center_x_by_y[:] = np.interp(np.arange(height), rows, center_interp).astype(np.float32)
        center_valid_by_y = np.zeros(height, dtype=bool)
        center_valid_by_y[rows] = True
        return center_x_by_y, center_valid_by_y

    @staticmethod
    def _find_road_runs(
        row_mask: np.ndarray,
        frame_width: int,
        row_y: int,
        frame_height: int,
    ) -> list[tuple[int, int]]:
        road_x = np.flatnonzero(row_mask)
        if len(road_x) == 0:
            return []

        min_run_width = max(18, int(frame_width * (0.015 + 0.035 * row_y / frame_height)))
        split_points = np.where(np.diff(road_x) > 1)[0] + 1
        groups = np.split(road_x, split_points)
        runs = []

        for group in groups:
            if len(group) >= min_run_width:
                runs.append((int(group[0]), int(group[-1])))

        return runs

    @staticmethod
    def _select_road_run(
        runs: list[tuple[int, int]],
        previous_center: float,
        previous_width: float,
    ) -> tuple[int, int] | None:
        if not runs:
            return None

        best_score = float("inf")
        best_run = None

        for left_x, right_x in runs:
            run_center = (left_x + right_x) / 2.0
            run_width = max(1.0, right_x - left_x + 1.0)
            center_distance = abs(run_center - previous_center)
            width_penalty = abs(run_width - previous_width) * 0.10
            contains_bonus = -previous_width * 0.15 if left_x <= previous_center <= right_x else 0.0
            score = center_distance + width_penalty + contains_bonus

            if score < best_score:
                best_score = score
                best_run = (left_x, right_x)

        return best_run

    @staticmethod
    def _smooth_curve(values: np.ndarray, window_size: int) -> np.ndarray:
        if len(values) < window_size:
            return values

        kernel = np.ones(window_size, dtype=np.float32) / window_size
        padded = np.pad(values, (window_size // 2, window_size // 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid").astype(np.float32)

    @staticmethod
    def _fit_center_curve(rows: np.ndarray, center_values: np.ndarray) -> np.ndarray:
        """Fit a smooth road center curve while keeping support for turns."""
        if len(rows) < 20:
            return center_values

        y_min = float(rows.min())
        y_range = max(float(rows.max() - rows.min()), 1.0)
        y_normalized = (rows.astype(np.float32) - y_min) / y_range
        degree = min(3, len(rows) - 1)
        weights = 0.35 + 0.65 * y_normalized

        try:
            coefficients = np.polyfit(
                y_normalized,
                center_values.astype(np.float32),
                degree,
                w=weights,
            )
        except np.linalg.LinAlgError:
            return center_values

        fitted_values = np.polyval(coefficients, y_normalized).astype(np.float32)
        return (0.25 * center_values + 0.75 * fitted_values).astype(np.float32)

    @staticmethod
    def _fit_separator_curve(
        valid_rows: np.ndarray,
        center_values: np.ndarray,
        target_rows: np.ndarray,
    ) -> np.ndarray:
        """Fit the painted separator and allow a small controlled extrapolation."""
        if len(valid_rows) < 3:
            return np.interp(target_rows, valid_rows, center_values).astype(np.float32)

        y_min = float(valid_rows.min())
        y_range = max(float(valid_rows.max() - valid_rows.min()), 1.0)
        valid_y = (valid_rows.astype(np.float32) - y_min) / y_range
        target_y = (target_rows.astype(np.float32) - y_min) / y_range
        row_span = int(valid_rows.max() - valid_rows.min())
        degree = min(2, len(valid_rows) - 1)
        if row_span < 120:
            degree = 1
        weights = 0.35 + 0.65 * valid_y

        try:
            coefficients = np.polyfit(
                valid_y,
                center_values.astype(np.float32),
                degree,
                w=weights,
            )
        except np.linalg.LinAlgError:
            return np.interp(target_rows, valid_rows, center_values).astype(np.float32)

        fitted_values = np.polyval(coefficients, target_y).astype(np.float32)
        interpolated_values = np.interp(target_rows, valid_rows, center_values).astype(np.float32)
        inside_support = (target_rows >= valid_rows.min()) & (target_rows <= valid_rows.max())
        return np.where(
            inside_support,
            (0.35 * interpolated_values) + (0.65 * fitted_values),
            fitted_values,
        ).astype(np.float32)

    @staticmethod
    def _x_on_line_at_y(line: tuple[int, int, int, int], y: int) -> float:
        x1, y1, x2, y2 = line
        if y2 == y1:
            return float((x1 + x2) / 2.0)
        ratio = (y - y1) / (y2 - y1)
        return float(x1 + ratio * (x2 - x1))

    @staticmethod
    def _has_usable_center(analysis: RoadAnalysis) -> bool:
        return int(np.count_nonzero(analysis.center_valid_by_y)) >= 25

    def _get_temporal_line_penalty(
        self,
        bottom_x: int,
        bottom_y: int,
        top_x: int,
        top_y: int,
        width: int,
    ) -> float:
        previous = self._cached_analysis
        if previous is None or not self._has_usable_center(previous):
            return 0.0

        height = previous.road_mask.shape[0]
        bottom_y = int(np.clip(bottom_y, 0, height - 1))
        top_y = int(np.clip(top_y, 0, height - 1))
        if not previous.center_valid_by_y[bottom_y] or not previous.center_valid_by_y[top_y]:
            return 0.0

        bottom_diff = abs(bottom_x - float(previous.center_x_by_y[bottom_y])) / width
        top_diff = abs(top_x - float(previous.center_x_by_y[top_y])) / width
        return (bottom_diff * 520.0) + (top_diff * 300.0)

    def _find_road_label_ids(self) -> list[int]:
        id_to_label = self.model.config.id2label
        road_ids = [
            int(label_id)
            for label_id, label_name in id_to_label.items()
            if str(label_name).lower() == "road"
        ]
        if not road_ids:
            raise RuntimeError(f"Модель {self.model_name} не содержит класса road.")
        return road_ids

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        if device is None:
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if device.isdigit():
            return f"cuda:{device}"
        return device
