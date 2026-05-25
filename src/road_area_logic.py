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
    ) -> None:
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.update_interval = max(1, update_interval)
        self.bottom_ignore_ratio = bottom_ignore_ratio
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

    def analyze(self, frame: cv2.typing.MatLike, frame_index: int) -> RoadAnalysis:
        """Return road analysis, reusing the previous mask between updates."""
        frame_shape = frame.shape[:2]
        should_update = (
            self._cached_analysis is None
            or self._cached_shape != frame_shape
            or frame_index % self.update_interval == 0
        )
        if not should_update:
            return self._cached_analysis

        road_mask = self._segment_road(frame)
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
        yellow_center = self._detect_yellow_separator_curve(frame, road_mask, cutoff_y)
        if yellow_center is not None:
            center_x_by_y, center_valid_by_y = yellow_center
            corridor_mask = road_mask.copy()
            corridor_mask[cutoff_y:, :] = False
            return corridor_mask, center_x_by_y, center_valid_by_y

        left_x_by_y = np.full(height, np.nan, dtype=np.float32)
        right_x_by_y = np.full(height, np.nan, dtype=np.float32)
        center_x_by_y = np.full(height, np.nan, dtype=np.float32)
        center_valid_by_y = np.zeros(height, dtype=bool)

        previous_center = width / 2.0
        previous_width = width * 0.5
        accepted_rows = 0

        for y in range(cutoff_y - 1, -1, -1):
            runs = self._find_road_runs(road_mask[y], width, y, height)
            if not runs:
                continue

            selected_run = self._select_road_run(runs, previous_center, previous_width)
            if selected_run is None:
                continue

            left_x, right_x = selected_run
            run_center = (left_x + right_x) / 2.0
            run_width = max(1.0, right_x - left_x + 1.0)

            if accepted_rows >= 4 and abs(run_center - previous_center) > width * 0.12:
                continue

            left_x_by_y[y] = float(left_x)
            right_x_by_y[y] = float(right_x)
            center_x_by_y[y] = run_center
            previous_center = (previous_center * 0.72) + (run_center * 0.28)
            previous_width = (previous_width * 0.72) + (run_width * 0.28)
            accepted_rows += 1

        valid_rows = np.flatnonzero(~np.isnan(center_x_by_y))
        if len(valid_rows) == 0:
            fallback_center = np.full(height, width / 2.0, dtype=np.float32)
            return np.zeros_like(road_mask, dtype=bool), fallback_center, center_valid_by_y

        corridor_rows = np.arange(valid_rows.min(), cutoff_y)
        left_interp = np.interp(corridor_rows, valid_rows, left_x_by_y[valid_rows])
        right_interp = np.interp(corridor_rows, valid_rows, right_x_by_y[valid_rows])
        center_interp = (left_interp + right_interp) / 2.0

        left_interp = self._smooth_curve(left_interp.astype(np.float32), window_size=51)
        right_interp = self._smooth_curve(right_interp.astype(np.float32), window_size=51)
        center_interp = self._smooth_curve(center_interp.astype(np.float32), window_size=61)
        center_interp = self._fit_center_curve(corridor_rows, center_interp)

        center_x_by_y[:] = np.interp(np.arange(height), corridor_rows, center_interp)
        center_valid_by_y[corridor_rows] = True

        corridor_mask = np.zeros_like(road_mask, dtype=bool)
        for row, left_x, right_x in zip(corridor_rows, left_interp, right_interp):
            left_idx = int(np.clip(round(left_x), 0, width - 1))
            right_idx = int(np.clip(round(right_x), 0, width - 1))
            if right_idx <= left_idx:
                continue
            corridor_mask[row, left_idx : right_idx + 1] = True

        return corridor_mask, center_x_by_y.astype(np.float32), center_valid_by_y

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
        separator_cutoff_y = max(1, cutoff_y - int(height * 0.04))
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
