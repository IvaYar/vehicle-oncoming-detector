"""ROI-based oncoming lane logic.

This module intentionally starts with a configurable polygon ROI instead of
classic Canny/Hough lane detection. It is more predictable for an MVP and can
later be replaced by lane or drivable-area segmentation.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import torch


NormalizedPoint = tuple[float, float]


DEFAULT_ROI_POINTS: dict[str, tuple[NormalizedPoint, ...]] = {
    # Right-hand traffic: oncoming lane is usually on the left side of the road.
    "right": (
        (0.06, 0.92),
        (0.46, 0.92),
        (0.50, 0.55),
        (0.42, 0.43),
        (0.18, 0.62),
    ),
    # Left-hand traffic: mirrored default ROI.
    "left": (
        (0.94, 0.92),
        (0.54, 0.92),
        (0.50, 0.55),
        (0.58, 0.43),
        (0.82, 0.62),
    ),
}


class OncomingLaneLogic:
    """Checks whether vehicle detections are inside the oncoming-lane ROI."""

    def __init__(
        self,
        traffic_side: str = "right",
        roi_points: Sequence[NormalizedPoint] | None = None,
    ) -> None:
        if traffic_side not in DEFAULT_ROI_POINTS:
            raise ValueError("traffic_side должен быть 'right' или 'left'.")

        self.traffic_side = traffic_side
        self.roi_points = tuple(roi_points or DEFAULT_ROI_POINTS[traffic_side])
        self._validate_roi_points()

    def get_oncoming_mask(self, boxes, frame_shape: tuple[int, int]) -> torch.Tensor | None:
        """Return a boolean mask for boxes whose anchor point is inside ROI."""
        if boxes is None or len(boxes) == 0:
            return None

        polygon = self.get_roi_polygon(frame_shape)
        xyxy = boxes.xyxy.detach().cpu().numpy()
        mask_values = []

        for x1, _y1, x2, y2 in xyxy:
            anchor_point = (float((x1 + x2) / 2.0), float(y2))
            is_inside = cv2.pointPolygonTest(polygon, anchor_point, False) >= 0
            mask_values.append(is_inside)

        return torch.tensor(mask_values, dtype=torch.bool, device=boxes.data.device)

    def get_roi_polygon(self, frame_shape: tuple[int, int]) -> np.ndarray:
        """Convert normalized ROI points to OpenCV pixel polygon."""
        height, width = frame_shape
        points = [
            (int(x_ratio * width), int(y_ratio * height))
            for x_ratio, y_ratio in self.roi_points
        ]
        return np.array(points, dtype=np.int32).reshape((-1, 1, 2))

    def draw_roi(self, frame: cv2.typing.MatLike) -> None:
        """Draw the oncoming-lane ROI on the frame."""
        polygon = self.get_roi_polygon(frame.shape[:2])
        overlay = frame.copy()
        fill_color = (255, 160, 0)
        border_color = (255, 220, 80)

        cv2.fillPoly(overlay, [polygon], fill_color)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.polylines(frame, [polygon], True, border_color, 2, cv2.LINE_AA)
        self._draw_roi_label(frame, polygon)

    def _validate_roi_points(self) -> None:
        if len(self.roi_points) < 3:
            raise ValueError("ROI должен содержать минимум 3 точки.")

        for x_ratio, y_ratio in self.roi_points:
            if not 0.0 <= x_ratio <= 1.0 or not 0.0 <= y_ratio <= 1.0:
                raise ValueError("Каждая точка ROI должна быть в диапазоне от 0.0 до 1.0.")

    @staticmethod
    def _draw_roi_label(frame: cv2.typing.MatLike, polygon: np.ndarray) -> None:
        x = int(polygon[:, 0, 0].min())
        y = int(polygon[:, 0, 1].min())
        origin = (max(12, x), max(28, y - 10))
        cv2.putText(
            frame,
            "ONCOMING ROI",
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 220, 80),
            2,
            cv2.LINE_AA,
        )
