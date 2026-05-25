"""YOLO-based vehicle detector."""

from __future__ import annotations

from typing import Any

import cv2
from ultralytics import YOLO

from src.config import DEFAULT_CONF, DEFAULT_DEVICE, DEFAULT_IMGSZ, DEFAULT_MODEL, TRANSPORT_CLASSES


class VehicleDetector:
    """Runs YOLO inference for COCO vehicle classes."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        conf: float = DEFAULT_CONF,
        imgsz: int = DEFAULT_IMGSZ,
        device: str | None = DEFAULT_DEVICE,
    ) -> None:
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.classes = TRANSPORT_CLASSES

    def detect(self, frame: cv2.typing.MatLike) -> list[Any]:
        """Detect vehicles on a single OpenCV frame."""
        predict_kwargs: dict[str, Any] = {
            "source": frame,
            "classes": self.classes,
            "conf": self.conf,
            "imgsz": self.imgsz,
            "verbose": False,
        }

        if self.device is not None:
            predict_kwargs["device"] = self.device

        return self.model.predict(**predict_kwargs)

