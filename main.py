"""Command-line entry point for vehicle detection on video files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import (
    DEFAULT_CONF,
    DEFAULT_DRAW_ROAD_MASK,
    DEFAULT_DRAW_ROI,
    DEFAULT_IMGSZ,
    DEFAULT_IGNORE_BOTTOM_RATIO,
    DEFAULT_MODEL,
    DEFAULT_ONCOMING_ONLY,
    DEFAULT_RED_LINE_RATIO,
    DEFAULT_ROAD_MODEL,
    DEFAULT_ROAD_SEG_INTERVAL,
    DEFAULT_ROAD_SEGMENTATION,
    DEFAULT_ROAD_LOCAL_FILES_ONLY,
    DEFAULT_SAVE_VIDEO,
    DEFAULT_SAVE_PATH,
    DEFAULT_SHOW,
    DEFAULT_TRAFFIC_SIDE,
    DEFAULT_YELLOW_LINE_RATIO,
)
from src.detector import VehicleDetector
from src.video_processor import VideoProcessor


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError("Ожидается true или false.")


def parse_ratio(value: str) -> float:
    try:
        ratio = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Ожидается число от 0.0 до 1.0.") from exc

    if not 0.0 <= ratio < 1.0:
        raise argparse.ArgumentTypeError("Ожидается число от 0.0 до 1.0.")
    return ratio


def parse_roi_points(value: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    try:
        for raw_point in value.split(";"):
            raw_point = raw_point.strip()
            if not raw_point:
                continue
            raw_x, raw_y = raw_point.split(",", maxsplit=1)
            x_ratio = float(raw_x.strip())
            y_ratio = float(raw_y.strip())
            if not 0.0 <= x_ratio <= 1.0 or not 0.0 <= y_ratio <= 1.0:
                raise ValueError
            points.append((x_ratio, y_ratio))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "ROI задаётся как 'x1,y1;x2,y2;x3,y3' с координатами от 0.0 до 1.0."
        ) from exc

    if len(points) < 3:
        raise argparse.ArgumentTypeError("ROI должен содержать минимум 3 точки.")

    return points


def parse_positive_int(value: str) -> int:
    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Ожидается целое число больше 0.") from exc

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("Ожидается целое число больше 0.")
    return parsed_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-time vehicle detection on a video file with Ultralytics YOLO."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Путь к входному видеофайлу.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Путь или имя YOLO-модели. По умолчанию: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help=f"Confidence threshold. По умолчанию: {DEFAULT_CONF}",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help=f"Размер изображения для инференса. По умолчанию: {DEFAULT_IMGSZ}",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Устройство для инференса, например cpu или 0. Если не задано, выбирается автоматически.",
    )
    parser.add_argument(
        "--save",
        default=str(DEFAULT_SAVE_PATH),
        help=f"Путь сохранения результата. По умолчанию: {DEFAULT_SAVE_PATH}",
    )
    parser.add_argument(
        "--save-video",
        type=parse_bool,
        default=DEFAULT_SAVE_VIDEO,
        help="Сохранять обработанное видео: true/false. По умолчанию: true",
    )
    parser.add_argument(
        "--show",
        type=parse_bool,
        default=DEFAULT_SHOW,
        help="Показывать окно OpenCV: true/false. По умолчанию: true",
    )
    parser.add_argument(
        "--ignore-bottom-ratio",
        type=parse_ratio,
        default=DEFAULT_IGNORE_BOTTOM_RATIO,
        help=(
            "Доля нижней части кадра, где игнорируются детекции своей машины. "
            f"По умолчанию: {DEFAULT_IGNORE_BOTTOM_RATIO}"
        ),
    )
    parser.add_argument(
        "--yellow-line-ratio",
        type=parse_ratio,
        default=DEFAULT_YELLOW_LINE_RATIO,
        help=(
            "Положение жёлтой линии по высоте кадра от 0.0 до 1.0. "
            f"По умолчанию: {DEFAULT_YELLOW_LINE_RATIO}"
        ),
    )
    parser.add_argument(
        "--red-line-ratio",
        type=parse_ratio,
        default=DEFAULT_RED_LINE_RATIO,
        help=(
            "Положение красной линии по высоте кадра от 0.0 до 1.0. "
            f"По умолчанию: {DEFAULT_RED_LINE_RATIO}"
        ),
    )
    parser.add_argument(
        "--traffic-side",
        choices=("right", "left"),
        default=DEFAULT_TRAFFIC_SIDE,
        help=(
            "Сторона дорожного движения: right для правостороннего, "
            f"left для левостороннего. По умолчанию: {DEFAULT_TRAFFIC_SIDE}"
        ),
    )
    parser.add_argument(
        "--draw-roi",
        type=parse_bool,
        default=DEFAULT_DRAW_ROI,
        help="Рисовать ROI встречной полосы: true/false. По умолчанию: true",
    )
    parser.add_argument(
        "--oncoming-only",
        type=parse_bool,
        default=DEFAULT_ONCOMING_ONLY,
        help=(
            "Оставлять на видео только машины внутри ROI встречной полосы: true/false. "
            "По умолчанию: true"
        ),
    )
    parser.add_argument(
        "--oncoming-roi",
        type=parse_roi_points,
        default=None,
        help=(
            "Пользовательский ROI встречной полосы в долях кадра: "
            "'x1,y1;x2,y2;x3,y3'. Если не задан, используется ROI по --traffic-side."
        ),
    )
    parser.add_argument(
        "--road-seg",
        type=parse_bool,
        default=DEFAULT_ROAD_SEGMENTATION,
        help="Использовать SegFormer для автоматической маски дороги: true/false.",
    )
    parser.add_argument(
        "--road-model",
        default=DEFAULT_ROAD_MODEL,
        help=f"Модель сегментации дороги. По умолчанию: {DEFAULT_ROAD_MODEL}",
    )
    parser.add_argument(
        "--road-seg-interval",
        type=parse_positive_int,
        default=DEFAULT_ROAD_SEG_INTERVAL,
        help=(
            "Как часто обновлять маску дороги, в кадрах. "
            f"По умолчанию: {DEFAULT_ROAD_SEG_INTERVAL}"
        ),
    )
    parser.add_argument(
        "--draw-road-mask",
        type=parse_bool,
        default=DEFAULT_DRAW_ROAD_MASK,
        help="Рисовать маску дороги и центральную линию: true/false. По умолчанию: true",
    )
    parser.add_argument(
        "--road-local-files-only",
        type=parse_bool,
        default=DEFAULT_ROAD_LOCAL_FILES_ONLY,
        help=(
            "Загружать road segmentation модель только из локального cache: true/false. "
            f"По умолчанию: {str(DEFAULT_ROAD_LOCAL_FILES_ONLY).lower()}"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.yellow_line_ratio >= args.red_line_ratio:
        print("Ошибка: --yellow-line-ratio должен быть меньше --red-line-ratio.", file=sys.stderr)
        return 1

    try:
        detector = VehicleDetector(
            model_path=args.model,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
        )
        processor = VideoProcessor(
            detector=detector,
            source=args.source,
            save_path=args.save,
            save_video=args.save_video,
            show=args.show,
            ignore_bottom_ratio=args.ignore_bottom_ratio,
            yellow_line_ratio=args.yellow_line_ratio,
            red_line_ratio=args.red_line_ratio,
            traffic_side=args.traffic_side,
            draw_roi=args.draw_roi,
            oncoming_only=args.oncoming_only,
            oncoming_roi_points=args.oncoming_roi,
            road_segmentation=args.road_seg,
            road_model=args.road_model,
            road_seg_interval=args.road_seg_interval,
            draw_road_mask=args.draw_road_mask,
            road_local_files_only=args.road_local_files_only,
            device=args.device,
        )
        stats = processor.process()
    except Exception as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    print("\nГотово.")
    if stats.save_path is None:
        print("Сохранённое видео: не сохранялось (--save-video false)")
    else:
        print(f"Сохранённое видео: {Path(stats.save_path).resolve()}")
    print(f"Средний FPS: {stats.average_fps:.2f}")
    print(f"Обработано кадров: {stats.processed_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
