# Vehicle Oncoming Detector

Computer vision project for detecting vehicles in video and highlighting vehicles that may be in the oncoming lane.

The current implementation uses:

- Ultralytics YOLO for vehicle detection
- A configurable oncoming-lane ROI
- Optional road-area segmentation with SegFormer Cityscapes
- OpenCV video rendering and result export

Detected COCO vehicle classes:

- `car`
- `motorcycle`
- `bus`
- `truck`

## Project Structure

```text
vehicle-oncoming-detector/
|-- main.py
|-- requirements.txt
|-- README.md
|-- yolo26s.pt
|-- outputs/
|-- test_videos/
`-- src/
    |-- __init__.py
    |-- config.py
    |-- detector.py
    |-- oncoming_lane_logic.py
    |-- road_area_logic.py
    `-- video_processor.py
```

`outputs/`, `test_videos/`, virtual environments, caches, and generated video files are ignored by Git.

## Setup

Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Basic Usage

Run detection on a video:

```bash
python main.py --source path/to/video.mp4
```

By default, the project uses:

- YOLO model: `yolo26s.pt`
- confidence threshold: `0.35`
- image size: `640`
- output video: `outputs/result.mp4`
- OpenCV preview window: enabled
- road segmentation: enabled

Stop processing by pressing `q` while the OpenCV preview window is focused.

## Model Selection

Use the included local YOLO model:

```bash
python main.py --source path/to/video.mp4 --model yolo26s.pt
```

You can also pass another local `.pt` model path or a model name supported by Ultralytics.

Run on CPU:

```bash
python main.py --source path/to/video.mp4 --device cpu
```

Run on GPU:

```bash
python main.py --source path/to/video.mp4 --device 0
```

## Useful Examples

Save the processed video to a custom path:

```bash
python main.py --source path/to/video.mp4 --save outputs/demo.mp4
```

Show the result without saving a video:

```bash
python main.py --source path/to/video.mp4 --save-video false
```

Use a higher confidence threshold and larger inference image size:

```bash
python main.py --source path/to/video.mp4 --conf 0.45 --imgsz 960
```

If the camera sees the hood or dashboard of the current car and YOLO detects it as a vehicle, increase the ignored bottom area:

```bash
python main.py --source path/to/video.mp4 --ignore-bottom-ratio 0.30
```

Disable that filter:

```bash
python main.py --source path/to/video.mp4 --ignore-bottom-ratio 0
```

## Distance Warning Lines

The video overlay includes two horizontal warning lines:

- yellow `CLOSE` line
- red `NO OVERTAKING` line

The line positions are configured as ratios of frame height. The value `0.0` means the top of the frame, and `1.0` means the bottom.

Defaults:

- yellow line: `0.47`
- red line: `0.53`

Example:

```bash
python main.py --source path/to/video.mp4 --yellow-line-ratio 0.55 --red-line-ratio 0.70
```

The warning is based on the lower edge of the vehicle bounding box.

## Oncoming-Lane ROI

The project can restrict warnings to vehicles whose lower center point is inside an oncoming-lane region of interest.

For right-hand traffic, the oncoming lane is usually on the left:

```bash
python main.py --source path/to/video.mp4 --traffic-side right
```

For left-hand traffic, the oncoming lane is mirrored to the right:

```bash
python main.py --source path/to/video.mp4 --traffic-side left
```

By default, only detections inside the oncoming-lane area are drawn. To draw all detected vehicles while keeping warnings limited to the oncoming area:

```bash
python main.py --source path/to/video.mp4 --oncoming-only false
```

Hide the ROI overlay:

```bash
python main.py --source path/to/video.mp4 --draw-roi false
```

Use a custom ROI polygon with normalized frame coordinates:

```bash
python main.py --source path/to/video.mp4 --oncoming-roi "0.08,0.90;0.46,0.90;0.50,0.55;0.42,0.44;0.18,0.62"
```

Coordinates use the top-left frame corner as `0.0,0.0` and the bottom-right frame corner as `1.0,1.0`.

## Road Segmentation

Road segmentation is enabled by default:

```bash
python main.py --source path/to/video.mp4 --road-seg true
```

Default road segmentation model:

```text
nvidia/segformer-b0-finetuned-cityscapes-640-1280
```

The first run may download this model from Hugging Face if it is not available locally.

The road mask is updated every 30 frames by default:

```bash
python main.py --source path/to/video.mp4 --road-seg-interval 30
```

Use a lower interval for more accurate but slower processing:

```bash
python main.py --source path/to/video.mp4 --road-seg-interval 1
```

Disable road segmentation and use only the manual ROI:

```bash
python main.py --source path/to/video.mp4 --road-seg false
```

Hide the green road mask and road center curve:

```bash
python main.py --source path/to/video.mp4 --draw-road-mask false
```

If the SegFormer model is already cached and you want to avoid network checks:

```bash
python main.py --source path/to/video.mp4 --road-local-files-only true
```

## Command-Line Options

| Option | Description | Default |
| --- | --- | --- |
| `--source` | Path to the input video. | required |
| `--model` | YOLO model path or model name. | `yolo26s.pt` |
| `--conf` | Confidence threshold. | `0.35` |
| `--imgsz` | Inference image size. | `640` |
| `--device` | Inference device, for example `cpu` or `0`. | auto |
| `--save` | Output video path. | `outputs/result.mp4` |
| `--save-video` | Save processed video: `true` or `false`. | `true` |
| `--show` | Show OpenCV preview window: `true` or `false`. | `true` |
| `--ignore-bottom-ratio` | Bottom frame area ignored as the current vehicle. | `0.25` |
| `--yellow-line-ratio` | Yellow warning line position by frame height. | `0.47` |
| `--red-line-ratio` | Red warning line position by frame height. | `0.53` |
| `--traffic-side` | Road traffic side: `right` or `left`. | `right` |
| `--draw-roi` | Draw oncoming-lane ROI overlay. | `true` |
| `--oncoming-only` | Draw only vehicles inside the oncoming area. | `true` |
| `--oncoming-roi` | Custom normalized ROI polygon: `"x1,y1;x2,y2;x3,y3"`. | auto |
| `--road-seg` | Use road segmentation. | `true` |
| `--road-model` | Road segmentation model. | `nvidia/segformer-b0-finetuned-cityscapes-640-1280` |
| `--road-seg-interval` | Road mask update interval in frames. | `30` |
| `--draw-road-mask` | Draw road mask and center curve. | `true` |
| `--road-local-files-only` | Load the road segmentation model only from local cache. | `false` |

## Notes

`src/road_area_logic.py` builds a road mask, estimates the main road corridor, and splits it into own/oncoming sides. If a yellow separator line is visible inside the road mask, it is used as the main divider. If the yellow separator cannot be detected, the code falls back to road-mask geometry.

The road center curve is smoothed to reduce flicker from segmentation noise, hood/dash visibility, and small gaps in the road mask.
