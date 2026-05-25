# Vehicle Oncoming Detector

MVP-проект для real-time computer vision detection транспорта на видео. Сейчас проект детектит транспортные классы COCO:

- `car`
- `motorcycle`
- `bus`
- `truck`

В дальнейшем сюда удобно добавить модуль `oncoming_lane_logic.py` для определения машин на встречной полосе.

## Структура

```text
vehicle-oncoming-detector/
├── main.py
├── requirements.txt
├── README.md
├── outputs/
└── src/
    ├── __init__.py
    ├── detector.py
    ├── oncoming_lane_logic.py
    ├── road_area_logic.py
    ├── video_processor.py
    └── config.py
```

## Создание виртуального окружения

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

## Установка зависимостей

```bash
pip install -r requirements.txt
```

Для автоматического определения проезжей части используются `transformers` и `pillow`. При первом запуске модель сегментации дороги будет скачана в cache Hugging Face, если её ещё нет локально.

## Запуск

```bash
python main.py --source path/to/video.mp4
```

По умолчанию используется модель `yolo26s.pt`, confidence threshold `0.35`, размер изображения `640`, сохранение в `outputs/result.mp4` и показ окна OpenCV.

## Выбор модели

```bash
python main.py --source path/to/video.mp4 --model yolo26s.pt
```

Можно указать путь к локальному `.pt` файлу или имя модели, поддерживаемое Ultralytics.

## Запуск на CPU

```bash
python main.py --source path/to/video.mp4 --device cpu
```

## Запуск на GPU

```bash
python main.py --source path/to/video.mp4 --device 0
```

## Полезные параметры

```bash
python main.py --source path/to/video.mp4 --conf 0.45 --imgsz 960 --save outputs/demo.mp4 --show false
```

Чтобы проверить результат в окне OpenCV без сохранения файла в `outputs/`:

```bash
python main.py --source test_videos/car_test1.mp4 --road-local-files-only true --save-video false
```

Если камера видит капот или торпеду своей машины и модель детектит её как `car`, можно настроить нижнюю игнорируемую зону кадра:

```bash
python main.py --source test_videos/cars.mp4 --ignore-bottom-ratio 0.25
```

Значение `0.25` означает, что нижние 25% кадра используются как зона своей машины. Детекции с центром в этой зоне, а также очень большие рамки, которые касаются этой зоны, не будут отрисованы. Если рамка торпеды всё ещё появляется, попробуйте `0.30` или `0.35`.

Чтобы выключить фильтр:

```bash
python main.py --source test_videos/cars.mp4 --ignore-bottom-ratio 0
```

Для контроля приближения транспорта на кадре рисуются две горизонтальные линии:

- жёлтая `CLOSE` - машина находится близко;
- красная `NO OVERTAKING` - после этой линии обгон запрещён.

По умолчанию жёлтая линия находится на `0.45` высоты кадра, красная - на `0.55`. Значения считаются сверху вниз: `0.0` - верх кадра, `1.0` - низ кадра.

```bash
python main.py --source test_videos/cars.mp4 --yellow-line-ratio 0.55 --red-line-ratio 0.70
```

Пересечение определяется по нижней границе bounding box машины. Без object tracking это означает, что статус отображается, пока хотя бы одна текущая детекция находится ниже соответствующей линии.

## ROI встречной полосы

Чтобы не реагировать на все машины в кадре, проект использует ROI встречной полосы. Для каждой детекции берётся нижняя центральная точка bbox:

```text
x = (x1 + x2) / 2
y = y2
```

Если эта точка находится внутри ROI, машина считается кандидатом на встречной полосе. Жёлтая и красная линии срабатывают только для таких машин.

Для правостороннего движения встречная зона по умолчанию находится слева:

```bash
python main.py --source test_videos/cars.mp4 --traffic-side right
```

Для левостороннего движения встречная зона зеркалируется вправо:

```bash
python main.py --source test_videos/cars.mp4 --traffic-side left
```

По умолчанию на видео остаются только машины внутри ROI встречной полосы. Чтобы видеть все bbox, но оставлять предупреждения только по встречному ROI:

```bash
python main.py --source test_videos/cars.mp4 --oncoming-only false
```

ROI рисуется на кадре полупрозрачным полигоном. Его можно скрыть:

```bash
python main.py --source test_videos/cars.mp4 --draw-roi false
```

Если стандартный ROI плохо подходит под камеру, можно задать свой полигон в долях кадра:

```bash
python main.py --source test_videos/cars.mp4 --oncoming-roi "0.08,0.90;0.46,0.90;0.50,0.55;0.42,0.44;0.18,0.62"
```

Координаты считаются от верхнего левого угла: `0.0,0.0` - левый верх, `1.0,1.0` - правый низ.

## Автоматическая маска дороги

По умолчанию включена сегментация проезжей части через SegFormer Cityscapes:

```bash
python main.py --source test_videos/cars.mp4 --road-seg true
```

Модель по умолчанию:

```text
nvidia/segformer-b0-finetuned-cityscapes-640-1280
```

Логика:

```text
YOLO bbox машины
-> нижняя центральная точка bbox
-> точка должна быть внутри mask road
-> road mask делится центральной кривой на две стороны
-> traffic-side right: встречка слева от центра дороги
-> traffic-side left: встречка справа от центра дороги
-> жёлтая/красная линии срабатывают только для таких машин
```

Маска дороги обновляется не каждый кадр, а раз в несколько кадров:

```bash
python main.py --source test_videos/cars.mp4 --road-seg-interval 30
```

Для более точной, но медленной работы:

```bash
python main.py --source test_videos/cars.mp4 --road-seg-interval 1
```

Чтобы отключить автоматическую маску дороги и вернуться к ручному ROI:

```bash
python main.py --source test_videos/cars.mp4 --road-seg false
```

Чтобы скрыть зелёную маску дороги и центральную линию:

```bash
python main.py --source test_videos/cars.mp4 --draw-road-mask false
```

Если модель уже скачана и нужно запускаться без сетевых проверок:

```bash
python main.py --source test_videos/cars.mp4 --road-local-files-only true
```

Параметры командной строки:

- `--source` - путь к входному видео.
- `--model` - путь или имя модели, по умолчанию `yolo26s.pt`.
- `--conf` - confidence threshold, по умолчанию `0.35`.
- `--imgsz` - размер изображения для инференса, по умолчанию `640`.
- `--device` - устройство, например `cpu` или `0`.
- `--save` - путь сохранения результата, по умолчанию `outputs/result.mp4`.
- `--save-video` - сохранять обработанное видео, `true` или `false`.
- `--show` - показывать ли окно OpenCV, `true` или `false`.
- `--ignore-bottom-ratio` - доля нижней части кадра, где игнорируются детекции своей машины, по умолчанию `0.25`.
- `--yellow-line-ratio` - положение жёлтой линии по высоте кадра, по умолчанию `0.45`.
- `--red-line-ratio` - положение красной линии по высоте кадра, по умолчанию `0.55`.
- `--traffic-side` - сторона дорожного движения: `right` или `left`, по умолчанию `right`.
- `--draw-roi` - рисовать ROI встречной полосы, `true` или `false`.
- `--oncoming-only` - оставлять bbox только внутри встречного ROI, `true` или `false`.
- `--oncoming-roi` - пользовательский ROI в формате `"x1,y1;x2,y2;x3,y3"`.
- `--road-seg` - использовать SegFormer для автоматической маски дороги, `true` или `false`.
- `--road-model` - модель сегментации дороги.
- `--road-seg-interval` - как часто обновлять маску дороги, в кадрах.
- `--draw-road-mask` - рисовать маску дороги и центральную линию, `true` или `false`.
- `--road-local-files-only` - загружать road segmentation модель только из локального cache, `true` или `false`.

Остановить обработку можно клавишей `q`, если включён показ окна.

## Что можно добавить дальше

Сейчас проект выполняет только детекцию транспорта на тестовом видео. Следующие шаги для системы обнаружения машин на встречной полосе:

- Более точная настройка ROI для разных камер.
- Более стабильная drivable area segmentation на сложных дорогах и бездорожье.
- Object tracking.
- Классификацию `front/rear/side` для машин.
- Определение приближающихся машин.

Модуль `src/road_area_logic.py` сейчас строит маску класса `road` и центральную кривую дороги. Если внутри маски дороги видна жёлтая разделительная разметка, она используется как основной разделитель направлений движения. Если жёлтую разметку найти не удалось, используется fallback по геометрии дорожной маски. Позже этот модуль можно заменить более специализированной drivable area моделью без переписывания детектора и обработки видео.

Центральная кривая строится не по всем road-пикселям кадра, а по основному дорожному коридору. Нижняя зона своей машины игнорируется, а сама линия сглаживается, чтобы не реагировать на шумы сегментации, капот и небольшие разрывы маски.
