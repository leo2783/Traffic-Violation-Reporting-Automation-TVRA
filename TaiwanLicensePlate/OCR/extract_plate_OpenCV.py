import cv2
from ultralytics import YOLO
from pathlib import Path


class LicensePlateExtractor:
    def __init__(
        self,
        model_path,
        output_dir="plates",
        conf_threshold=0.75,
        frame_interval=5,
        iou_threshold=0.45,
        device="cuda:0"
    )-> None:
        self._model_path = model_path
        self._output_dir = Path(output_dir)

        self._conf_threshold = conf_threshold
        self._frame_interval = frame_interval
        self._iou_threshold = iou_threshold
        self._device = device

        self._output_dir.mkdir(exist_ok=True)

        self._model = YOLO(self._model_path)
        

    def extract(self,video_path)-> None:
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        _frame_count = 0
        _save_count = 0

        try:
            while True:
                _ret, _frame = self._cap.read()

                if not _ret:
                    break
                if _frame_count % self._frame_interval != 0:
                    _frame_count += 1
                    continue

                _h, _w = _frame.shape[:2]

                _min_plate_w = int(_w * 0.02)
                _min_plate_h = int(_h * 0.01)
                _result = self._model.predict(_frame,verbose=False,conf=self._conf_threshold,iou=self._iou_threshold,classes=[0],device=self._device,stream=False)[0]

                for _box in _result.boxes:
                    _x1, _y1, _x2, _y2 = map(int, _box.xyxy[0])
                    conf = float(_box.conf[0])

                    _x1 = max(0, _x1)
                    _y1 = max(0, _y1)
                    _x2 = min(_w, _x2)
                    _y2 = min(_h, _y2)

                    if _x2 <= _x1 or _y2 <= _y1:
                        continue

                    _box_w = _x2 - _x1
                    _box_h = _y2 - _y1

                    if _box_w < _min_plate_w or _box_h < _min_plate_h:
                        continue

                    _plate_img = _frame[_y1:_y2, _x1:_x2]
                    _save_path = self._output_dir / f"frame_{_frame_count}_conf_{conf:.2f}_{_save_count}.jpg"
                    _success = cv2.imwrite(str(_save_path), _plate_img)

                    if _success:
                        _save_count += 1

                _frame_count += 1

        finally:
            self._cap.release()

        print(f"Extracted {_save_count} license plate images to {self._output_dir}")