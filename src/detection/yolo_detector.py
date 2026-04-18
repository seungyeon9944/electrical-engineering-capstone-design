"""
YOLOv7 기반 객체 탐지 모듈
Compound Model Scaling (depth, width, resolution) 적용
"""

import cv2
import numpy as np
import torch
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str


class YOLOv7Detector:
    """
    YOLOv7 탐지기
    가중치 파일이 없을 경우 OpenCV DNN 폴백 모드로 동작
    """

    CLASS_NAMES = ["license_plate", "vehicle", "person", "animal"]

    def __init__(self, config: dict):
        self.cfg = config["yolo"]
        self.conf_thresh = self.cfg["conf_threshold"]
        self.iou_thresh = self.cfg["iou_threshold"]
        self.img_size = self.cfg["img_size"]
        self.device = self._select_device(self.cfg["device"])
        self.model = None
        self._load_model()

    def _select_device(self, device_str: str) -> torch.device:
        if device_str == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self):
        weights_path = Path(self.cfg["weights"])
        if not weights_path.exists():
            print(f"[YOLOv7] 가중치 파일 없음: {weights_path}")
            print("[YOLOv7] 데모 모드로 실행합니다 (OpenCV Haar Cascade 사용)")
            self.model = None
            self.demo_mode = True
            self._load_demo_cascade()
            return

        try:
            self.model = torch.hub.load(
                "WongKinYiu/yolov7",
                "custom",
                path_or_model=str(weights_path),
                source="local",
                force_reload=False,
            )
            self.model.to(self.device)
            self.model.eval()
            if self.cfg.get("half") and self.device.type == "cuda":
                self.model.half()
            self.demo_mode = False
            print(f"[YOLOv7] 모델 로드 완료 (device={self.device})")
        except Exception as e:
            print(f"[YOLOv7] 모델 로드 실패: {e}")
            print("[YOLOv7] 데모 모드로 전환합니다")
            self.model = None
            self.demo_mode = True
            self._load_demo_cascade()

    def _load_demo_cascade(self):
        # 데모용: OpenCV 내장 차량 탐지기 사용
        cascade_path = cv2.data.haarcascades + "haarcascade_car.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            print("[YOLOv7] 데모 cascade 로드 실패 — 더미 탐지 사용")
            self.cascade = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self.demo_mode:
            return self._detect_demo(frame)
        return self._detect_yolo(frame)

    def _detect_yolo(self, frame: np.ndarray) -> List[Detection]:
        img = self._preprocess_frame(frame)
        with torch.no_grad():
            results = self.model(img, size=self.img_size)
        return self._parse_results(results, frame.shape)

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        tensor = torch.from_numpy(img).to(self.device).float() / 255.0
        return tensor.unsqueeze(0)

    def _parse_results(self, results, original_shape) -> List[Detection]:
        detections = []
        h, w = original_shape[:2]

        try:
            pred = results.xyxyn[0].cpu().numpy()
        except Exception:
            return detections

        for det in pred:
            x1, y1, x2, y2, conf, cls_id = det[:6]
            if conf < self.conf_thresh:
                continue
            cls_id = int(cls_id)
            detections.append(
                Detection(
                    bbox=(int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)),
                    confidence=float(conf),
                    class_id=cls_id,
                    class_name=self.CLASS_NAMES[cls_id]
                    if cls_id < len(self.CLASS_NAMES)
                    else "unknown",
                )
            )
        return detections

    def _detect_demo(self, frame: np.ndarray) -> List[Detection]:
        """가중치 없을 때 사용하는 데모용 탐지 (Haar Cascade)"""
        detections = []
        if self.cascade is None or self.cascade.empty():
            return detections

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = self.cascade.detectMultiScale(gray, 1.1, 3, minSize=(60, 60))
        for x, y, w, h in cars:
            detections.append(
                Detection(
                    bbox=(x, y, x + w, y + h),
                    confidence=0.75,
                    class_id=1,
                    class_name="vehicle",
                )
            )
        return detections

    def filter_by_class(
        self, detections: List[Detection], class_names: List[str]
    ) -> List[Detection]:
        return [d for d in detections if d.class_name in class_names]

    def draw(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        colors = {
            "license_plate": (0, 255, 0),
            "vehicle": (255, 165, 0),
            "person": (0, 0, 255),
            "animal": (255, 0, 255),
        }
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = colors.get(det.class_name, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(
                frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
            )
        return frame
