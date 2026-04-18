"""YOLOv7 탐지 모듈 단위 테스트 (데모 모드)"""

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detection.yolo_detector import YOLOv7Detector, Detection
from src.detection.plate_detector import PlateDetector

CONFIG = {
    "yolo": {
        "weights": "models/yolov7/nonexistent.pt",  # 데모 모드 강제
        "img_size": 640,
        "conf_threshold": 0.45,
        "iou_threshold": 0.45,
        "device": "cpu",
        "half": False,
    },
    "preprocessing": {},
}


@pytest.fixture
def blank_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def plate_detector():
    return PlateDetector(CONFIG)


def test_detector_loads_in_demo_mode():
    det = YOLOv7Detector(CONFIG)
    assert det.demo_mode is True


def test_detect_returns_list(blank_frame):
    det = YOLOv7Detector(CONFIG)
    results = det.detect(blank_frame)
    assert isinstance(results, list)


def test_filter_by_class():
    det = YOLOv7Detector(CONFIG)
    dets = [
        Detection((0, 0, 10, 10), 0.9, 0, "license_plate"),
        Detection((10, 10, 50, 50), 0.8, 1, "vehicle"),
    ]
    filtered = det.filter_by_class(dets, ["license_plate"])
    assert len(filtered) == 1
    assert filtered[0].class_name == "license_plate"


def test_draw_does_not_crash(blank_frame):
    det = YOLOv7Detector(CONFIG)
    dets = [Detection((10, 10, 100, 80), 0.7, 1, "vehicle")]
    result = det.draw(blank_frame.copy(), dets)
    assert result.shape == blank_frame.shape


def test_plate_detector_contour(plate_detector, blank_frame):
    results = plate_detector._contour_based_detection(blank_frame)
    assert isinstance(results, list)
