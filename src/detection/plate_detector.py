"""
번호판 영역 검출 모듈
YOLOv7 탐지 결과에서 번호판 ROI 추출
YOLOv7 미사용 시 윤곽선 기반 폴백 탐지
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from .yolo_detector import Detection


class PlateDetector:
    """번호판 영역 추출 및 검증"""

    # 한국 번호판 가로세로 비율 범위
    ASPECT_RATIO_MIN = 2.0
    ASPECT_RATIO_MAX = 5.5
    MIN_AREA = 1500

    def __init__(self, config: dict):
        self.cfg = config

    def extract_plate_roi(
        self,
        frame: np.ndarray,
        detections: List[Detection],
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        YOLOv7이 탐지한 license_plate 박스에서 ROI 크롭.
        탐지 결과가 없으면 윤곽선 기반 폴백 탐지를 수행.
        반환: (roi_image, (x1,y1,x2,y2)) 리스트
        """
        plates = [d for d in detections if d.class_name == "license_plate"]

        if plates:
            return self._extract_from_detections(frame, plates)
        return self._contour_based_detection(frame)

    def _extract_from_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
    ) -> List[Tuple[np.ndarray, Tuple]]:
        results = []
        h, w = frame.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = frame[y1:y2, x1:x2]
            results.append((roi, (x1, y1, x2, y2)))
        return results

    def _contour_based_detection(
        self, frame: np.ndarray
    ) -> List[Tuple[np.ndarray, Tuple]]:
        """YOLOv7 없이 형태학적 특성으로 번호판 후보 추출"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        sobel = cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=3)

        _, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        results = []
        h, w = frame.shape[:2]
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if ch == 0:
                continue
            area = cw * ch
            ratio = cw / ch
            if (
                area >= self.MIN_AREA
                and self.ASPECT_RATIO_MIN <= ratio <= self.ASPECT_RATIO_MAX
            ):
                roi = frame[y : y + ch, x : x + cw]
                results.append((roi, (x, y, x + cw, y + ch)))

        results.sort(key=lambda r: r[0].size, reverse=True)
        return results[:3]

    @staticmethod
    def draw_plates(
        frame: np.ndarray,
        plate_bboxes: List[Tuple],
        plate_texts: List[str],
    ) -> np.ndarray:
        for (x1, y1, x2, y2), text in zip(plate_bboxes, plate_texts):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                frame,
                text,
                (x1, y1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
        return frame
