"""
주차장 통합 관리 모듈
YOLOv7 탐지 → OCR → 슬롯 배정 전체 파이프라인 조율
"""

import cv2
import time
import logging
import numpy as np
from typing import Optional, Dict, Tuple
from datetime import datetime

from ..detection.yolo_detector import YOLOv7Detector
from ..detection.plate_detector import PlateDetector
from ..ocr.ocr_engine import OCREngine
from ..parking.slot_manager import SlotManager
from ..database.db_manager import DatabaseManager
from ..utils.image_utils import draw_status_overlay


logger = logging.getLogger(__name__)


class ParkingManager:
    """메인 주차장 관리 클래스"""

    COOLDOWN_SEC = 5.0  # 같은 번호판 재처리 방지 쿨다운

    def __init__(self, config: dict):
        self.config = config
        self.db = DatabaseManager(config.get("database", {}).get("path", "data/parking.db"))
        self.yolo = YOLOv7Detector(config)
        self.plate_detector = PlateDetector(config)
        self.ocr = OCREngine(config)
        self.slot_manager = SlotManager(config, self.db)

        self._recent: Dict[str, float] = {}  # plate -> last_processed_time
        self._current_plates: Dict[str, Dict] = {}  # plate -> slot info

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """
        단일 프레임 처리.
        반환: (어노테이션된 프레임, 인식된 번호판 문자열 또는 None)
        """
        detections = self.yolo.detect(frame)
        annotated = self.yolo.draw(frame.copy(), detections)

        plate_rois = self.plate_detector.extract_plate_roi(frame, detections)

        recognized_plate = None
        plate_bboxes, plate_texts = [], []

        for roi, bbox in plate_rois:
            plate_text, conf = self.ocr.read_plate(roi)
            if not plate_text:
                continue

            plate_bboxes.append(bbox)
            plate_texts.append(f"{plate_text} ({conf:.0f}%)")
            recognized_plate = plate_text

            if self._is_cooldown(plate_text):
                continue

            self._handle_vehicle(plate_text)
            self._recent[plate_text] = time.time()

        annotated = self.plate_detector.draw_plates(annotated, plate_bboxes, plate_texts)
        annotated = draw_status_overlay(annotated, self.db.get_stats(), self._current_plates)

        return annotated, recognized_plate

    def _is_cooldown(self, plate: str) -> bool:
        last = self._recent.get(plate, 0)
        return (time.time() - last) < self.COOLDOWN_SEC

    def _handle_vehicle(self, plate: str):
        """입차 or 출차 판단 및 처리"""
        if plate in self._current_plates:
            self._handle_exit(plate)
        else:
            self._handle_entry(plate)

    def _handle_entry(self, plate: str):
        is_registered = self.db.is_registered(plate)
        slot = self.slot_manager.assign_slot(plate)
        slot_id = slot["id"] if slot else None
        slot_name = slot["slot_name"] if slot else "없음"

        self.db.record_entry(plate, slot_id)
        self._current_plates[plate] = {
            "slot_id": slot_id,
            "slot_name": slot_name,
            "entry_time": datetime.now().strftime("%H:%M:%S"),
            "is_registered": is_registered,
        }

        status = "등록차량" if is_registered else "미등록차량"
        logger.info(f"[입차] {plate} | {status} | 슬롯: {slot_name}")
        print(f"\n>>> 입차: {plate} ({status}) → 슬롯 {slot_name}")

    def _handle_exit(self, plate: str):
        info = self._current_plates.pop(plate)
        fee = self._calculate_fee(info)
        if info["slot_id"]:
            self.slot_manager.release_slot(info["slot_id"])
        self.db.record_exit(plate, fee)
        logger.info(f"[출차] {plate} | 요금: {fee:,}원")
        print(f"\n<<< 출차: {plate} | 슬롯 {info['slot_name']} | 요금: {fee:,}원")

    def _calculate_fee(self, info: Dict) -> float:
        if info.get("is_registered"):
            return 0.0
        entry_str = info.get("entry_time", "")
        try:
            entry_t = datetime.strptime(entry_str, "%H:%M:%S")
            now = datetime.now().replace(
                year=entry_t.year if hasattr(entry_t, 'year') else datetime.now().year
            )
            minutes = max(0, (datetime.now() - entry_t).seconds // 60)
        except Exception:
            minutes = 0
        # 기본 요금: 30분 1000원, 이후 10분당 500원
        if minutes <= 30:
            return 1000.0
        return 1000.0 + ((minutes - 30) // 10 + 1) * 500.0

    # ── 편의 메서드 ────────────────────────────────────────────

    def register_vehicle(self, plate: str, name: str, phone: str = "",
                         slot: Optional[int] = None) -> bool:
        return self.db.register_vehicle(plate, name, phone, slot)

    def print_status(self):
        stats = self.db.get_stats()
        print(f"\n{'='*40}")
        print(f"  주차장 현황 ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
        print(f"{'='*40}")
        print(f"  전체 슬롯 : {stats['total_slots']}")
        print(f"  사용 중   : {stats['occupied']}")
        print(f"  여유      : {stats['available']}")
        print(f"  점유율    : {stats['occupancy_rate']}%")
        print(f"  오늘 입차 : {stats['today_entries']}")
        print(f"{'='*40}")
        self.slot_manager.print_layout()
