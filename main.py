"""
무인 주차장 지정좌석제 시스템 메인 진입점
YOLOv7 + Tesseract OCR 번호판 인식
"""

import argparse
import logging
import sys
import yaml
import cv2
from pathlib import Path

from src.parking.parking_manager import ParkingManager


def setup_logging(config: dict):
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO"))
    log_file = log_cfg.get("file", "logs/parking_system.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    handlers = [logging.StreamHandler(sys.stdout)]
    try:
        from logging.handlers import RotatingFileHandler
        handlers.append(
            RotatingFileHandler(
                log_file,
                maxBytes=log_cfg.get("max_size_mb", 50) * 1024 * 1024,
                backupCount=log_cfg.get("backup_count", 5),
                encoding="utf-8",
            )
        )
    except Exception:
        pass

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="무인 주차장 지정좌석제 시스템 (YOLOv7 + Tesseract OCR)"
    )
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--source", default=None,
                   help="카메라 소스 (config 설정 덮어쓰기)")
    p.add_argument("--register", nargs=3,
                   metavar=("번호판", "이름", "전화번호"),
                   help="차량 등록 후 종료")
    p.add_argument("--status", action="store_true",
                   help="현황 출력 후 종료")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    setup_logging(config)
    logger = logging.getLogger("main")

    manager = ParkingManager(config)

    # 차량 등록 모드
    if args.register:
        plate, name, phone = args.register
        ok = manager.register_vehicle(plate, name, phone)
        msg = f"등록 완료: {plate} ({name})" if ok else f"이미 등록된 번호판: {plate}"
        print(msg)
        return

    # 현황 출력 모드
    if args.status:
        manager.print_status()
        return

    # 실시간 처리 모드
    source_cfg = args.source or config.get("camera", {}).get("source", 0)
    src = int(source_cfg) if str(source_cfg).isdigit() else source_cfg

    logger.info(f"시스템 시작 (소스={src})")
    print("\n" + "=" * 50)
    print("  무인 주차장 지정좌석제 시스템")
    print("  YOLOv7 + Tesseract OCR")
    print("  q: 종료  |  s: 현황  |  r: 슬롯 맵")
    print("=" * 50 + "\n")

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logger.error(f"카메라 소스를 열 수 없습니다: {src}")
        sys.exit(1)

    cam_cfg = config.get("camera", {})
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 720))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("프레임 읽기 실패")
                break

            annotated, plate = manager.process_frame(frame)

            cv2.imshow("Smart Parking System - YOLOv7 + OCR", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                manager.print_status()
            elif key == ord("r"):
                manager.slot_manager.print_layout()

    except KeyboardInterrupt:
        logger.info("사용자 종료 요청")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("시스템 종료")


if __name__ == "__main__":
    main()
