"""
데모 실행 스크립트
웹캠 / 동영상 / 이미지 파일로 시스템 동작 확인
"""

import sys
import os
import argparse
import cv2
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.parking.parking_manager import ParkingManager


def parse_args():
    p = argparse.ArgumentParser(description="무인 주차장 시스템 데모")
    p.add_argument("--source", default="0",
                   help="입력 소스 (0=웹캠, 파일 경로, RTSP URL)")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--mode", choices=["video", "image", "auto"], default="auto")
    p.add_argument("--output", default="", help="결과 저장 경로 (선택)")
    p.add_argument("--no-display", action="store_true", help="화면 표시 비활성화")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_video(manager: ParkingManager, source, output: str, no_display: bool):
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"[오류] 소스를 열 수 없습니다: {source}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output, fourcc, fps, (w, h))

    print(f"[데모] 시작 (소스={source}, {w}×{h} @ {fps:.0f}fps)")
    print("  q 키를 누르면 종료, s 키를 누르면 현황 출력")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[데모] 스트림 종료")
            break

        frame_count += 1
        # 매 프레임마다 YOLO 추론 (실제 GPU에서 ~30fps)
        annotated, plate = manager.process_frame(frame)

        if writer:
            writer.write(annotated)

        if not no_display:
            cv2.imshow("Smart Parking System", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                manager.print_status()

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def run_image(manager: ParkingManager, path: str, no_display: bool):
    frame = cv2.imread(path)
    if frame is None:
        print(f"[오류] 이미지를 읽을 수 없습니다: {path}")
        return

    annotated, plate = manager.process_frame(frame)
    print(f"[결과] 인식된 번호판: {plate}")
    manager.print_status()

    if not no_display:
        cv2.imshow("결과", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    out_path = path.replace(".", "_result.")
    cv2.imwrite(out_path, annotated)
    print(f"[저장] {out_path}")


def main():
    args = parse_args()
    config = load_config(args.config)

    # 샘플 등록 차량 추가 (데모용)
    manager = ParkingManager(config)
    manager.register_vehicle("12가3456", "홍길동", "010-1234-5678", assigned_slot=1)
    manager.register_vehicle("34나7890", "김철수", "010-9876-5432", assigned_slot=5)
    manager.register_vehicle("56다1234", "이영희", "010-5555-6666", assigned_slot=10)

    source = args.source
    is_image = (
        args.mode == "image"
        or (args.mode == "auto" and source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
    )

    if is_image:
        run_image(manager, source, args.no_display)
    else:
        run_video(manager, source, args.output, args.no_display)


if __name__ == "__main__":
    main()
