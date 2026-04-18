"""
YOLOv7 파인튜닝 스크립트
Roboflow 데이터셋 기준 커스텀 학습
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv7 번호판 탐지 학습")
    p.add_argument("--data", required=True, help="dataset.yaml 경로")
    p.add_argument("--weights", default="models/yolov7/yolov7.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--device", default="0", help="GPU id (0,1,..) 또는 cpu")
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="parking_detector")
    p.add_argument("--hyp", default="data/hyp.scratch.p5.yaml")
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def check_yolov7_repo():
    """YOLOv7 공식 레포 클론 여부 확인"""
    yolov7_path = Path("yolov7_repo")
    if not yolov7_path.exists():
        print("[학습] YOLOv7 공식 레포 클론 중...")
        subprocess.run(
            ["git", "clone", "https://github.com/WongKinYiu/yolov7.git", str(yolov7_path)],
            check=True,
        )
        print("[학습] 클론 완료")
    return yolov7_path


def main():
    args = parse_args()
    yolov7_path = check_yolov7_repo()

    train_script = yolov7_path / "train.py"
    if not train_script.exists():
        print(f"[오류] {train_script} 없음")
        sys.exit(1)

    cmd = [
        sys.executable, str(train_script),
        "--data", args.data,
        "--weights", args.weights,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--img-size", str(args.img_size),
        "--device", args.device,
        "--project", args.project,
        "--name", args.name,
        "--workers", str(args.workers),
        "--cache",
    ]

    if Path(args.hyp).exists():
        cmd += ["--hyp", args.hyp]

    print("[학습] 시작:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\n[학습 완료] 결과: {args.project}/{args.name}/weights/best.pt")
    print("config/config.yaml 의 yolo.weights 경로를 best.pt 로 업데이트하세요.")


if __name__ == "__main__":
    main()
