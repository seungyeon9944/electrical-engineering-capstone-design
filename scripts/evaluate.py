"""
OCR 인식 성능 평가 스크립트
정확도, 처리 속도, 혼동 행렬 출력
"""

import sys
import os
import argparse
import time
import csv
import cv2
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ocr.preprocessor import PlatePreprocessor
from src.ocr.ocr_engine import OCREngine


def parse_args():
    p = argparse.ArgumentParser(description="번호판 OCR 성능 평가")
    p.add_argument("--test-images", required=True,
                   help="테스트 이미지 디렉토리 (파일명=번호판 or CSV)")
    p.add_argument("--labels", default="",
                   help="CSV 라벨 파일 (image_path,plate_num)")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--output", default="results/eval_result.csv")
    p.add_argument("--visualize", action="store_true", help="전처리 파이프라인 시각화 저장")
    return p.parse_args()


def load_labels(test_dir: str, label_csv: str):
    """(image_path, ground_truth) 리스트 반환"""
    pairs = []

    if label_csv and Path(label_csv).exists():
        with open(label_csv, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    pairs.append((row[0].strip(), row[1].strip()))
        return pairs

    # 파일명에서 번호판 추출 (예: 12가3456.jpg)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for p in sorted(Path(test_dir).iterdir()):
        if p.suffix.lower() in exts:
            label = p.stem.replace("_", "").replace("-", "")
            pairs.append((str(p), label))
    return pairs


def evaluate(engine: OCREngine, preprocessor: PlatePreprocessor,
             pairs, output_path: str, visualize: bool):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    total = len(pairs)
    correct = 0
    latencies = []
    results = []

    for img_path, gt in pairs:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[경고] 읽기 실패: {img_path}")
            continue

        t0 = time.perf_counter()
        pred, conf = engine.read_plate(img)
        latency = (time.perf_counter() - t0) * 1000

        is_correct = (pred == gt)
        if is_correct:
            correct += 1
        latencies.append(latency)

        results.append({
            "image": img_path,
            "ground_truth": gt,
            "predicted": pred or "",
            "confidence": f"{conf:.1f}",
            "correct": is_correct,
            "latency_ms": f"{latency:.1f}",
        })

        mark = "O" if is_correct else "X"
        print(f"[{mark}] {Path(img_path).name:30s} GT={gt:10s} PRED={pred or '—':10s} {conf:.0f}% {latency:.0f}ms")

        if visualize:
            vis = preprocessor.visualize_pipeline(img)
            vis_path = Path(output_path).parent / f"vis_{Path(img_path).stem}.jpg"
            cv2.imwrite(str(vis_path), vis)

    accuracy = correct / total * 100 if total else 0
    avg_latency = np.mean(latencies) if latencies else 0

    print(f"\n{'='*55}")
    print(f"  인식 정확도  : {accuracy:.1f}% ({correct}/{total})")
    print(f"  평균 처리 시간: {avg_latency:.1f} ms/장")
    print(f"{'='*55}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys() if results else [])
        writer.writeheader()
        writer.writerows(results)
    print(f"[저장] 평가 결과: {output_path}")


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    engine = OCREngine(config)
    preprocessor = PlatePreprocessor(config)
    pairs = load_labels(args.test_images, args.labels)

    if not pairs:
        print("[오류] 테스트 이미지를 찾을 수 없습니다.")
        return

    print(f"[평가] {len(pairs)}개 이미지 처리 중...")
    evaluate(engine, preprocessor, pairs, args.output, args.visualize)


if __name__ == "__main__":
    main()
