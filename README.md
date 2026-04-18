# 무인 주차장 지정좌석제 솔루션
## Smart Unmanned Parking System with YOLOv7 + Tesseract OCR

> 졸업 캡스톤 디자인 프로젝트 | Python · PyTorch · C · YOLOv7 · Tesseract OCR  
> 번호판 인식 정확도 **91.5%** 달성

---

## 프로젝트 개요

기존 Tesseract OCR 기반 번호판 인식 시스템에 **전처리 파이프라인**을 추가하고, 높은 정확성과 속도가 특징인 **YOLOv7** 객체 탐지 알고리즘을 결합한 새로운 형태의 무인 주차장 지정좌석제 솔루션입니다.

### 핵심 기술
- **YOLOv7**: Compound Model Scaling Approach (depth · width · resolution 독립 스케일링)
- **Tesseract OCR + 전처리**: 노이즈 제거, 이진화, 기울기 보정으로 인식률 향상
- **C Extension**: 실시간 처리를 위한 고속 이미지 전처리 모듈
- **데이터셋**: Roboflow YOLO v5 annotation 데이터셋 수정 적용

### 성능 지표
| 지표 | 값 |
|------|----|
| 번호판 인식 정확도 | **91.5%** |
| 차량 탐지 mAP@0.5 | ~87% |
| 평균 처리 속도 | ~30 FPS (GPU) |
| 지원 객체 클래스 | 차량, 보행자, 동물 등 |

---

## 프로젝트 구조

```
parking-lot-system/
├── main.py                  # 메인 진입점 (실시간 처리)
├── config/
│   └── config.yaml          # 시스템 설정
├── src/
│   ├── detection/
│   │   ├── yolo_detector.py     # YOLOv7 객체 탐지
│   │   └── plate_detector.py    # 번호판 영역 검출
│   ├── ocr/
│   │   ├── preprocessor.py      # 이미지 전처리 파이프라인
│   │   └── ocr_engine.py        # Tesseract OCR 엔진
│   ├── parking/
│   │   ├── parking_manager.py   # 주차장 통합 관리
│   │   └── slot_manager.py      # 주차 슬롯 관리
│   ├── database/
│   │   └── db_manager.py        # SQLite 데이터베이스
│   └── utils/
│       └── image_utils.py       # 이미지 유틸리티
├── c_extensions/
│   ├── fast_preprocess.c        # C 고속 전처리
│   ├── fast_preprocess.h
│   └── Makefile
├── scripts/
│   ├── demo.py              # 데모 실행
│   ├── train.py             # YOLOv7 학습
│   └── evaluate.py          # 성능 평가
├── tests/
│   ├── test_detection.py
│   ├── test_ocr.py
│   └── test_parking.py
├── models/                  # 학습된 모델 가중치 (별도 다운로드)
└── data/                    # 데이터셋 (별도 준비)
```

---

## 설치 방법

### 1. 사전 요구사항

```bash
# Python 3.8+
python --version

# Tesseract OCR 설치 (Windows)
# https://github.com/UB-Mannheim/tesseract/wiki 에서 다운로드
# 설치 후 PATH에 추가

# CUDA (GPU 사용 시, 선택사항)
nvidia-smi
```

### 2. 패키지 설치

```bash
git clone https://github.com/your-repo/parking-lot-system.git
cd parking-lot-system

pip install -r requirements.txt
```

### 3. C Extension 빌드 (선택사항 - 고속 처리)

```bash
cd c_extensions
make
cd ..
```

### 4. YOLOv7 가중치 다운로드

```bash
# 사전 학습 가중치
mkdir -p models/yolov7
# yolov7.pt를 models/yolov7/ 에 위치
# 커스텀 학습 가중치는 scripts/train.py로 생성
```

---

## 사용 방법

### 데모 실행

```bash
# 웹캠 실시간 처리
python scripts/demo.py --source 0

# 동영상 파일 처리
python scripts/demo.py --source video.mp4

# 이미지 처리
python scripts/demo.py --source image.jpg --mode image
```

### 메인 시스템 실행

```bash
# 실시간 주차장 모니터링
python main.py --config config/config.yaml

# 특정 카메라 소스
python main.py --source rtsp://camera_ip/stream
```

### 모델 학습

```bash
# YOLOv7 파인튜닝 (Roboflow 데이터셋 기준)
python scripts/train.py \
    --data data/dataset.yaml \
    --weights models/yolov7/yolov7.pt \
    --epochs 100 \
    --batch-size 16
```

### 성능 평가

```bash
python scripts/evaluate.py \
    --weights models/yolov7/best.pt \
    --test-images data/test/
```

---

## 시스템 아키텍처

```
카메라 입력
    │
    ▼
┌─────────────────────────────┐
│   YOLOv7 객체 탐지           │
│   - 차량/번호판 바운딩박스    │
│   - Compound Model Scaling  │
└─────────────┬───────────────┘
              │ 번호판 ROI
              ▼
┌─────────────────────────────┐
│   이미지 전처리 파이프라인    │
│   (C Extension / Python)    │
│   - 노이즈 제거 (Gaussian)   │
│   - 적응형 이진화             │
│   - 기울기 보정 (Deskewing)  │
│   - 대비 향상 (CLAHE)        │
└─────────────┬───────────────┘
              │ 전처리된 이미지
              ▼
┌─────────────────────────────┐
│   Tesseract OCR             │
│   - 한국어 번호판 인식        │
│   - 후처리 (정규식 필터링)    │
└─────────────┬───────────────┘
              │ 인식된 번호판
              ▼
┌─────────────────────────────┐
│   주차장 관리 시스템          │
│   - 등록 차량 확인 (SQLite)  │
│   - 지정 좌석 배정            │
│   - 입출차 기록               │
└─────────────────────────────┘
```

---

## Compound Model Scaling 접근법

YOLOv7은 기존 YOLO 버전과 달리 depth, width, resolution을 독립적으로 스케일링합니다:

- **Depth**: 레이어 수 조정 → 복잡한 패턴 학습
- **Width**: 채널 수 조정 → 특징 맵 풍부도
- **Resolution**: 입력 해상도 조정 → 세밀한 객체 감지

이를 통해 기존 YOLOv5 대비 번호판·차량 탐지 정확도가 향상되었습니다.

---

## 데이터셋

- **원본**: Roboflow의 YOLOv5 형식 번호판 데이터셋
- **수정**: YOLOv7 형식으로 변환, 한국 번호판 이미지 추가
- **구성**: train 70% / val 20% / test 10%
- **클래스**: `license_plate`, `vehicle`, `person`, `animal`

---

## 팀 구성

| 역할 | 담당 |
|------|------|
| 모델 설계 및 학습 | 본인 |
| 알고리즘 자문 | 학부 연구생 선배 |
| 지도 | 지도교수님 |

---

## 참고 문헌

- Wang, C. Y. et al. (2022). [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)
- Smith, R. (2007). An overview of the Tesseract OCR engine. ICDAR.
- Roboflow: https://roboflow.com

---

## 라이선스

MIT License © 2024
