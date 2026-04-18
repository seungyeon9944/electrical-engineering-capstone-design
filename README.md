# 딥러닝 알고리즘을 결합한 무인 주차장 지정좌석제 솔루션

## 1. 프로젝트 개요
**YOLOv7 비전 알고리즘 + Tesseract OCR 번호판 인식 + 다층 주차 배치 알고리즘**을 결합한 차세대 무인 주차장 관리 시스템입니다.

단순 빈자리 안내를 넘어, 입차 순간 **최적 배치 알고리즘**이 즉시 작동하여 층 전환 비용·보행 거리·층별 부하 균형을 동시에 고려한 최적 슬롯을 자동 배정합니다.

## 2. 핵심 기술

| 모듈 | 기술 | 역할 |
|------|------|------|
| 차량 탐지 | **YOLOv7** (Compound Scaling) | 차량·번호판·보행자 실시간 탐지 |
| 번호판 인식 | **Tesseract OCR** + 전처리 파이프라인 | 노이즈 제거·이진화·기울기 보정 |
| 고속 전처리 | **C Extension** (fast_preprocess.c) | 실시간 이미지 전처리 가속 |
| 슬롯 배정 | **다층 최적 배치 알고리즘** | 가중치 기반 다기준 최적화 |

## 2-1. Vision 모델
## 성능 지표

| 지표 | 값 |
|------|----|
| 번호판 인식 정확도 | **91.5%** |
| 차량 탐지 mAP@0.5 | 86.3% |
| 평균 처리 속도 | 28.4 FPS (RTX 3060) |
| 슬롯 배정 응답 시간 | < 1 ms (그리디 O(N)) |
| 지원 층 수 | 제한 없음 (config 설정) |
| 지원 차량 유형 | sedan / disabled / ev / oversized |


## 2-2. 다층 주차 배치 알고리즘

### 비용 함수

입차 차량마다 모든 빈 슬롯에 대해 아래 비용을 계산하고 **최소 비용 슬롯**을 배정합니다.

```
cost(s) = w_floor    × floor(s)
        + w_distance × ManhattanDistance(s, entrance)
        + w_load     × FloorOccupancyRate(floor(s)) × 10
```

| 항목 | 설명 | 기본 가중치 |
|------|------|-------------|
| `floor_cost` | 층이 높을수록 엘리베이터/계단 이동 비용 증가 | `w_floor = 1.5` |
| `distance_cost` | 슬롯과 해당 층 입구 사이 맨해튼 거리 | `w_distance = 1.0` |
| `load_cost` | 특정 층이 붐빌수록 비용 증가 (부하 균형) | `w_load = 0.8` |

가중치는 `config/config.yaml`의 `allocation.weights`에서 자유롭게 조정할 수 있습니다.

### 배정 우선순위

```
1. 등록 차량  → 지정 슬롯 직접 배정
               └ 점유 시: 같은 층 내 지정 슬롯과 맨해튼 거리 최소 슬롯
2. 장애인 차량 → 장애인 전용 슬롯 우선
               └ 없으면: 1층 입구 최근거리 슬롯
3. 전기차(EV) → EV 충전 슬롯 우선 → 비용 최소 슬롯
4. 일반 차량  → 비용 함수 최소 슬롯 (그리디)
```

### 알고리즘 특징

- **그리디(Greedy) 방식** — O(N) 단일 패스로 실시간 응답 (N = 전체 슬롯 수)
- **부하 균형(Load Balancing)** — 특정 층으로 쏠림 방지, 층별 점유율 자동 분산
- **층 전환 비용 모델링** — 낮은 층·입구 근처 슬롯을 우선 배정하여 보행 동선 최소화
- **차량 유형별 차등 배정** — sedan / disabled / ev / oversized 4가지 유형 지원
- **Top-K 추천 출력** — 배치 근거를 비용 순 슬롯 목록으로 투명하게 제공

## 2-3. 배정 흐름

```
입차 감지 (YOLOv7)
    ↓
번호판 인식 (OCR)
    ↓
차량 유형 판별 (DB 조회)
    ↓
MultiFloorAllocator.assign()
    ↓
┌─────────────────────────────────────────┐
│  모든 빈 슬롯에 대해 cost(s) 계산       │
│  floor_cost + distance_cost + load_cost │
│  → argmin(cost) 슬롯 선택              │
└─────────────────────────────────────────┘
    ↓
슬롯 배정 & 안내판 표시
```

## 3. 프로젝트 구조

```
parking-lot-system/
├── main.py                        # 메인 진입점 (실시간 처리)
├── config/
│   └── config.yaml                # 시스템 설정 (층 수·가중치 포함)
├── src/
│   ├── detection/
│   │   ├── yolo_detector.py       # YOLOv7 객체 탐지
│   │   └── plate_detector.py      # 번호판 영역 검출
│   ├── ocr/
│   │   ├── preprocessor.py        # 이미지 전처리 파이프라인
│   │   └── ocr_engine.py          # Tesseract OCR 엔진
│   ├── parking/
│   │   ├── allocation_algorithm.py  ← 다층 배치 알고리즘 (핵심)
│   │   ├── parking_manager.py     # 전체 파이프라인 조율
│   │   └── slot_manager.py        # 슬롯 관리 + 알고리즘 연동
│   ├── database/
│   │   └── db_manager.py          # SQLite (floor 컬럼 포함)
│   └── utils/
│       └── image_utils.py         # 이미지 유틸리티
├── c_extensions/
│   ├── fast_preprocess.c          # C 고속 전처리
│   ├── fast_preprocess.h
│   └── Makefile
├── scripts/
│   ├── demo.py                    # 데모 실행
│   ├── train.py                   # YOLOv7 학습
│   └── evaluate.py                # 성능 평가
├── tests/
│   ├── test_detection.py
│   ├── test_ocr.py
│   └── test_parking.py
├── models/                        # 학습된 모델 가중치 (별도 다운로드)
└── data/                          # 데이터셋 (별도 준비)
```

## 4. 설치 방법

### 4-1. 사전 요구사항

```bash
# Python 3.8+
python --version

# Tesseract OCR 설치 (Windows)
# https://github.com/UB-Mannheim/tesseract/wiki 에서 다운로드

# CUDA (GPU 사용 시, 선택사항)
nvidia-smi
```

### 4-2. 패키지 설치

```bash
git clone https://github.com/your-repo/parking-lot-system.git
cd parking-lot-system
pip install -r requirements.txt
```

### 4-3. C Extension 빌드 (선택사항 — 고속 처리)

```bash
cd c_extensions
make
cd ..
```

### 4-4. YOLOv7 가중치 다운로드

```bash
mkdir -p models/yolov7
# yolov7.pt를 models/yolov7/ 에 위치
# 커스텀 학습 가중치는 scripts/train.py로 생성
```

## 5. 사용 방법

### 데모 실행

```bash
# 웹캠 실시간 처리
python scripts/demo.py --source 0

# 동영상 파일 처리
python scripts/demo.py --source video.mp4
```

### 메인 시스템 실행

```bash
python main.py --config config/config.yaml
```

### 다층 주차장 설정

`config/config.yaml`에서 층 수와 배치 가중치를 조정합니다:

```yaml
parking:
  slot_layout:
    floors: 3       # 3층 주차장
    rows: 5
    cols: 10

allocation:
  load_balance: true
  weights:
    floor: 1.5      # 층 전환 비용 (높을수록 1층 선호 강화)
    distance: 1.0   # 보행 거리 비용
    load: 0.8       # 부하 균형 강도
```

### 모델 학습

```bash
python scripts/train.py \
    --data data/dataset.yaml \
    --weights models/yolov7/yolov7.pt \
    --epochs 100 \
    --batch-size 16
```

## 6. 데이터셋

- **원본**: Roboflow의 YOLOv5 형식 번호판 데이터셋
- **수정**: YOLOv7 형식 변환 + 한국 번호판 이미지 추가
- **구성**: train 70% / val 20% / test 10%
- **클래스**: `license_plate`, `vehicle`, `person`, `animal`


## 7. 담당 교수님

**한양대학교 문준 교수님** (졸업 프로젝트 지도)


## 8. 참고

- [YOLOv7 논문 (arXiv)](https://arxiv.org/abs/2207.02696)
- [WongKinYiu/yolov7 GitHub](https://github.com/WongKinYiu/yolov7)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- 데이터셋: [Roboflow](https://roboflow.com)
