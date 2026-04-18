"""
번호판 이미지 전처리 파이프라인
Tesseract OCR 인식률 향상을 위한 다단계 처리
"""

import cv2
import numpy as np
from typing import Optional


class PlatePreprocessor:
    """
    번호판 ROI 전처리 파이프라인
    1. 크기 정규화
    2. 그레이스케일 변환
    3. CLAHE 대비 향상
    4. 가우시안 노이즈 제거
    5. 기울기 보정 (Deskewing)
    6. 적응형 이진화
    7. 형태학적 정제
    """

    def __init__(self, config: dict):
        cfg = config.get("preprocessing", {})
        self.blur_kernel = cfg.get("gaussian_blur_kernel", 5)
        self.clahe_clip = cfg.get("clahe_clip_limit", 3.0)
        self.clahe_grid = tuple(cfg.get("clahe_tile_grid", [8, 8]))
        self.deskew_thresh = cfg.get("deskew_threshold", 2.0)
        self.morph_k = cfg.get("morph_kernel_size", 3)
        self.use_c = cfg.get("use_c_extension", False)

        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip, tileGridSize=self.clahe_grid
        )

        if self.use_c:
            self._try_load_c_extension()

    def _try_load_c_extension(self):
        try:
            import ctypes, os
            lib_path = os.path.join(
                os.path.dirname(__file__), "../../c_extensions/fast_preprocess.so"
            )
            self._c_lib = ctypes.CDLL(lib_path)
            print("[Preprocessor] C Extension 로드 완료")
        except Exception as e:
            print(f"[Preprocessor] C Extension 로드 실패, Python 처리 사용: {e}")
            self.use_c = False

    def process(self, roi: np.ndarray) -> np.ndarray:
        """전체 전처리 파이프라인 실행"""
        if roi is None or roi.size == 0:
            return roi

        img = self._normalize_size(roi)
        img = self._to_gray(img)
        img = self._enhance_contrast(img)
        img = self._denoise(img)
        img = self._deskew(img)
        img = self._binarize(img)
        img = self._morphological_clean(img)
        return img

    def _normalize_size(self, img: np.ndarray) -> np.ndarray:
        """번호판을 표준 크기(400×120)로 리사이즈"""
        target_w, target_h = 400, 120
        h, w = img.shape[:2]
        if w < 60 or h < 15:
            scale = max(target_w / w, target_h / h) * 1.5
        else:
            scale = max(target_w / w, target_h / h)
        new_w = max(int(w * scale), target_w)
        new_h = max(int(h * scale), target_h)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """CLAHE 적응형 히스토그램 평활화"""
        return self.clahe.apply(gray)

    def _denoise(self, gray: np.ndarray) -> np.ndarray:
        k = self.blur_kernel if self.blur_kernel % 2 == 1 else self.blur_kernel + 1
        return cv2.GaussianBlur(gray, (k, k), 0)

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        """텍스트 줄 기준 기울기 보정"""
        coords = np.column_stack(np.where(gray < 128))
        if len(coords) < 50:
            return gray
        angle = cv2.minAreaRect(coords)[-1]
        # minAreaRect 각도 보정
        if angle < -45:
            angle += 90
        if abs(angle) < self.deskew_thresh:
            return gray
        h, w = gray.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        """Otsu 이진화 + 적응형 이진화 결합"""
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=8,
        )
        # 두 이진화 결과의 OR → 더 많은 텍스트 픽셀 보존
        return cv2.bitwise_or(otsu, adaptive)

    def _morphological_clean(self, binary: np.ndarray) -> np.ndarray:
        """잡음 픽셀 제거 및 문자 연결"""
        k = self.morph_k
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (k + 2, k))
        img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_close)
        return img

    def visualize_pipeline(self, roi: np.ndarray) -> np.ndarray:
        """전처리 단계별 결과를 가로로 이어 붙인 시각화 이미지 반환"""
        stages = []
        img = self._normalize_size(roi)
        stages.append(("원본", img.copy() if len(img.shape) == 2
                        else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        img = self._to_gray(img)
        img = self._enhance_contrast(img); stages.append(("CLAHE", img.copy()))
        img = self._denoise(img);          stages.append(("노이즈제거", img.copy()))
        img = self._deskew(img);           stages.append(("기울기보정", img.copy()))
        img = self._binarize(img);         stages.append(("이진화", img.copy()))
        img = self._morphological_clean(img); stages.append(("형태학처리", img.copy()))

        target_h = 80
        strips = []
        for title, s in stages:
            h, w = s.shape[:2]
            scale = target_h / h
            resized = cv2.resize(s, (int(w * scale), target_h))
            color = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            cv2.putText(color, title, (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
            strips.append(color)

        return np.hstack(strips)
