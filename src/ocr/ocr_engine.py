"""
Tesseract OCR 엔진 래퍼
한국 번호판 형식 검증 및 후처리 포함
"""

import re
import cv2
import numpy as np
import pytesseract
from typing import Optional, Tuple
from .preprocessor import PlatePreprocessor


class OCREngine:
    """Tesseract OCR + 한국 번호판 후처리"""

    # 한국 번호판 패턴
    # 구형: 12가1234, 신형: 123가1234
    KR_PLATE_PATTERN = re.compile(
        r"([0-9]{2,3})\s*([가-힣])\s*([0-9]{4})"
    )

    # OCR 오인식 보정 테이블 (숫자/영문 혼동)
    CHAR_CORRECTION = {
        "O": "0", "I": "1", "l": "1", "S": "5",
        "B": "8", "G": "6", "Z": "2", "Q": "0",
    }

    def __init__(self, config: dict):
        cfg = config.get("ocr", {})
        tesseract_cmd = cfg.get("tesseract_cmd", "tesseract")
        if tesseract_cmd != "tesseract":
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        self.lang = cfg.get("lang", "kor+eng")
        self.tess_config = cfg.get("config", "--psm 8 --oem 3")
        self.min_confidence = cfg.get("min_confidence", 60)
        self.preprocessor = PlatePreprocessor(config)

    def read_plate(
        self, roi: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        번호판 ROI에서 텍스트 추출.
        반환: (정규화된 번호판 문자열 또는 None, 신뢰도)
        """
        if roi is None or roi.size == 0:
            return None, 0.0

        processed = self.preprocessor.process(roi)

        # 전처리 이미지 + 원본 이미지 모두 시도해 더 좋은 결과 채택
        result_proc = self._run_tesseract(processed)
        result_orig = self._run_tesseract(
            cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if len(roi.shape) == 3 else roi
        )

        best_text, best_conf = self._pick_best(result_proc, result_orig)
        normalized = self._normalize_plate(best_text)
        return normalized, best_conf

    def _run_tesseract(self, gray: np.ndarray) -> Tuple[str, float]:
        try:
            data = pytesseract.image_to_data(
                gray,
                lang=self.lang,
                config=self.tess_config,
                output_type=pytesseract.Output.DICT,
            )
            texts, confs = [], []
            for txt, conf in zip(data["text"], data["conf"]):
                txt = txt.strip()
                if txt and int(conf) > 0:
                    texts.append(txt)
                    confs.append(int(conf))
            combined = "".join(texts)
            avg_conf = float(np.mean(confs)) if confs else 0.0
            return combined, avg_conf
        except Exception as e:
            return "", 0.0

    def _pick_best(
        self,
        a: Tuple[str, float],
        b: Tuple[str, float],
    ) -> Tuple[str, float]:
        text_a, conf_a = a
        text_b, conf_b = b
        # 유효한 번호판 패턴을 가진 쪽 우선
        match_a = bool(self.KR_PLATE_PATTERN.search(text_a))
        match_b = bool(self.KR_PLATE_PATTERN.search(text_b))
        if match_a and not match_b:
            return text_a, conf_a
        if match_b and not match_a:
            return text_b, conf_b
        return (text_a, conf_a) if conf_a >= conf_b else (text_b, conf_b)

    def _normalize_plate(self, raw: str) -> Optional[str]:
        """오인식 보정 + 정규식 추출"""
        corrected = "".join(self.CHAR_CORRECTION.get(c, c) for c in raw)
        match = self.KR_PLATE_PATTERN.search(corrected)
        if not match:
            return None
        num1, korean, num2 = match.group(1), match.group(2), match.group(3)
        return f"{num1}{korean}{num2}"

    def batch_read(self, rois: list) -> list:
        """여러 ROI 일괄 처리"""
        return [self.read_plate(roi) for roi in rois]
