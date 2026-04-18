"""OCR 전처리 파이프라인 단위 테스트"""

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ocr.preprocessor import PlatePreprocessor

CONFIG = {
    "preprocessing": {
        "gaussian_blur_kernel": 5,
        "clahe_clip_limit": 3.0,
        "clahe_tile_grid": [8, 8],
        "deskew_threshold": 2.0,
        "morph_kernel_size": 3,
        "use_c_extension": False,
    }
}


@pytest.fixture
def preprocessor():
    return PlatePreprocessor(CONFIG)


@pytest.fixture
def sample_plate():
    # 회색조 번호판 모조 이미지
    img = np.ones((60, 200, 3), dtype=np.uint8) * 200
    return img


def test_normalize_size(preprocessor, sample_plate):
    out = preprocessor._normalize_size(sample_plate)
    assert out.shape[1] >= 400


def test_to_gray(preprocessor, sample_plate):
    gray = preprocessor._to_gray(sample_plate)
    assert len(gray.shape) == 2


def test_enhance_contrast(preprocessor, sample_plate):
    gray = preprocessor._to_gray(sample_plate)
    enhanced = preprocessor._enhance_contrast(gray)
    assert enhanced.shape == gray.shape


def test_binarize_output_values(preprocessor, sample_plate):
    gray = preprocessor._to_gray(sample_plate)
    binary = preprocessor._binarize(gray)
    unique = np.unique(binary)
    assert set(unique).issubset({0, 255})


def test_full_pipeline_does_not_crash(preprocessor, sample_plate):
    result = preprocessor.process(sample_plate)
    assert result is not None
    assert result.size > 0


def test_empty_input(preprocessor):
    empty = np.array([])
    result = preprocessor.process(empty)
    assert result is not None
