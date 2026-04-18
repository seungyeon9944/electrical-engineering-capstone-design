"""
이미지 유틸리티 함수
"""

import cv2
import numpy as np
from typing import Dict, Any
from datetime import datetime


def draw_status_overlay(
    frame: np.ndarray,
    stats: Dict[str, Any],
    current_plates: Dict,
) -> np.ndarray:
    """화면 우측 상단에 주차장 현황 오버레이 표시"""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    panel_w, panel_h = 280, 160
    x0, y0 = w - panel_w - 10, 10
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    lines = [
        f"주차장 현황  {datetime.now().strftime('%H:%M:%S')}",
        f"전체: {stats.get('total_slots', 0)}  점유: {stats.get('occupied', 0)}",
        f"여유: {stats.get('available', 0)}  ({stats.get('occupancy_rate', 0)}%)",
        f"오늘 입차: {stats.get('today_entries', 0)}",
        f"현재 내부: {len(current_plates)}대",
    ]
    for i, line in enumerate(lines):
        color = (0, 255, 200) if i == 0 else (220, 220, 220)
        cv2.putText(
            frame, line,
            (x0 + 8, y0 + 22 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1,
        )
    return frame


def resize_with_aspect(img: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = target_w / w
    return cv2.resize(img, (target_w, int(h * scale)))


def put_korean_text(img: np.ndarray, text: str, pos, color=(255, 255, 255)):
    """PIL을 이용해 한글 텍스트를 이미지에 렌더링"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("malgun.ttf", 20)
        except Exception:
            font = ImageFont.load_default()
        draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except ImportError:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return img
