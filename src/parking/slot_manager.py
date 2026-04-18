"""
주차 슬롯 관리 모듈 (다층 지원)

YOLOv7 + OCR로 인식된 번호판을 받아
MultiFloorAllocator 배치 알고리즘으로 최적 슬롯을 배정한다.
"""

from typing import Optional, Dict, List

from ..database.db_manager import DatabaseManager
from .allocation_algorithm import MultiFloorAllocator, SlotCandidate


class SlotManager:
    def __init__(self, config: dict, db: DatabaseManager):
        park_cfg = config.get("parking", {})
        layout = park_cfg.get("slot_layout", {})
        self.floors = layout.get("floors", 1)
        self.rows = layout.get("rows", 5)
        self.cols = layout.get("cols", 10)
        self.total_slots = self.floors * self.rows * self.cols
        self.db = db
        self.allocator = MultiFloorAllocator(config)
        self.db.initialize_slots(self.rows, self.cols, self.floors)

    # ── 슬롯 배정 ─────────────────────────────────────────────────

    def assign_slot(self, plate_num: str) -> Optional[Dict]:
        """
        번호판으로 최적 슬롯 배정.

        등록 차량은 지정 슬롯 우선,
        미등록 차량은 MultiFloorAllocator 비용 함수로 결정.
        """
        vehicle_info = self.db.get_vehicle_info(plate_num)
        preferred_slot_id: Optional[int] = None
        preferred_floor: Optional[int] = None
        vehicle_type = "sedan"

        if vehicle_info:
            preferred_slot_id = vehicle_info.get("assigned_slot")
            vehicle_type = vehicle_info.get("vehicle_type", "sedan")

        candidates = self._build_candidates()

        # 지정 슬롯이 있으면 preferred_floor 추출
        if preferred_slot_id:
            ref = next((c for c in candidates if c.slot_id == preferred_slot_id), None)
            if ref:
                preferred_floor = ref.floor

        best = self.allocator.assign(
            candidates,
            vehicle_type=vehicle_type,
            preferred_slot_id=preferred_slot_id,
            preferred_floor=preferred_floor,
        )
        if best is None:
            return None

        self.db.set_slot_occupied(best.slot_id, True)
        return self.db.get_slot_by_id(best.slot_id)

    def release_slot(self, slot_id: int):
        self.db.set_slot_occupied(slot_id, False)

    # ── 조회/시각화 ───────────────────────────────────────────────

    def get_layout_map(self) -> Dict[int, List[List[Dict]]]:
        """층 → 2D 슬롯 배치 맵 반환"""
        all_slots = self.db.get_all_slots()
        layout: Dict[int, List[List[Dict]]] = {}
        for f in range(self.floors):
            grid = [[{} for _ in range(self.cols)] for _ in range(self.rows)]
            layout[f] = grid
        for s in all_slots:
            f, r, c = s.get("floor", 0), s["row_idx"], s["col_idx"]
            if f in layout and 0 <= r < self.rows and 0 <= c < self.cols:
                layout[f][r][c] = s
        return layout

    def get_floor_summary(self) -> Dict[int, Dict]:
        """층별 점유 현황 반환"""
        candidates = self._build_candidates()
        return self.allocator.floor_summary(candidates)

    def get_top_slots(self, top_k: int = 5) -> list:
        """비용 순 상위 추천 슬롯 목록 반환 (배치 알고리즘 출력)"""
        candidates = self._build_candidates()
        return self.allocator.rank_candidates(candidates, top_k=top_k)

    def print_layout(self):
        layout_map = self.get_layout_map()
        summary = self.get_floor_summary()
        for f in range(self.floors):
            occ_info = summary.get(f, {})
            print(f"\n[{f+1}층]  점유: {occ_info.get('occupied', 0)}/"
                  f"{occ_info.get('total', 0)}  ({occ_info.get('occupancy_rate', 0)}%)")
            header = "    " + "  ".join(f"{c+1:02d}" for c in range(self.cols))
            print(header)
            print("    " + "---" * self.cols)
            grid = layout_map.get(f, [])
            for r, row in enumerate(grid):
                row_str = chr(65 + r) + " | "
                for slot in row:
                    if not slot:
                        row_str += " ? "
                    elif slot.get("is_occupied"):
                        row_str += "[X]"
                    elif slot.get("is_reserved"):
                        row_str += "[R]"
                    else:
                        row_str += "[ ]"
                print(row_str)

        stats = self.db.get_stats()
        print(f"\n전체: {stats['occupied']}/{stats['total_slots']} "
              f"({stats['occupancy_rate']}%)")

    # ── 내부 헬퍼 ─────────────────────────────────────────────────

    def _build_candidates(self) -> list:
        """DB 슬롯 목록 → SlotCandidate 리스트 변환"""
        all_slots = self.db.get_all_slots()
        return [
            SlotCandidate(
                slot_id=s["id"],
                slot_name=s["slot_name"],
                floor=s.get("floor", 0),
                row_idx=s["row_idx"],
                col_idx=s["col_idx"],
                slot_type=s.get("slot_type", "normal"),
                is_occupied=bool(s["is_occupied"]),
                is_reserved=bool(s["is_reserved"]),
            )
            for s in all_slots
        ]
