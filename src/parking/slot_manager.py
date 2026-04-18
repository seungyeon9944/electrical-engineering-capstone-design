"""
주차 슬롯 관리 모듈
지정좌석 배정 및 실시간 슬롯 상태 추적
"""

from typing import Optional, Dict, List
from ..database.db_manager import DatabaseManager


class SlotManager:
    def __init__(self, config: dict, db: DatabaseManager):
        park_cfg = config.get("parking", {})
        self.rows = park_cfg.get("slot_layout", {}).get("rows", 5)
        self.cols = park_cfg.get("slot_layout", {}).get("cols", 10)
        self.total_slots = self.rows * self.cols
        self.db = db
        self.db.initialize_slots(self.rows, self.cols)

    def assign_slot(self, plate_num: str) -> Optional[Dict]:
        """
        등록 차량이면 지정 슬롯, 미등록이면 빈 슬롯 중 가장 가까운 입구 배정
        """
        vehicle_info = self.db.get_vehicle_info(plate_num)
        if vehicle_info and vehicle_info.get("assigned_slot"):
            slot_id = vehicle_info["assigned_slot"]
            all_slots = {s["id"]: s for s in self.db.get_all_slots()}
            slot = all_slots.get(slot_id)
            if slot and not slot["is_occupied"]:
                self.db.set_slot_occupied(slot_id, True)
                return slot
            # 지정 슬롯이 이미 점유된 경우 인접 슬롯 탐색
            return self._assign_nearest(slot_id)

        # 미등록 차량: 첫 번째 빈 슬롯
        available = self.db.get_available_slots()
        if not available:
            return None
        slot = available[0]
        self.db.set_slot_occupied(slot["id"], True)
        return slot

    def _assign_nearest(self, preferred_id: int) -> Optional[Dict]:
        available = self.db.get_available_slots()
        if not available:
            return None
        # 선호 슬롯과 ID 차이가 가장 작은 슬롯
        nearest = min(available, key=lambda s: abs(s["id"] - preferred_id))
        self.db.set_slot_occupied(nearest["id"], True)
        return nearest

    def release_slot(self, slot_id: int):
        self.db.set_slot_occupied(slot_id, False)

    def get_layout_map(self) -> List[List[Dict]]:
        """2D 슬롯 배치 맵 반환 (rows × cols)"""
        all_slots = self.db.get_all_slots()
        slot_map = {(s["row_idx"], s["col_idx"]): s for s in all_slots}
        layout = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                row.append(slot_map.get((r, c), {}))
            layout.append(row)
        return layout

    def print_layout(self):
        layout = self.get_layout_map()
        header = "    " + "  ".join(f"{c+1:02d}" for c in range(self.cols))
        print(header)
        print("    " + "--" * self.cols * 2)
        for r, row in enumerate(layout):
            row_str = chr(65 + r) + " | "
            for slot in row:
                if not slot:
                    row_str += "?? "
                elif slot["is_occupied"]:
                    row_str += "[X]"
                elif slot["is_reserved"]:
                    row_str += "[R]"
                else:
                    row_str += "[ ]"
            print(row_str)
        stats = self.db.get_stats()
        print(f"\n점유: {stats['occupied']}/{stats['total_slots']} "
              f"({stats['occupancy_rate']}%)")
