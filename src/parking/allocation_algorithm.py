"""
다층 주차장 최적 배치 알고리즘

[알고리즘 개요]
가중치 기반 다기준 비용 함수(Weighted Multi-Criteria Cost Function)를 이용한
그리디 슬롯 배정 알고리즘.

비용 함수:
    cost(s) = w_floor   * floor(s)
            + w_dist    * manhattan_distance(s, floor_entrance)
            + w_load    * floor_occupancy_rate(floor(s)) * 10

배정 우선순위:
    1. 등록 차량 → 지정 슬롯 (같은 층 인접 슬롯 폴백)
    2. 장애인 차량 → 1층 전용 슬롯 → 1층 입구 최근거리
    3. 전기차(EV)  → EV 충전 슬롯 우선
    4. 일반 미등록 → cost 최소 슬롯 (층 + 거리 + 부하 균형)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class SlotCandidate:
    """슬롯 한 칸의 속성 집합"""

    slot_id: int
    slot_name: str
    floor: int          # 층 (0 = 지상 1층)
    row_idx: int        # 층 내 행 인덱스
    col_idx: int        # 층 내 열 인덱스
    slot_type: str = "normal"   # normal | disabled | ev | oversized
    is_occupied: bool = False
    is_reserved: bool = False

    def cost(
        self,
        floor_occupancy: Dict[int, float],
        entrance_row: int = 0,
        entrance_col: int = 0,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        슬롯 배정 비용 계산.

        낮을수록 입구에서 가깝고 덜 붐비는 슬롯.
        """
        w = weights or {"floor": 1.5, "distance": 1.0, "load": 0.8}
        floor_cost = self.floor * w["floor"]
        dist_cost = (
            abs(self.row_idx - entrance_row) + abs(self.col_idx - entrance_col)
        ) * w["distance"]
        load_cost = floor_occupancy.get(self.floor, 0.0) * 10.0 * w["load"]
        return floor_cost + dist_cost + load_cost


class MultiFloorAllocator:
    """
    다층 주차장 슬롯 배정 알고리즘.

    가중치 기반 비용 함수로 최적 슬롯을 그리디하게 선택한다.
    층 전환 비용 · 보행 거리 · 층별 부하 균형을 동시에 고려한다.

    Parameters
    ----------
    config : dict
        config.yaml 전체 딕셔너리. ``allocation`` 섹션을 읽는다.

    Usage
    -----
    >>> allocator = MultiFloorAllocator(config)
    >>> best = allocator.assign(candidates, vehicle_type="sedan")
    """

    # 층별 입구는 (row=0, col=0)으로 가정 (config로 변경 가능)
    _ENTRANCE_DEFAULT = {"entrance_row": 0, "entrance_col": 0}

    def __init__(self, config: dict):
        algo_cfg = config.get("allocation", {})
        self.weights: Dict[str, float] = algo_cfg.get(
            "weights", {"floor": 1.5, "distance": 1.0, "load": 0.8}
        )
        self.enable_load_balance: bool = algo_cfg.get("load_balance", True)
        entrance_cfg = algo_cfg.get("entrance_position", {})
        self._entrance = {
            "entrance_row": entrance_cfg.get("row", 0),
            "entrance_col": entrance_cfg.get("col", 0),
        }

    # ── 공개 API ──────────────────────────────────────────────────

    def assign(
        self,
        candidates: List[SlotCandidate],
        vehicle_type: str = "sedan",
        preferred_slot_id: Optional[int] = None,
        preferred_floor: Optional[int] = None,
    ) -> Optional[SlotCandidate]:
        """
        최적 슬롯 1개를 반환한다.

        Parameters
        ----------
        candidates      : 전체 슬롯 목록 (점유/예약 포함)
        vehicle_type    : "sedan" | "disabled" | "ev" | "oversized"
        preferred_slot_id : 등록 차량의 지정 슬롯 ID
        preferred_floor : 등록 차량의 선호 층
        """
        available = [
            s for s in candidates if not s.is_occupied and not s.is_reserved
        ]
        if not available:
            return None

        # 1순위: 등록 차량 지정 슬롯
        if preferred_slot_id is not None:
            result = self._assign_registered(
                available, candidates, preferred_slot_id, preferred_floor
            )
            if result:
                return result

        # 2순위: 장애인 차량
        if vehicle_type == "disabled":
            return self._assign_disabled(available)

        # 3순위: 전기차 충전 슬롯
        if vehicle_type == "ev":
            ev_slots = [s for s in available if s.slot_type == "ev"]
            if ev_slots:
                return self._min_cost(ev_slots, candidates)

        # 4순위: 비용 최소화 (일반)
        return self._min_cost(available, candidates)

    def rank_candidates(
        self,
        candidates: List[SlotCandidate],
        top_k: int = 5,
    ) -> List[Tuple[SlotCandidate, float]]:
        """빈 슬롯을 비용 오름차순으로 정렬하여 반환 (시각화·디버그용)."""
        available = [s for s in candidates if not s.is_occupied and not s.is_reserved]
        floor_occ = self._floor_occupancy(candidates)
        scored = [
            (s, s.cost(floor_occ, **self._entrance, weights=self.weights))
            for s in available
        ]
        scored.sort(key=lambda x: x[1])
        return scored[:top_k]

    def floor_summary(self, candidates: List[SlotCandidate]) -> Dict[int, Dict]:
        """층별 점유 현황 딕셔너리 반환."""
        summary: Dict[int, Dict] = {}
        for s in candidates:
            f = s.floor
            if f not in summary:
                summary[f] = {"total": 0, "occupied": 0, "available": 0}
            summary[f]["total"] += 1
            if s.is_occupied:
                summary[f]["occupied"] += 1
            else:
                summary[f]["available"] += 1
        for f in summary:
            t = summary[f]["total"]
            summary[f]["occupancy_rate"] = (
                round(summary[f]["occupied"] / t * 100, 1) if t else 0.0
            )
        return summary

    # ── 내부 헬퍼 ─────────────────────────────────────────────────

    def _assign_registered(
        self,
        available: List[SlotCandidate],
        all_candidates: List[SlotCandidate],
        preferred_slot_id: int,
        preferred_floor: Optional[int],
    ) -> Optional[SlotCandidate]:
        # 지정 슬롯이 비어 있으면 즉시 반환
        exact = next((s for s in available if s.slot_id == preferred_slot_id), None)
        if exact:
            return exact

        # 같은 층에서 지정 슬롯과 맨해튼 거리 최소 슬롯
        target_floor = preferred_floor if preferred_floor is not None else None
        pool = (
            [s for s in available if s.floor == target_floor]
            if target_floor is not None
            else available
        )
        if not pool:
            pool = available

        pref_ref = next(
            (s for s in all_candidates if s.slot_id == preferred_slot_id), None
        )
        if pref_ref:
            return min(
                pool,
                key=lambda s: abs(s.row_idx - pref_ref.row_idx)
                + abs(s.col_idx - pref_ref.col_idx),
            )
        return min(pool, key=lambda s: abs(s.slot_id - preferred_slot_id))

    def _assign_disabled(
        self, available: List[SlotCandidate]
    ) -> Optional[SlotCandidate]:
        # 전용 슬롯 우선
        special = [s for s in available if s.slot_type == "disabled"]
        if special:
            return min(special, key=lambda s: s.row_idx + s.col_idx)
        # 없으면 1층(floor=0) 입구 최근거리
        ground = [s for s in available if s.floor == 0]
        pool = ground if ground else available
        return min(pool, key=lambda s: s.row_idx + s.col_idx)

    def _min_cost(
        self,
        pool: List[SlotCandidate],
        all_candidates: List[SlotCandidate],
    ) -> Optional[SlotCandidate]:
        if not pool:
            return None
        floor_occ = self._floor_occupancy(all_candidates) if self.enable_load_balance else {}
        return min(
            pool,
            key=lambda s: s.cost(floor_occ, **self._entrance, weights=self.weights),
        )

    @staticmethod
    def _floor_occupancy(candidates: List[SlotCandidate]) -> Dict[int, float]:
        floor_total: Dict[int, int] = {}
        floor_occ_cnt: Dict[int, int] = {}
        for s in candidates:
            floor_total[s.floor] = floor_total.get(s.floor, 0) + 1
            if s.is_occupied:
                floor_occ_cnt[s.floor] = floor_occ_cnt.get(s.floor, 0) + 1
        return {
            f: floor_occ_cnt.get(f, 0) / floor_total[f]
            for f in floor_total
        }
