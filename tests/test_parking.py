"""주차장 관리 시스템 단위 테스트"""

import sys, os, tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.database.db_manager import DatabaseManager
from src.parking.slot_manager import SlotManager


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    return DatabaseManager(db_path)


@pytest.fixture
def slot_mgr(db):
    config = {"parking": {"slot_layout": {"rows": 3, "cols": 5}}}
    return SlotManager(config, db)


# ── DB 테스트 ──────────────────────────────────────────────────

def test_register_and_lookup(db):
    ok = db.register_vehicle("12가3456", "홍길동", "010-0000-0000")
    assert ok is True
    assert db.is_registered("12가3456") is True
    assert db.is_registered("99나9999") is False


def test_duplicate_registration(db):
    db.register_vehicle("12가3456", "홍길동")
    ok = db.register_vehicle("12가3456", "다른사람")
    assert ok is False


def test_entry_exit(db):
    db.register_vehicle("12가3456", "홍길동")
    record_id = db.record_entry("12가3456", slot_id=1)
    assert isinstance(record_id, int)
    current = db.get_current_parking()
    assert any(r["plate_num"] == "12가3456" for r in current)

    ok = db.record_exit("12가3456", fee=1500)
    assert ok is True
    current = db.get_current_parking()
    assert not any(r["plate_num"] == "12가3456" for r in current)


def test_stats(db):
    stats = db.get_stats()
    assert "total_slots" in stats
    assert "occupied" in stats
    assert "occupancy_rate" in stats


# ── 슬롯 관리 테스트 ────────────────────────────────────────────

def test_slot_initialization(slot_mgr, db):
    slots = db.get_all_slots()
    assert len(slots) == 15  # 3×5


def test_assign_unregistered_vehicle(slot_mgr, db):
    slot = slot_mgr.assign_slot("99나9999")
    assert slot is not None
    assert slot["is_occupied"] == 1 or slot["is_occupied"] is True


def test_assign_registered_vehicle(slot_mgr, db):
    db.register_vehicle("12가3456", "홍길동", assigned_slot=3)
    slot = slot_mgr.assign_slot("12가3456")
    assert slot is not None
    assert slot["id"] == 3


def test_release_slot(slot_mgr, db):
    slot = slot_mgr.assign_slot("99나0001")
    assert slot is not None
    slot_mgr.release_slot(slot["id"])
    all_slots = {s["id"]: s for s in db.get_all_slots()}
    assert all_slots[slot["id"]]["is_occupied"] == 0
