"""
SQLite 데이터베이스 관리 모듈
등록 차량, 주차 기록, 슬롯 배정 정보 관리
"""

import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


class DatabaseManager:
    def __init__(self, db_path: str = "data/parking.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS registered_vehicles (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_num   TEXT    NOT NULL UNIQUE,
                    owner_name  TEXT    NOT NULL,
                    owner_phone TEXT,
                    assigned_slot INTEGER,
                    vehicle_type TEXT DEFAULT 'sedan',
                    registered_at TEXT DEFAULT (datetime('now','localtime')),
                    is_active   INTEGER DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS parking_records (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_num   TEXT    NOT NULL,
                    slot_id     INTEGER,
                    entry_time  TEXT    NOT NULL,
                    exit_time   TEXT,
                    fee         REAL    DEFAULT 0,
                    is_registered INTEGER DEFAULT 0,
                    FOREIGN KEY (slot_id) REFERENCES parking_slots(id)
                );

                CREATE TABLE IF NOT EXISTS parking_slots (
                    id          INTEGER PRIMARY KEY,
                    slot_name   TEXT    NOT NULL UNIQUE,
                    row_idx     INTEGER NOT NULL,
                    col_idx     INTEGER NOT NULL,
                    is_occupied INTEGER DEFAULT 0,
                    is_reserved INTEGER DEFAULT 0,
                    reserved_for TEXT,
                    slot_type   TEXT    DEFAULT 'normal'
                );

                CREATE TABLE IF NOT EXISTS access_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_num   TEXT,
                    action      TEXT,    -- 'ENTRY' | 'EXIT' | 'DENIED'
                    timestamp   TEXT     DEFAULT (datetime('now','localtime')),
                    note        TEXT
                );
            """)

    # ── 차량 등록 ──────────────────────────────────────────────

    def register_vehicle(
        self,
        plate_num: str,
        owner_name: str,
        owner_phone: str = "",
        assigned_slot: Optional[int] = None,
        vehicle_type: str = "sedan",
    ) -> bool:
        try:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO registered_vehicles
                       (plate_num, owner_name, owner_phone, assigned_slot, vehicle_type)
                       VALUES (?, ?, ?, ?, ?)""",
                    (plate_num, owner_name, owner_phone, assigned_slot, vehicle_type),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def is_registered(self, plate_num: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM registered_vehicles WHERE plate_num=? AND is_active=1",
                (plate_num,),
            ).fetchone()
        return row is not None

    def get_vehicle_info(self, plate_num: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM registered_vehicles WHERE plate_num=? AND is_active=1",
                (plate_num,),
            ).fetchone()
        return dict(row) if row else None

    def get_all_registered(self) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM registered_vehicles WHERE is_active=1 ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── 주차 기록 ──────────────────────────────────────────────

    def record_entry(self, plate_num: str, slot_id: Optional[int]) -> int:
        is_reg = 1 if self.is_registered(plate_num) else 0
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO parking_records (plate_num, slot_id, entry_time, is_registered)
                   VALUES (?, ?, datetime('now','localtime'), ?)""",
                (plate_num, slot_id, is_reg),
            )
            record_id = cur.lastrowid
        self._log_access(plate_num, "ENTRY")
        return record_id

    def record_exit(self, plate_num: str, fee: float = 0.0) -> bool:
        with self._connect() as conn:
            result = conn.execute(
                """UPDATE parking_records
                   SET exit_time = datetime('now','localtime'), fee = ?
                   WHERE plate_num = ? AND exit_time IS NULL""",
                (fee, plate_num),
            )
            updated = result.rowcount > 0
        if updated:
            self._log_access(plate_num, "EXIT")
        return updated

    def get_current_parking(self) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT r.*, v.owner_name, v.owner_phone
                   FROM parking_records r
                   LEFT JOIN registered_vehicles v ON r.plate_num = v.plate_num
                   WHERE r.exit_time IS NULL
                   ORDER BY r.entry_time""",
            ).fetchall()
        return [dict(r) for r in rows]

    # ── 슬롯 관리 ──────────────────────────────────────────────

    def initialize_slots(self, rows: int, cols: int):
        with self._connect() as conn:
            for r in range(rows):
                for c in range(cols):
                    slot_name = f"{chr(65+r)}{c+1:02d}"
                    slot_id = r * cols + c + 1
                    conn.execute(
                        """INSERT OR IGNORE INTO parking_slots
                           (id, slot_name, row_idx, col_idx) VALUES (?, ?, ?, ?)""",
                        (slot_id, slot_name, r, c),
                    )

    def set_slot_occupied(self, slot_id: int, occupied: bool):
        with self._connect() as conn:
            conn.execute(
                "UPDATE parking_slots SET is_occupied=? WHERE id=?",
                (1 if occupied else 0, slot_id),
            )

    def get_available_slots(self) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM parking_slots
                   WHERE is_occupied=0 AND is_reserved=0
                   ORDER BY id""",
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_slots(self) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM parking_slots ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── 접근 로그 ──────────────────────────────────────────────

    def _log_access(self, plate_num: str, action: str, note: str = ""):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO access_log (plate_num, action, note) VALUES (?, ?, ?)",
                (plate_num, action, note),
            )

    def get_access_logs(self, limit: int = 100) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM access_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── 통계 ──────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            total_slots = conn.execute(
                "SELECT COUNT(*) FROM parking_slots"
            ).fetchone()[0]
            occupied = conn.execute(
                "SELECT COUNT(*) FROM parking_slots WHERE is_occupied=1"
            ).fetchone()[0]
            total_registered = conn.execute(
                "SELECT COUNT(*) FROM registered_vehicles WHERE is_active=1"
            ).fetchone()[0]
            today_entries = conn.execute(
                """SELECT COUNT(*) FROM parking_records
                   WHERE date(entry_time)=date('now','localtime')"""
            ).fetchone()[0]
        return {
            "total_slots": total_slots,
            "occupied": occupied,
            "available": total_slots - occupied,
            "occupancy_rate": round(occupied / total_slots * 100, 1) if total_slots else 0,
            "total_registered": total_registered,
            "today_entries": today_entries,
        }
