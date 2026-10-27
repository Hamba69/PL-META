from datetime import date, datetime, timedelta, timezone
from contextlib import contextmanager
import math
from pathlib import Path
import sqlite3
from typing import Iterator


class CostTracker:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS model_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    model TEXT NOT NULL,
                    call_kind TEXT NOT NULL,
                    unit_name TEXT NOT NULL,
                    units REAL NOT NULL,
                    cost_usd REAL NOT NULL,
                    occurred_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS model_calls_day_idx ON model_calls(occurred_at)"
            )

    def record_call(
        self,
        *,
        model: str,
        call_kind: str,
        unit_name: str,
        units: float,
        cost_usd: float,
        run_id: str | None = None,
        occurred_at: datetime | None = None,
    ) -> int:
        if not model.strip() or not call_kind.strip() or not unit_name.strip():
            raise ValueError("model, call kind, and unit name must not be empty")
        if not math.isfinite(units) or units < 0:
            raise ValueError("consumed units must be finite and non-negative")
        if not math.isfinite(cost_usd) or cost_usd < 0:
            raise ValueError("cost must be finite and non-negative")
        timestamp = occurred_at or datetime.now(timezone.utc)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("call timestamp must include a timezone")
        timestamp_utc = timestamp.astimezone(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO model_calls (
                    run_id, model, call_kind, unit_name, units, cost_usd, occurred_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, model, call_kind, unit_name, units, cost_usd, timestamp_utc),
            )
            return int(cursor.lastrowid)

    def daily_total(self, day: date | None = None) -> float:
        target_day = day or datetime.now(timezone.utc).date()
        start = datetime.combine(target_day, datetime.min.time(), tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COALESCE(SUM(cost_usd), 0.0) AS total
                FROM model_calls
                WHERE occurred_at >= ? AND occurred_at < ?
                """,
                (start.isoformat(), end.isoformat()),
            ).fetchone()
        return float(row["total"])

    def run_total(self, run_id: str) -> float:
        if not run_id:
            raise ValueError("run ID must not be empty")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COALESCE(SUM(cost_usd), 0.0) AS total
                FROM model_calls
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        return float(row["total"])

    def run_breakdown(self, run_id: str) -> dict[str, float]:
        if not run_id:
            raise ValueError("run ID must not be empty")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT call_kind, SUM(cost_usd) AS total
                FROM model_calls
                WHERE run_id = ?
                GROUP BY call_kind
                """,
                (run_id,),
            ).fetchall()
        return {str(row["call_kind"]): float(row["total"]) for row in rows}

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()
