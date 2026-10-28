from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator

_SHA256_PATTERN = set("0123456789abcdef")


@dataclass(frozen=True)
class CacheEntry:
    value: Any
    created_at: datetime


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("hash chunk size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class ContentCache:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_cache (
                    cache_key TEXT PRIMARY KEY,
                    video_hash TEXT NOT NULL,
                    step TEXT NOT NULL,
                    version TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def make_key(
        self,
        video_hash: str,
        step: str,
        version: str,
        parameters: dict[str, Any],
    ) -> str:
        _validate_hash(video_hash)
        if not step.strip() or not version.strip():
            raise ValueError("cache step and version must not be empty")
        identity = json.dumps(
            {
                "video_hash": video_hash,
                "step": step,
                "version": version,
                "parameters": parameters,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def get(
        self,
        video_hash: str,
        step: str,
        version: str,
        parameters: dict[str, Any],
    ) -> CacheEntry | None:
        cache_key = self.make_key(video_hash, step, version, parameters)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value_json, created_at FROM pipeline_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return CacheEntry(
            value=json.loads(row["value_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def put(
        self,
        video_hash: str,
        step: str,
        version: str,
        parameters: dict[str, Any],
        value: Any,
    ) -> str:
        cache_key = self.make_key(video_hash, step, version, parameters)
        parameters_json = json.dumps(
            parameters,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        value_json = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO pipeline_cache (
                    cache_key, video_hash, step, version,
                    parameters_json, value_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    parameters_json = excluded.parameters_json,
                    value_json = excluded.value_json,
                    created_at = excluded.created_at
                """,
                (
                    cache_key,
                    video_hash,
                    step,
                    version,
                    parameters_json,
                    value_json,
                    created_at,
                ),
            )
        return cache_key

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()


def _validate_hash(video_hash: str) -> None:
    if len(video_hash) != 64 or any(character not in _SHA256_PATTERN for character in video_hash):
        raise ValueError("video hash must be a lowercase SHA-256 hex digest")
