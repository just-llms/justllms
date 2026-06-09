"""SQLite-backed persistent cache backend."""

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from justllms.cache.base import BaseCacheBackend

DEFAULT_CACHE_PATH = "~/.justllms/cache.db"

# Purge expired rows on every Nth write to keep the file from growing.
_PURGE_EVERY_N_SETS = 100


class DiskCache(BaseCacheBackend):
    """Persistent cache backed by a SQLite database file.

    Survives process restarts. Thread-safe via a single shared connection
    guarded by a lock. Expired rows are purged lazily on reads and
    periodically on writes.

    Args:
        path: Path to the SQLite database file. Defaults to
            ``~/.justllms/cache.db``. Parent directories are created.
    """

    def __init__(self, path: Optional[Union[str, Path]] = None) -> None:
        resolved = Path(path if path is not None else DEFAULT_CACHE_PATH).expanduser()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.path = resolved
        self._lock = threading.Lock()
        self._set_count = 0
        self._hits = 0
        self._misses = 0
        self._conn = sqlite3.connect(str(resolved), check_same_thread=False)
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS response_cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at REAL,
                    created_at REAL NOT NULL
                )
                """)
            self._conn.commit()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached value, deleting it if expired."""
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT value, expires_at FROM response_cache WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                self._misses += 1
                return None
            value_text, expires_at = row
            if expires_at is not None and now >= expires_at:
                self._conn.execute("DELETE FROM response_cache WHERE key = ?", (key,))
                self._conn.commit()
                self._misses += 1
                return None
            self._hits += 1
        loaded = json.loads(value_text)
        return loaded if isinstance(loaded, dict) else None

    def set(self, key: str, value: Dict[str, Any], ttl: Optional[float] = None) -> None:
        """Store a value, occasionally purging expired rows."""
        now = time.time()
        expires_at = now + ttl if ttl is not None else None
        value_text = json.dumps(value, default=str)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO response_cache (key, value, expires_at, created_at) "
                "VALUES (?, ?, ?, ?)",
                (key, value_text, expires_at, now),
            )
            self._set_count += 1
            if self._set_count % _PURGE_EVERY_N_SETS == 0:
                self._conn.execute(
                    "DELETE FROM response_cache WHERE expires_at IS NOT NULL AND expires_at <= ?",
                    (now,),
                )
            self._conn.commit()

    def delete(self, key: str) -> None:
        """Remove a single entry if present."""
        with self._lock:
            self._conn.execute("DELETE FROM response_cache WHERE key = ?", (key,))
            self._conn.commit()

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._conn.execute("DELETE FROM response_cache")
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()

    @property
    def stats(self) -> Dict[str, Any]:
        """Return hit/miss counters and current row count."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM response_cache").fetchone()
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": int(row[0]) if row else 0,
                "path": str(self.path),
            }
