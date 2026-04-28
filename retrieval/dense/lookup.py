"""SQLite-backed passage text lookup for dense retrieval serving.

Maps chunk_id → (passage_id, text). At serve time, after FAISS returns chunk IDs,
this lookup fetches the passage text for context assembly.

For MS MARCO v1 (passages are not chunked further): chunk_id == passage_id.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


class PassageLookup:
    """Persistent chunk_id → (passage_id, text) store backed by SQLite.

    Usage:
        # Build once offline
        with PassageLookup(db_path).open() as lk:
            lk.build([(pid, pid, text) for pid, text in passages])

        # Query at serve time
        with PassageLookup(db_path).open() as lk:
            passage_id, text = lk.get("pid_123")
    """

    _CREATE_SQL = """
        CREATE TABLE IF NOT EXISTS passages (
            chunk_id   TEXT PRIMARY KEY,
            passage_id TEXT NOT NULL,
            text       TEXT NOT NULL
        ) WITHOUT ROWID
    """
    _INSERT_SQL = "INSERT OR REPLACE INTO passages VALUES (?, ?, ?)"

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def open(self) -> "PassageLookup":
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-65536")  # 64 MB page cache
        self._conn.execute(self._CREATE_SQL)
        return self

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "PassageLookup":
        return self.open()

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def build(
        self,
        chunks: list[tuple[str, str, str]],
        batch_size: int = 10_000,
    ) -> None:
        """Bulk-insert (chunk_id, passage_id, text) rows.

        Args:
            chunks:     list of (chunk_id, passage_id, text)
            batch_size: rows per SQLite transaction
        """
        conn = self._conn
        conn.execute("BEGIN")
        for i in range(0, len(chunks), batch_size):
            conn.executemany(self._INSERT_SQL, chunks[i : i + batch_size])
        conn.commit()
        print(f"PassageLookup: {len(chunks):,} rows stored in {self.db_path}")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, chunk_id: str) -> tuple[str, str] | None:
        """Return (passage_id, text) for chunk_id, or None."""
        row = self._conn.execute(
            "SELECT passage_id, text FROM passages WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        return (row[0], row[1]) if row else None

    def get_batch(self, chunk_ids: list[str]) -> dict[str, tuple[str, str]]:
        """Return {chunk_id: (passage_id, text)} for a list of chunk_ids.

        Missing IDs are silently omitted from the result.
        """
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"SELECT chunk_id, passage_id, text FROM passages "
            f"WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        return {row[0]: (row[1], row[2]) for row in rows}

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM passages").fetchone()[0]

    # ------------------------------------------------------------------
    # Convenience builder
    # ------------------------------------------------------------------

    @classmethod
    def from_corpus(
        cls,
        passages: list[tuple[str, str]],
        db_path: Path,
    ) -> "PassageLookup":
        """Create a PassageLookup from a (pid, text) list in one call.

        For MS MARCO v1: chunk_id == passage_id (no chunking applied).
        """
        lk = cls(db_path)
        with lk:
            rows = [(pid, pid, text) for pid, text in passages]
            lk.build(rows)
        return lk
