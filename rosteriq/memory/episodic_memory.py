import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional


class EpisodicMemory:
    """
    Episodic memory backed by a lightweight SQLite database.
    Stores past investigations so the agent can reference prior work.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        default_path = os.path.join(base_dir, "memory", "episodic_memory.db")
        self.db_path = db_path or default_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS investigations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    market TEXT,
                    issue TEXT,
                    stage TEXT,
                    organizations TEXT,
                    summary TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def add_episode(
        self,
        market: Optional[str],
        issue: str,
        stage: Optional[str],
        organizations: Optional[List[str]],
        summary: str,
    ) -> None:
        """
        Persist a new investigation episode.
        """
        ts = datetime.utcnow().isoformat()
        org_str = ", ".join(sorted(set(organizations))) if organizations else None

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO investigations (timestamp, market, issue, stage, organizations, summary)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ts, market, issue, stage, org_str, summary),
            )
            conn.commit()
        finally:
            conn.close()

    def retrieve_recent(self, limit: int = 5) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM investigations ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    def retrieve_similar(
        self,
        market: Optional[str],
        issue: str,
        n_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent investigations that match the same market (if provided)
        and are most recent. This is a simple recency-based approximation.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            if market:
                rows = conn.execute(
                    """
                    SELECT * FROM investigations
                    WHERE UPPER(COALESCE(market, '')) = UPPER(?)
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (market, n_results),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM investigations
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (n_results,),
                ).fetchall()
        finally:
            conn.close()

        return [dict(r) for r in rows]


