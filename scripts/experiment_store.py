"""Small opt-in SQLite store for local recommendation experiment events."""
import hashlib
import hmac
import json
import sqlite3
import uuid
from pathlib import Path


class ExperimentStore:
    def __init__(self, path: Path, secret: str):
        if len(secret.encode("utf-8")) < 16:
            raise ValueError("Experiment secret must contain at least 16 UTF-8 bytes.")
        self.path = Path(path)
        self.secret = secret.encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS impressions (
                    id TEXT PRIMARY KEY, participant_hash TEXT NOT NULL,
                    variant TEXT NOT NULL CHECK(variant IN ('A','B')),
                    actual_strategy TEXT NOT NULL, item_ids TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS clicks (
                    impression_id TEXT NOT NULL REFERENCES impressions(id),
                    item_id INTEGER NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (impression_id, item_id)
                );
            """)

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=5)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def assignment(self, participant: str) -> tuple[str, str]:
        if not participant or len(participant) > 128:
            raise ValueError("Participant ID must contain 1 to 128 characters.")
        digest = hmac.new(self.secret, participant.encode("utf-8"), hashlib.sha256).hexdigest()
        return digest, "A" if int(digest[:8], 16) % 2 == 0 else "B"

    def record_impression(self, participant: str, variant: str, strategy: str, item_ids: list[int]) -> str:
        participant_hash, expected = self.assignment(participant)
        if variant != expected:
            raise ValueError("Variant does not match participant assignment.")
        if not strategy or not item_ids or len(item_ids) > 100 or len(set(item_ids)) != len(item_ids):
            raise ValueError("An impression needs a strategy and 1 to 100 unique items.")
        if any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in item_ids):
            raise ValueError("Impression item IDs must be positive integers.")
        impression_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute("INSERT INTO impressions (id, participant_hash, variant, actual_strategy, item_ids) VALUES (?, ?, ?, ?, ?)",
                         (impression_id, participant_hash, variant, strategy, json.dumps(item_ids)))
        return impression_id

    def record_click(self, impression_id: str, item_id: int) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT item_ids FROM impressions WHERE id = ?", (impression_id,)).fetchone()
            if row is None or item_id not in json.loads(row[0]):
                raise ValueError("Click must refer to a displayed item in an existing impression.")
            cursor = conn.execute("INSERT OR IGNORE INTO clicks (impression_id, item_id) VALUES (?, ?)",
                                  (impression_id, item_id))
            return cursor.rowcount == 1

    def summary(self) -> dict:
        with self._connect() as conn:
            rows = conn.execute("SELECT id, participant_hash, variant, actual_strategy, item_ids FROM impressions").fetchall()
            clicks = conn.execute("SELECT impression_id, COUNT(*) FROM clicks GROUP BY impression_id").fetchall()
        by_impression = dict(clicks)
        result = {}
        for variant in ("A", "B"):
            group = [row for row in rows if row[2] == variant]
            displayed = sum(len(json.loads(row[4])) for row in group)
            clicked = sum(by_impression.get(row[0], 0) for row in group)
            result[variant] = {"participants": len({row[1] for row in group}), "impressions": len(group),
                               "displayed_items": displayed, "clicked_items": clicked,
                               "item_ctr": clicked / displayed if displayed else None,
                               "actual_strategies": {s: sum(row[3] == s for row in group)
                                                     for s in sorted({row[3] for row in group})}}
        return result
