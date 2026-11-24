import sqlite3
import time
from pathlib import Path


class Memory:
    def __init__(self, db_path: Path):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS facts(
                ns TEXT, k TEXT, v TEXT, ts REAL, PRIMARY KEY(ns, k))"""
        )
        self.db.commit()

    def remember(self, key: str, value: str, ns: str = "global") -> str:
        self.db.execute(
            "INSERT OR REPLACE INTO facts VALUES (?,?,?,?)",
            (ns, key, value, time.time()),
        )
        self.db.commit()
        return f"remembered [{ns}] {key}"

    def recall(self, key: str | None = None, ns: str = "global") -> str:
        if key:
            row = self.db.execute(
                "SELECT v FROM facts WHERE ns=? AND k=?", (ns, key)
            ).fetchone()
            return row[0] if row else f"no fact '{key}'"
        rows = self.db.execute(
            "SELECT k, v FROM facts WHERE ns=? ORDER BY ts DESC", (ns,)
        ).fetchall()
        return "\n".join(f"{k}: {v}" for k, v in rows) or "(empty)"