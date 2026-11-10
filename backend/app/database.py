"""SQLite database layer with investigation-level isolation and composite foreign keys."""

import glob
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from .config import DATABASE_PATH, DATABASE_URL
from .logging import logger

# Directory containing versioned migration SQL files
MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "db" / "migrations"


def get_connection() -> sqlite3.Connection:
    """Create a new SQLite connection with foreign keys enforced and WAL mode."""
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


@contextmanager
def get_transaction():
    """Context manager for a database transaction with automatic commit/rollback."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _migration_version(filename: str) -> int:
    """Extract the version number from a migration filename like V001__name.sql."""
    match = re.match(r"V(\d+)__.*\.sql$", filename)
    if match:
        return int(match.group(1))
    return 0


def _get_applied_versions(conn: sqlite3.Connection) -> set:
    """Return the set of migration versions already applied."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    ).fetchone()
    if row is None:
        return set()
    rows = conn.execute("SELECT version FROM schema_version").fetchall()
    return {r[0] for r in rows}


def _apply_migration(conn: sqlite3.Connection, path: Path) -> None:
    """Apply a single migration file within a transaction."""
    sql = path.read_text(encoding="utf-8")
    conn.executescript(sql)
    # Extract version and record it
    match = re.match(r"V(\d+)__.*\.sql$", path.name)
    if match:
        version = int(match.group(1))
        conn.execute(
            "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (?, datetime('now'))",
            (version,),
        )


def init_database() -> None:
    """Initialize the database and apply all pending migrations in order."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        # Ensure schema_version table exists (created by V001)
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        ).fetchone()

    if row is None:
        # Fresh database: apply V001 first
        v001_path = MIGRATIONS_DIR / "V001__initial_schema.sql"
        if not v001_path.exists():
            logger.error("Initial schema not found: %s", v001_path)
            raise FileNotFoundError(f"Initial schema not found: {v001_path}")
        with get_connection() as conn:
            _apply_migration(conn, v001_path)
        logger.info("Database initialized with V001 initial schema")

    # Apply any pending migrations in version order
    migration_files = sorted(
        [f for f in os.listdir(MIGRATIONS_DIR) if f.endswith(".sql")],
        key=_migration_version,
    )

    applied_versions: set = set()
    with get_connection() as conn:
        applied_versions = _get_applied_versions(conn)

    for migration_file in migration_files:
        version = _migration_version(migration_file)
        if version in applied_versions:
            continue
        migration_path = MIGRATIONS_DIR / migration_file
        logger.info("Applying migration %s", migration_file)
        with get_connection() as conn:
            _apply_migration(conn, migration_path)
        logger.info("Migration %s applied successfully", migration_file)

    # Verify foreign key integrity
    with get_connection() as conn:
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            logger.error("Foreign key violations after migrations: %s", violations)
            raise RuntimeError(f"Foreign key violations: {violations}")
        logger.info("Foreign key check passed: zero violations")


def reset_database() -> None:
    """Delete and recreate the database (for testing only)."""
    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()
    # Also remove WAL and SHM files
    for suffix in ("-wal", "-shm"):
        p = Path(str(DATABASE_PATH) + suffix)
        if p.exists():
            p.unlink()
    init_database()


__all__ = ["get_connection", "get_transaction", "init_database", "reset_database", "DATABASE_PATH"]
