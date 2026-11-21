import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from pydantic import BaseModel, Field, field_validator

USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{2,31}$")
MIN_PASSWORD_LENGTH = 10

ROLE_USER = "user"
ROLE_ADMIN = "admin"
STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_DISABLED = "disabled"


class UserError(Exception):
    """Raised on invalid user operations (duplicate name, unknown id, ...)."""


class User(BaseModel):
    id: int
    username: str
    role: str
    status: str
    created_at: str
    approved_at: str | None = None
    last_login_at: str | None = None


class PasswordPolicy(BaseModel):
    """Validated on registration and password change."""

    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=256)

    @field_validator("password")
    @classmethod
    def password_not_trivial(cls, value: str) -> str:
        lowered = value.lower()
        if lowered in {"changeme123", "password123", "1234567890"}:
            raise ValueError("Password is too common")
        return value


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class UserStore:
    """SQLite-backed user records with Argon2-hashed passwords."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._hasher = PasswordHasher()
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user'
                        CHECK (role IN ('user', 'admin')),
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'approved', 'disabled')),
                    created_at TEXT NOT NULL,
                    approved_at TEXT,
                    last_login_at TEXT
                )
                """
            )
            # Drop the write-ahead sidecar files back into the main database
            # so a container restart with a mounted volume stays consistent.
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> User:
        return User(
            id=row["id"],
            username=row["username"],
            role=row["role"],
            status=row["status"],
            created_at=row["created_at"],
            approved_at=row["approved_at"],
            last_login_at=row["last_login_at"],
        )

    def validate_username(self, username: str) -> str:
        username = username.strip()
        if not USERNAME_PATTERN.match(username):
            raise UserError(
                "Username must be 3-32 characters: letters, digits, dot, dash, underscore"
            )
        return username

    def create_user(
        self,
        username: str,
        password: str,
        *,
        role: str = ROLE_USER,
        status: str = STATUS_PENDING,
    ) -> User:
        username = self.validate_username(username)
        hash_ = self._hasher.hash(password)
        now = _now()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM users WHERE username = ?", (username,)
            ).fetchone()
            if existing is not None:
                raise UserError("Username is already taken")
            cur = conn.execute(
                """
                INSERT INTO users (username, password_hash, role, status, created_at, approved_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (username, hash_, role, status, now, now if status == STATUS_APPROVED else None),
            )
            row = conn.execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone()
        assert row is not None
        return self._row_to_user(row)

    def get(self, user_id: int) -> User | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return self._row_to_user(row) if row is not None else None

    def get_by_username(self, username: str) -> User | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username.strip(),)
            ).fetchone()
        return self._row_to_user(row) if row is not None else None

    def verify_password(self, username: str, password: str) -> User | None:
        """Return the user when credentials are valid, else None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username.strip(),)
            ).fetchone()
        if row is None:
            # Constant-ish work for unknown users too, to limit timing oracles.
            self._hasher.hash("invalid-password")
            return None
        try:
            self._hasher.verify(row["password_hash"], password)
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return None
        user = self._row_to_user(row)
        self._maybe_rehash(row["id"], row["password_hash"], password)
        self._touch_login(user.id)
        return user

    def _maybe_rehash(self, user_id: int, current_hash: str, password: str) -> None:
        if not self._hasher.check_needs_rehash(current_hash):
            return
        new_hash = self._hasher.hash(password)
        with self._connect() as conn:
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user_id)
            )

    def _touch_login(self, user_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE users SET last_login_at = ? WHERE id = ?", (_now(), user_id)
            )

    def set_password(self, user_id: int, password: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (self._hasher.hash(password), user_id),
            )

    def list_users(self) -> list[User]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
        return [self._row_to_user(row) for row in rows]

    def set_status(self, user_id: int, status: str) -> User:
        if status not in {STATUS_PENDING, STATUS_APPROVED, STATUS_DISABLED}:
            raise UserError(f"Invalid status {status!r}")
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if row is None:
                raise UserError("User not found")
            approved_at = _now() if status == STATUS_APPROVED else None
            conn.execute(
                "UPDATE users SET status = ?, approved_at = ? WHERE id = ?",
                (status, approved_at, user_id),
            )
            updated = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        assert updated is not None
        return self._row_to_user(updated)

    def delete(self, user_id: int) -> None:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        if cur.rowcount == 0:
            raise UserError("User not found")

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def has_admin(self) -> bool:
        with self._connect() as conn:
            return (
                conn.execute(
                    "SELECT 1 FROM users WHERE role = 'admin' AND status = 'approved' LIMIT 1"
                ).fetchone()
                is not None
            )

    def ensure_admin(self, username: str, password: str) -> User | None:
        """Create the seeded admin once. Returns the user when newly created."""
        existing = self.get_by_username(username)
        if existing is not None:
            if existing.role != ROLE_ADMIN:
                raise UserError(
                    f"Seeded admin username {username!r} is already taken by a non-admin user"
                )
            if existing.status != STATUS_APPROVED:
                return self.set_status(existing.id, STATUS_APPROVED)
            return None
        return self.create_user(username, password, role=ROLE_ADMIN, status=STATUS_APPROVED)
