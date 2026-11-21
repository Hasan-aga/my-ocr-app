import logging
import secrets
import time
from pathlib import Path

import jwt

logger = logging.getLogger("my_ocr_app")

TOKEN_TYPE = "access"


class TokenError(Exception):
    """Raised when a token is missing, malformed, expired, or forged."""


class TokenManager:
    """Issues and verifies HS256 session tokens.

    The signing secret comes from configuration; when unset it is generated
    once and persisted in the data directory so sessions survive restarts.
    """

    def __init__(self, secret: str | None, data_dir: Path, expire_hours: int) -> None:
        self._secret = secret or self._load_or_create_secret(data_dir)
        self._expire_seconds = expire_hours * 3600

    @staticmethod
    def _load_or_create_secret(data_dir: Path) -> str:
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / "jwt.secret"
        if path.exists():
            secret = path.read_text(encoding="utf-8").strip()
            if secret:
                return secret
        secret = secrets.token_hex(32)
        path.write_text(secret, encoding="utf-8")
        path.chmod(0o600)
        logger.info("Generated session signing secret at %s", path)
        return secret

    def issue(self, *, user_id: int, username: str, role: str) -> tuple[str, int]:
        now = int(time.time())
        exp = now + self._expire_seconds
        token = jwt.encode(
            {
                "sub": str(user_id),
                "usr": username,
                "rol": role,
                "typ": TOKEN_TYPE,
                "iat": now,
                "exp": exp,
            },
            self._secret,
            algorithm="HS256",
        )
        return token, exp

    def verify(self, token: str) -> tuple[int, str, str]:
        """Return (user_id, username, role) for a valid token or raise TokenError."""
        try:
            claims = jwt.decode(token, self._secret, algorithms=["HS256"])
        except jwt.PyJWTError:
            raise TokenError("Invalid or expired session") from None
        if claims.get("typ") != TOKEN_TYPE:
            raise TokenError("Invalid token type")
        try:
            user_id = int(claims["sub"])
        except (KeyError, TypeError, ValueError):
            raise TokenError("Invalid token subject") from None
        return user_id, str(claims.get("usr", "")), str(claims.get("rol", ""))
