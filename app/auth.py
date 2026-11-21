import secrets
from dataclasses import dataclass

from fastapi import HTTPException, Request

from app.tokens import TokenManager
from app.user_store import ROLE_ADMIN, STATUS_APPROVED, User, UserStore

KIND_USER = "user"
KIND_API_KEY = "api_key"


@dataclass(frozen=True)
class Principal:
    """The authenticated identity of a request."""

    kind: str
    user: User | None = None

    @property
    def is_admin(self) -> bool:
        return self.user is not None and self.user.role == ROLE_ADMIN


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=401,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _bearer_token(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def authenticate_request(
    request: Request,
    *,
    store: UserStore,
    tokens: TokenManager,
    api_key: str,
) -> Principal:
    """Accept a user session token or the legacy API key; reject anything else."""
    token = _bearer_token(request)
    if token is None:
        raise _unauthorized("Missing bearer token")

    try:
        user_id, _, _ = tokens.verify(token)
    except Exception:
        # Not a session token: fall through to the static API key check.
        if secrets.compare_digest(token, api_key):
            return Principal(kind=KIND_API_KEY)
        raise _unauthorized("Invalid credentials") from None

    user = store.get(user_id)
    if user is None:
        raise _unauthorized("Account no longer exists")
    if user.status != STATUS_APPROVED:
        raise HTTPException(status_code=403, detail=f"Account is {user.status}")
    return Principal(kind=KIND_USER, user=user)
