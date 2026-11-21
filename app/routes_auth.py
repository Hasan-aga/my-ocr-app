import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from app.auth import KIND_USER, Principal, authenticate_request
from app.config import Settings
from app.schemas import (
    LoginRequest,
    LoginResponse,
    MeResponse,
    PasswordChangeRequest,
    RegisterRequest,
    RegisterResponse,
    UserListResponse,
    UserOut,
    UserStatusResponse,
)
from app.tokens import TokenManager
from app.user_store import (
    ROLE_ADMIN,
    STATUS_APPROVED,
    STATUS_DISABLED,
    STATUS_PENDING,
    PasswordPolicy,
    User,
    UserError,
    UserStore,
)

logger = logging.getLogger("my_ocr_app")


def _to_out(user: User) -> UserOut:
    return UserOut.model_validate(user, from_attributes=True)


def get_principal(request: Request) -> Principal:
    settings: Settings = request.app.state.settings
    return authenticate_request(
        request,
        store=request.app.state.user_store,
        tokens=request.app.state.tokens,
        api_key=settings.api_key,
    )


def require_user(principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
    if principal.kind != KIND_USER or principal.user is None:
        raise HTTPException(status_code=403, detail="A user session is required")
    return principal


def require_admin(principal: Annotated[Principal, Depends(require_user)]) -> Principal:
    if not principal.is_admin:
        raise HTTPException(status_code=403, detail="Administrator access required")
    return principal


def create_accounts_router() -> APIRouter:
    router = APIRouter()

    def _store(request: Request) -> UserStore:
        return request.app.state.user_store

    def _tokens(request: Request) -> TokenManager:
        return request.app.state.tokens

    @router.post("/auth/register", response_model=RegisterResponse, status_code=201)
    async def register(request: Request, body: RegisterRequest) -> RegisterResponse:
        store = _store(request)
        try:
            PasswordPolicy(password=body.password)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        # With no admin configured or bootstrapped yet, the first account must
        # become the administrator or nobody could ever approve anyone.
        first_account = not store.has_admin() and store.count() == 0
        try:
            if first_account:
                user = store.create_user(
                    body.username, body.password, role=ROLE_ADMIN, status=STATUS_APPROVED
                )
            else:
                user = store.create_user(body.username, body.password)
        except UserError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if first_account:
            logger.warning(
                "Bootstrap: first account %r was granted administrator role", user.username
            )
            return RegisterResponse(
                status=STATUS_APPROVED,
                detail=(
                    "You are the first user: this account is an approved administrator. "
                    "Sign in now."
                ),
            )
        return RegisterResponse(
            status=STATUS_PENDING,
            detail="Account created. An administrator must approve it before you can sign in.",
        )

    @router.post("/auth/login", response_model=LoginResponse)
    async def login(request: Request, body: LoginRequest) -> LoginResponse:
        store = _store(request)
        user = store.verify_password(body.username, body.password)
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        if user.status == STATUS_PENDING:
            raise HTTPException(
                status_code=403,
                detail="Your account is awaiting administrator approval.",
            )
        if user.status == STATUS_DISABLED:
            raise HTTPException(
                status_code=403,
                detail="Your account has been disabled. Contact the administrator.",
            )
        token, expires_at = _tokens(request).issue(
            user_id=user.id, username=user.username, role=user.role
        )
        return LoginResponse(token=token, expires_at=expires_at, user=_to_out(user))

    @router.get("/auth/me", response_model=MeResponse)
    async def me(principal: Annotated[Principal, Depends(get_principal)]) -> MeResponse:
        return MeResponse(
            kind=principal.kind,
            user=_to_out(principal.user) if principal.user is not None else None,
        )

    @router.post("/auth/password", status_code=204)
    async def change_password(
        request: Request,
        body: PasswordChangeRequest,
        principal: Annotated[Principal, Depends(require_user)],
    ) -> None:
        assert principal.user is not None
        store = _store(request)
        if store.verify_password(principal.user.username, body.current_password) is None:
            raise HTTPException(status_code=401, detail="Current password is incorrect")
        try:
            PasswordPolicy(password=body.new_password)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        store.set_password(principal.user.id, body.new_password)

    @router.get("/admin/users", response_model=UserListResponse)
    async def list_users(
        request: Request,
        _: Annotated[Principal, Depends(require_admin)],
    ) -> UserListResponse:
        users = _store(request).list_users()
        return UserListResponse(users=[_to_out(user) for user in users])

    @router.post("/admin/users/{user_id}/approve", response_model=UserStatusResponse)
    async def approve_user(
        request: Request,
        user_id: int,
        _: Annotated[Principal, Depends(require_admin)],
    ) -> UserStatusResponse:
        return _set_status(request, user_id, STATUS_APPROVED)

    @router.post("/admin/users/{user_id}/disable", response_model=UserStatusResponse)
    async def disable_user(
        request: Request,
        user_id: int,
        principal: Annotated[Principal, Depends(require_admin)],
    ) -> UserStatusResponse:
        _guard_not_self(principal, user_id)
        return _set_status(request, user_id, STATUS_DISABLED)

    @router.delete("/admin/users/{user_id}", status_code=204)
    async def delete_user(
        request: Request,
        user_id: int,
        principal: Annotated[Principal, Depends(require_admin)],
    ) -> None:
        _guard_not_self(principal, user_id)
        try:
            _store(request).delete(user_id)
        except UserError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return router


def _guard_not_self(principal: Principal, user_id: int) -> None:
    assert principal.user is not None
    if principal.user.id == user_id:
        raise HTTPException(status_code=400, detail="You cannot modify your own account here")


def _set_status(request: Request, user_id: int, status: str) -> UserStatusResponse:
    try:
        user = request.app.state.user_store.set_status(user_id, status)
    except UserError as exc:
        detail = str(exc)
        raise HTTPException(
            status_code=404 if "not found" in detail else 400, detail=detail
        ) from exc
    return UserStatusResponse(user=_to_out(user))
