import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from app.auth import Principal
from app.config import Settings
from app.engines.base import EngineError, EngineUsage, OcrEngine, ParsedLine
from app.engines.registry import EngineRegistry, EngineRegistryError, create_registry
from app.images import (
    ALLOWED_CONTENT_TYPES,
    ImageTooLargeError,
    ImageValidationError,
    PreparedImage,
    UnsupportedImageTypeError,
    prepare_image,
)
from app.routes_auth import create_accounts_router, get_principal
from app.schemas import (
    HealthResponse,
    ImageInfo,
    ModelsResponse,
    OcrLine,
    OcrResponse,
)
from app.security import (
    SlidingWindowLimiter,
    client_key,
    rate_limited_response,
)
from app.tokens import TokenManager
from app.user_store import UserStore

logger = logging.getLogger("my_ocr_app")

STATIC_DIR = Path(__file__).resolve().parent / "static"
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

# Top-level path segments owned by the API; everything else falls through to
# the single-page application.
API_SEGMENTS = {"ocr", "models", "auth", "admin", "healthz", "docs", "redoc", "openapi.json"}


def resolve_static_dir() -> Path | None:
    """Prefer a built SPA; Docker images ship it at app/static."""
    for candidate in (STATIC_DIR, FRONTEND_DIST):
        if (candidate / "index.html").is_file():
            return candidate
    return None


async def read_upload(file: UploadFile, max_bytes: int) -> bytes:
    if file.size is not None and file.size > max_bytes:
        raise ImageTooLargeError("Upload size limit exceeded")

    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > max_bytes:
            raise ImageTooLargeError("Upload size limit exceeded")
        chunks.append(chunk)

    if total == 0:
        raise ImageValidationError("Empty image")
    return b"".join(chunks)


def _run_pipeline(
    data: bytes,
    settings: Settings,
    engine: OcrEngine,
) -> tuple[list[ParsedLine], PreparedImage, float, EngineUsage | None]:
    started = time.perf_counter()
    prepared = prepare_image(data, settings)
    try:
        result = engine.recognize(prepared.path)
    finally:
        prepared.path.unlink(missing_ok=True)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return result.lines, prepared, elapsed_ms, result.usage


def create_app(
    settings: Settings | None = None,
    registry: EngineRegistry | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active_registry = registry or await asyncio.to_thread(create_registry, resolved_settings)
        app.state.settings = resolved_settings
        app.state.registry = active_registry
        app.state.semaphore = asyncio.Semaphore(1)
        app.state.ocr_limiter = SlidingWindowLimiter(resolved_settings.rate_limit_ocr_per_minute)
        app.state.models_limiter = SlidingWindowLimiter(
            resolved_settings.rate_limit_models_per_minute
        )
        app.state.login_limiter = SlidingWindowLimiter(
            resolved_settings.rate_limit_login_per_minute
        )
        app.state.register_limiter = SlidingWindowLimiter(
            resolved_settings.rate_limit_register_per_minute
        )

        user_store = UserStore(resolved_settings.data_path / "users.db")
        username = resolved_settings.admin_username
        password = resolved_settings.admin_password
        if (username is None) != (password is None):
            raise RuntimeError(
                "OCR_ADMIN_USERNAME and OCR_ADMIN_PASSWORD must be set together"
            )
        if username and password:
            created = user_store.ensure_admin(username, password)
            if created is not None:
                logger.info("Seeded administrator account %r", username)
        app.state.user_store = user_store
        app.state.tokens = TokenManager(
            secret=resolved_settings.jwt_secret,
            data_dir=resolved_settings.data_path,
            expire_hours=resolved_settings.jwt_expire_hours,
        )
        if not user_store.has_admin():
            logger.warning(
                "No administrator exists yet: the first account registered via "
                "/auth/register will become the admin"
            )

        try:
            if resolved_settings.warmup and registry is None:
                for engine in active_registry.ids():
                    selected = active_registry.get(engine)
                    if selected.is_local:
                        await asyncio.to_thread(selected.warmup)
                        break
            logger.info(
                "OCR ready; default model=%s (%s)",
                active_registry.default_id,
                active_registry.default().provider,
            )
            yield
        finally:
            active_registry.close()

    fastapi_kwargs = {
        "title": "My OCR App",
        "version": "0.1.0",
        "lifespan": lifespan,
    }
    if not resolved_settings.docs_enabled:
        fastapi_kwargs.update({"docs_url": None, "redoc_url": None, "openapi_url": None})

    app = FastAPI(**fastapi_kwargs)

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        path = request.url.path
        limiter = None
        if path == "/ocr":
            limiter = request.app.state.ocr_limiter
        elif path == "/models":
            limiter = request.app.state.models_limiter
        elif path == "/auth/login" and request.method == "POST":
            limiter = request.app.state.login_limiter
        elif path == "/auth/register" and request.method == "POST":
            limiter = request.app.state.register_limiter
        if limiter is not None:
            retry_after = limiter.retry_after(client_key(request))
            if retry_after:
                return rate_limited_response(retry_after)
        return await call_next(request)

    app.include_router(create_accounts_router())

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz(request: Request) -> HealthResponse:
        return HealthResponse(
            status="ok",
            ready=True,
            model=request.app.state.registry.default_model_info(),
        )

    @app.get("/models", response_model=ModelsResponse)
    async def models(
        request: Request,
        principal: Annotated[Principal, Depends(get_principal)],
    ) -> ModelsResponse:
        return ModelsResponse(models=request.app.state.registry.catalog())

    @app.post("/ocr", response_model=OcrResponse)
    async def ocr(
        request: Request,
        file: Annotated[UploadFile, File()],
        principal: Annotated[Principal, Depends(get_principal)],
        model: Annotated[str | None, Query()] = None,
        min_confidence: Annotated[float, Query(ge=0, le=1)] = 0.0,
    ) -> OcrResponse:
        active_settings: Settings = request.app.state.settings
        registry: EngineRegistry = request.app.state.registry

        content_type = (file.content_type or "").lower()
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=415, detail="Unsupported image content type")

        try:
            data = await read_upload(file, active_settings.max_upload_bytes)
        except ImageTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except ImageValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            active_engine = registry.get(model) if model else registry.default()
        except EngineRegistryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        semaphore: asyncio.Semaphore = request.app.state.semaphore
        acquired = False
        try:
            if active_engine.is_local:
                try:
                    await asyncio.wait_for(
                        semaphore.acquire(),
                        timeout=active_settings.busy_timeout_seconds,
                    )
                    acquired = True
                except TimeoutError as exc:
                    raise HTTPException(status_code=429, detail="OCR engine is busy") from exc

            try:
                lines, prepared, elapsed_ms, usage = await asyncio.to_thread(
                    _run_pipeline,
                    data,
                    active_settings,
                    active_engine,
                )
            except UnsupportedImageTypeError as exc:
                raise HTTPException(status_code=415, detail=str(exc)) from exc
            except ImageTooLargeError as exc:
                raise HTTPException(status_code=413, detail=str(exc)) from exc
            except ImageValidationError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except EngineError as exc:
                logger.exception("OCR engine failed")
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            except Exception as exc:
                logger.exception("OCR pipeline failed")
                raise HTTPException(status_code=500, detail="OCR processing failed") from exc

            filtered = [line for line in lines if line.confidence >= min_confidence]
            return OcrResponse(
                text="\n".join(line.text for line in filtered),
                lines=[
                    OcrLine(text=line.text, confidence=line.confidence, box=line.box)
                    for line in filtered
                ],
                line_count=len(filtered),
                image=ImageInfo(
                    width=prepared.width,
                    height=prepared.height,
                    original_width=prepared.original_width,
                    original_height=prepared.original_height,
                    format=prepared.format,
                    content_type=content_type,
                ),
                model=active_engine.model_info(),
                usage=usage.as_schema() if usage is not None else None,
                processing_time_ms=round(elapsed_ms, 2),
            )
        finally:
            if acquired:
                semaphore.release()

    _mount_frontend(app)

    return app


def _mount_frontend(app: FastAPI) -> None:
    """Serve the built SPA. Resolution is lazy so dev builds are picked up."""

    def _file_or_none(static_dir: Path, rel: str) -> FileResponse | None:
        candidate = (static_dir / rel).resolve()
        if candidate.is_file() and static_dir.resolve() in candidate.parents:
            return FileResponse(candidate)
        return None

    @app.get("/")
    async def spa_root() -> Response:
        static_dir = resolve_static_dir()
        if static_dir is None:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "Web UI is not built. Run 'npm install && npm run build' "
                    "in frontend/ or use the Docker image which bundles the UI."
                },
            )
        return FileResponse(static_dir / "index.html", media_type="text/html")

    @app.get("/{full_path:path}")
    async def spa(full_path: str) -> Response:
        static_dir = resolve_static_dir()
        if static_dir is None:
            raise HTTPException(status_code=404, detail="Not found")
        top = full_path.split("/", 1)[0]
        if top in API_SEGMENTS:
            raise HTTPException(status_code=404, detail="Not found")
        served = _file_or_none(static_dir, full_path)
        if served is not None:
            return served
        # Client-side route: serve the app shell.
        return FileResponse(static_dir / "index.html", media_type="text/html")
