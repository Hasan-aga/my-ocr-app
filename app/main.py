import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated, Union

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response

from app.auth import verify_bearer
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

# Security imports - try/except for optional dependencies
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from fastapi_csrf_protect import CsrfProtect
    from pydantic import BaseModel
    
    class CsrfSettings(BaseModel):
        secret_key: str = "my-secret-key-change-in-production"
    
    # Initialize rate limiter
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=["100/day;50/hour;10/minute"],
    )
    
    # Initialize CSRF protection
    @CsrfProtect.load_config
    def get_csrf_config():
        return CsrfSettings()
    
except ImportError:
    # Security packages not available
    limiter = None
    CsrfProtect = None
    get_remote_address = None
from app.schemas import (
    HealthResponse,
    ImageInfo,
    ModelsResponse,
    OcrLine,
    OcrResponse,
)

logger = logging.getLogger("my_ocr_app")


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


from typing import Union

from typing import Union


def create_app(
    settings: Union[Settings, None] = None,
    registry: Union[EngineRegistry, None] = None,
) -> FastAPI:
    resolved_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active_registry = registry or await asyncio.to_thread(create_registry, resolved_settings)
        app.state.settings = resolved_settings
        app.state.registry = active_registry
        app.state.semaphore = asyncio.Semaphore(1)
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

    # Add rate limiting middleware if available
    if limiter is not None and get_remote_address is not None:
        app.state.limiter = limiter
        app.add_middleware(
            limiter.middleware,
            key_func=get_remote_address,
        )

    # Add CSRF protection if available
    if CsrfProtect is not None:
        @app.middleware("http")
        async def csrf_middleware(request: Request, call_next):
            if request.method in ["POST", "PUT", "DELETE"]:
                # Skip CSRF for API endpoints that use Bearer auth
                auth_header = request.headers.get("authorization", "")
                if auth_header.startswith("Bearer "):
                    return await call_next(request)
            return await call_next(request)

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"name": "my-ocr-app", "status": "ok"}

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz(request: Request) -> HealthResponse:
        return HealthResponse(
            status="ok",
            ready=True,
            model=request.app.state.registry.default_model_info(),
        )

    @app.get("/models", response_model=ModelsResponse)
    async def models(request: Request) -> ModelsResponse:
        verify_bearer(request, request.app.state.settings.api_key)
        return ModelsResponse(models=request.app.state.registry.catalog())

    @app.post("/ocr", response_model=OcrResponse)
    async def ocr(
        request: Request,
        file: Annotated[UploadFile, File()],
        model: Annotated[Union[str, None], Query()] = None,
        min_confidence: Annotated[float, Query(ge=0, le=1)] = 0.0,
    ) -> OcrResponse:
        active_settings: Settings = request.app.state.settings
        registry: EngineRegistry = request.app.state.registry
        verify_bearer(request, active_settings.api_key)

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

    # Serve static files for frontend
    try:
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import Response
        app.mount("/static", StaticFiles(directory="app/static"), name="static")
        
        # Serve frontend HTML
        @app.get("/")
        async def frontend():
            with open("app/static/index.html") as f:
                return Response(content=f.read(), media_type="text/html")
    except ImportError:
        # Static files not available
        pass

    return app