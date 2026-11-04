import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from starlette.middleware.base import BaseHTTPMiddleware

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
from app.logging_config import configure_logging, request_id_var
from app.schemas import (
    HealthResponse,
    ImageInfo,
    ModelsResponse,
    OcrLine,
    OcrResponse,
)

configure_logging()
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


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Assigns a request id, logs every request with status and duration."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                "request failed method=%s path=%s duration_ms=%.1f",
                request.method,
                request.url.path,
                elapsed_ms,
            )
            request_id_var.reset(token)
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request method=%s path=%s status=%d duration_ms=%.1f client=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request.client.host if request.client else "unknown",
        )
        request_id_var.reset(token)
        return response


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
    app.add_middleware(RequestLoggingMiddleware)

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
        catalog = request.app.state.registry.catalog()
        logger.info(
            "model catalog served count=%d providers=%s",
            len(catalog),
            ",".join(sorted({m.provider for m in catalog})) or "none",
        )
        return ModelsResponse(models=catalog)

    @app.post("/ocr", response_model=OcrResponse)
    async def ocr(
        request: Request,
        file: Annotated[UploadFile, File()],
        model: Annotated[str | None, Query()] = None,
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
            logger.info(
                "model selection failed requested=%r available=%s",
                model,
                ",".join(registry.ids()),
            )
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

            logger.info(
                "ocr started engine=%s provider=%s content_type=%s bytes=%d",
                active_engine.id,
                active_engine.provider,
                content_type,
                len(data),
            )
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

    return app