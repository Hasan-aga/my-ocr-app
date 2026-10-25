import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile

from app.auth import verify_bearer
from app.config import Settings
from app.images import (
    ALLOWED_CONTENT_TYPES,
    ImageTooLargeError,
    ImageValidationError,
    PreparedImage,
    UnsupportedImageTypeError,
    prepare_image,
)
from app.ocr import OcrEngine, OcrEngineProtocol, ParsedLine
from app.schemas import HealthResponse, ImageInfo, OcrLine, OcrResponse

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
    engine: OcrEngineProtocol,
) -> tuple[list[ParsedLine], PreparedImage, float]:
    started = time.perf_counter()
    prepared = prepare_image(data, settings)
    try:
        lines = engine.recognize(prepared.path)
    finally:
        prepared.path.unlink(missing_ok=True)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return lines, prepared, elapsed_ms


def create_app(
    settings: Settings | None = None,
    engine: OcrEngineProtocol | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = resolved_settings
        app.state.engine = engine or await asyncio.to_thread(OcrEngine, resolved_settings)
        app.state.semaphore = asyncio.Semaphore(1)
        if resolved_settings.warmup and engine is None:
            await asyncio.to_thread(app.state.engine.warmup)
        logger.info(
            "OCR ready with %s / %s",
            app.state.engine.det_model,
            app.state.engine.rec_model,
        )
        yield

    fastapi_kwargs = {
        "title": "My OCR App",
        "version": "0.1.0",
        "lifespan": lifespan,
    }
    if not resolved_settings.docs_enabled:
        fastapi_kwargs.update({"docs_url": None, "redoc_url": None, "openapi_url": None})

    app = FastAPI(**fastapi_kwargs)

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"name": "my-ocr-app", "status": "ok"}

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz(request: Request) -> HealthResponse:
        return HealthResponse(
            status="ok",
            ready=True,
            model=request.app.state.engine.model_info(),
        )

    @app.post("/ocr", response_model=OcrResponse)
    async def ocr(
        request: Request,
        file: Annotated[UploadFile, File()],
        min_confidence: Annotated[float, Query(ge=0, le=1)] = 0.0,
    ) -> OcrResponse:
        active_settings: Settings = request.app.state.settings
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

        semaphore: asyncio.Semaphore = request.app.state.semaphore
        acquired = False
        try:
            try:
                await asyncio.wait_for(
                    semaphore.acquire(),
                    timeout=active_settings.busy_timeout_seconds,
                )
                acquired = True
            except TimeoutError as exc:
                raise HTTPException(status_code=429, detail="OCR engine is busy") from exc

            try:
                lines, prepared, elapsed_ms = await asyncio.to_thread(
                    _run_pipeline,
                    data,
                    active_settings,
                    request.app.state.engine,
                )
            except UnsupportedImageTypeError as exc:
                raise HTTPException(status_code=415, detail=str(exc)) from exc
            except ImageTooLargeError as exc:
                raise HTTPException(status_code=413, detail=str(exc)) from exc
            except ImageValidationError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
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
                model=request.app.state.engine.model_info(),
                processing_time_ms=round(elapsed_ms, 2),
            )
        finally:
            if acquired:
                semaphore.release()

    return app
