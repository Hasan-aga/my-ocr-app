import logging

import httpx

from app.config import Settings
from app.engines.base import EngineError, OcrEngine
from app.engines.paddle import PADDLE_ID, PaddleEngine
from app.engines.qwen import QwenEngine, QwenProvider
from app.engines.qwen_catalog import catalog_entry
from app.logging_config import redact_secret
from app.schemas import CatalogModel, ModelInfo, Pricing

logger = logging.getLogger("my_ocr_app")


class EngineRegistryError(EngineError):
    """Raised when the engine registry cannot be built or resolved."""


class EngineRegistry:
    def __init__(
        self,
        engines: dict[str, OcrEngine],
        default_id: str,
        qwen: QwenProvider | None = None,
        qwen_ids: list[str] | None = None,
    ) -> None:
        self._engines = engines
        self._default_id = default_id
        self._qwen = qwen
        self._qwen_ids = list(qwen_ids or [])
        self._qwen_cache: dict[str, QwenEngine] = {}

    @property
    def default_id(self) -> str:
        return self._default_id

    def ids(self) -> list[str]:
        return list(self._engines) + list(self._qwen_ids)

    def default(self) -> OcrEngine:
        return self.get(self._default_id)

    def get(self, model_id: str) -> OcrEngine:
        if model_id in self._engines:
            return self._engines[model_id]
        if self._qwen is not None:
            if model_id not in self._qwen_cache:
                self._qwen_cache[model_id] = self._qwen.build(model_id)
            return self._qwen_cache[model_id]
        available = ", ".join(self.ids()) or "none"
        raise EngineRegistryError(
            f"Unknown model {model_id!r}; available: {available}"
        ) from None

    def default_model_info(self) -> ModelInfo:
        return self.default().model_info()

    def catalog(self) -> list[CatalogModel]:
        models: list[CatalogModel] = []
        for ident in self.ids():
            if ident in self._engines:
                engine = self._engines[ident]
                provider = engine.provider
                label = engine.label
            else:
                provider = "qwen"
                label = None
            entry = catalog_entry(ident)
            if entry is not None:
                label = entry.label
            pricing = Pricing()
            if entry is not None:
                pricing = Pricing(
                    input_usd_per_1m=entry.input_usd_per_1m,
                    output_usd_per_1m=entry.output_usd_per_1m,
                )
            models.append(
                CatalogModel(
                    id=ident,
                    provider=provider,
                    label=label,
                    note=entry.note if entry is not None else None,
                    pricing=pricing,
                )
            )
        return models

    def close(self) -> None:
        for engine in self._engines.values():
            closer = getattr(engine, "close", None)
            if callable(closer):
                closer()
        for engine in self._qwen_cache.values():
            engine.close()


def create_registry(
    settings: Settings,
    qwen_transport: httpx.BaseTransport | None = None,
) -> EngineRegistry:
    engines: dict[str, OcrEngine] = {}

    if settings.enable_paddle:
        engines[PADDLE_ID] = PaddleEngine(settings)

    qwen: QwenProvider | None = None
    qwen_ids: list[str] = []
    if settings.qwen_api_key:
        logger.info(
            "qwen configured base_url=%s api_key=%s strict_models=%s",
            settings.qwen_base_url,
            redact_secret(settings.qwen_api_key),
            bool(settings.qwen_model_ids),
        )
        qwen = QwenProvider(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            timeout_seconds=settings.qwen_timeout_seconds,
            max_retries=settings.qwen_max_retries,
            transport=qwen_transport,
        )
        if settings.qwen_model_ids:
            qwen_ids = settings.qwen_model_ids
            logger.info("Qwen models from OCR_QWEN_MODELS: %s", ", ".join(qwen_ids))
        else:
            try:
                qwen_ids = qwen.discover()
            except EngineError as exc:
                logger.warning(
                    "Qwen model discovery failed (continuing with PaddleOCR): %s. "
                    "Check that OCR_QWEN_API_KEY is valid for OCR_QWEN_BASE_URL "
                    "(Singapore/international keys need dashscope-intl.aliyuncs.com)",
                    exc,
                )
                qwen_ids = []
            logger.info("Qwen provider ready with %d model(s)", len(qwen_ids))
    elif not settings.enable_paddle and settings.qwen_models:
        logger.warning(
            "Qwen cloud models configured but OCR_QWEN_API_KEY is not set; cloud engines disabled"
        )
    else:
        logger.info("qwen not configured (OCR_QWEN_API_KEY unset); using local engines only")

    known_ids = list(engines) + list(qwen_ids)
    if not known_ids:
        raise EngineRegistryError(
            "No OCR engines configured: enable PaddleOCR or set OCR_QWEN_API_KEY"
        )

    default_id = settings.default_model
    if default_id is None:
        default_id = PADDLE_ID if PADDLE_ID in engines else qwen_ids[0]
    if default_id is None or default_id not in known_ids:
        raise EngineRegistryError(
            f"Default model {default_id!r} is not available; "
            f"available: {', '.join(known_ids) or 'none'}"
        )

    registry = EngineRegistry(engines, default_id, qwen=qwen, qwen_ids=qwen_ids)
    logger.info(
        "OCR registry ready with %d engine(s); default=%s",
        len(known_ids),
        registry.default_id,
    )
    providers = {ident: "paddle" for ident in engines}
    providers.update({ident: "qwen" for ident in qwen_ids})
    for ident in registry.ids():
        logger.info("  - %s (%s)", ident, providers[ident])
    if qwen is not None and not qwen_ids:
        logger.warning(
            "no Qwen models registered: cloud discovery or OCR_QWEN_MODELS produced an empty list; "
            "only %s will be served",
            ", ".join(known_ids) or "none",
        )
    return registry