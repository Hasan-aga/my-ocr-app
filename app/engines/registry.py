import logging

from app.config import Settings
from app.engines.base import EngineError, OcrEngine
from app.engines.paddle import PADDLE_ID, PaddleEngine
from app.engines.qwen import QwenEngine
from app.engines.qwen_catalog import catalog_entry
from app.schemas import CatalogModel, ModelInfo, Pricing

logger = logging.getLogger("my_ocr_app")


class EngineRegistryError(EngineError):
    """Raised when the engine registry cannot be built or resolved."""


class EngineRegistry:
    def __init__(self, engines: dict[str, OcrEngine], default_id: str) -> None:
        self._engines = engines
        self._default_id = default_id

    @property
    def default_id(self) -> str:
        return self._default_id

    def ids(self) -> list[str]:
        return list(self._engines)

    def default(self) -> OcrEngine:
        return self._engines[self._default_id]

    def get(self, model_id: str) -> OcrEngine:
        try:
            return self._engines[model_id]
        except KeyError:
            available = ", ".join(self.ids()) or "none"
            raise EngineRegistryError(
                f"Unknown model {model_id!r}; available: {available}"
            ) from None

    def default_model_info(self) -> ModelInfo:
        return self.default().model_info()

    def catalog(self) -> list[CatalogModel]:
        models: list[CatalogModel] = []
        for ident, engine in self._engines.items():
            entry = catalog_entry(ident)
            pricing = Pricing()
            if entry is not None:
                pricing = Pricing(
                    input_usd_per_1m=entry.input_usd_per_1m,
                    output_usd_per_1m=entry.output_usd_per_1m,
                )
            models.append(
                CatalogModel(
                    id=ident,
                    provider=engine.provider,
                    label=engine.label,
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


def create_registry(settings: Settings) -> EngineRegistry:
    engines: dict[str, OcrEngine] = {}

    if settings.enable_paddle:
        engines[PADDLE_ID] = PaddleEngine(settings)

    if settings.qwen_api_key:
        for model_id in settings.qwen_model_ids:
            if model_id in engines:
                logger.warning("Ignoring duplicate model id %r in OCR_QWEN_MODELS", model_id)
                continue
            engines[model_id] = QwenEngine(
                model_id,
                api_key=settings.qwen_api_key,
                base_url=settings.qwen_base_url,
                timeout_seconds=settings.qwen_timeout_seconds,
                max_retries=settings.qwen_max_retries,
            )
    elif not settings.enable_paddle and settings.qwen_models:
        logger.warning(
            "Qwen cloud models configured but OCR_QWEN_API_KEY is not set; cloud engines disabled"
        )

    if not engines:
        raise EngineRegistryError(
            "No OCR engines configured: enable PaddleOCR or set OCR_QWEN_API_KEY"
        )

    default_id = settings.default_model
    if default_id is None:
        default_id = PADDLE_ID if PADDLE_ID in engines else next(iter(engines))
    if default_id not in engines:
        raise EngineRegistryError(
            f"Default model {default_id!r} is not enabled; "
            f"available: {', '.join(engines) or 'none'}"
        )

    logger.info("OCR registry ready with %d engine(s); default=%s", len(engines), default_id)
    for ident in engines:
        logger.info("  - %s (%s)", ident, engines[ident].provider)
    return EngineRegistry(engines, default_id)