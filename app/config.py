from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_QWEN_MODELS = "qwen3.5-ocr,qwen3.8-flash,qwen3.7-plus,qwen3.8-max,qwen3-vl-plus"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OCR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str = Field(min_length=16)
    default_model: str | None = None
    enable_paddle: bool = True
    model_tier: Literal["tiny", "small", "medium"] = "medium"
    det_model_name: str | None = None
    rec_model_name: str | None = None
    preprocess: Literal["none", "contrast"] = "none"
    max_upload_mb: int = Field(default=15, gt=0, le=100)
    max_image_pixels: int = Field(default=50_000_000, gt=0)
    max_side: int = Field(default=2000, gt=0)
    det_limit_side_len: int = Field(default=960, gt=0)
    det_limit_type: Literal["min", "max"] = "max"
    det_thresh: float = Field(default=0.3, ge=0, le=1)
    det_box_thresh: float = Field(default=0.6, ge=0, le=1)
    det_unclip_ratio: float = Field(default=2.0, gt=0)
    rec_score_thresh: float = Field(default=0.0, ge=0, le=1)
    cpu_threads: int = Field(default=3, gt=0, le=32)
    enable_mkldnn: bool = True
    warmup: bool = True
    use_textline_orientation: bool = False
    busy_timeout_seconds: float = Field(default=1.0, gt=0)
    docs_enabled: bool = False

    qwen_api_key: str | None = None
    qwen_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    qwen_models: str = DEFAULT_QWEN_MODELS
    qwen_timeout_seconds: float = Field(default=60.0, gt=0)
    qwen_max_retries: int = Field(default=3, ge=0, le=10)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def resolved_det_model(self) -> str:
        return self.det_model_name or f"PP-OCRv6_{self.model_tier}_det"

    @property
    def resolved_rec_model(self) -> str:
        return self.rec_model_name or f"PP-OCRv6_{self.model_tier}_rec"

    @property
    def qwen_model_ids(self) -> list[str]:
        return [model.strip() for model in self.qwen_models.split(",") if model.strip()]