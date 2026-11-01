from pydantic import BaseModel, Field


class OcrLine(BaseModel):
    text: str
    confidence: float = Field(ge=0, le=1)
    box: list[float] | None = None


class ImageInfo(BaseModel):
    width: int
    height: int
    original_width: int
    original_height: int
    format: str | None = None
    content_type: str | None = None


class ModelInfo(BaseModel):
    provider: str
    id: str
    label: str | None = None
    tier: str | None = None
    detection: str | None = None
    recognition: str | None = None


class Pricing(BaseModel):
    input_usd_per_1m: float | None = Field(default=None, ge=0)
    output_usd_per_1m: float | None = Field(default=None, ge=0)


class CatalogModel(BaseModel):
    id: str
    provider: str
    label: str | None = None
    note: str | None = None
    pricing: Pricing | None = None


class ModelsResponse(BaseModel):
    models: list[CatalogModel]


class UsageInfo(BaseModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)


class OcrResponse(BaseModel):
    text: str
    lines: list[OcrLine]
    line_count: int
    image: ImageInfo
    model: ModelInfo
    usage: UsageInfo | None = None
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    ready: bool
    model: ModelInfo