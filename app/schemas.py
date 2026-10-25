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
    detection: str
    recognition: str
    tier: str


class OcrResponse(BaseModel):
    text: str
    lines: list[OcrLine]
    line_count: int
    image: ImageInfo
    model: ModelInfo
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    ready: bool
    model: ModelInfo
