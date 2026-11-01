from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.schemas import ModelInfo, UsageInfo


class EngineError(Exception):
    """Raised when an OCR engine fails to process an image."""


@dataclass(frozen=True)
class ParsedLine:
    text: str
    confidence: float
    box: list[float] | None


@dataclass(frozen=True)
class EngineUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None

    def as_schema(self) -> UsageInfo:
        return UsageInfo(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            estimated_cost_usd=self.estimated_cost_usd,
        )


@dataclass(frozen=True)
class EngineResult:
    lines: list[ParsedLine]
    usage: EngineUsage | None = None


class OcrEngine(Protocol):
    provider: str
    id: str
    label: str | None
    is_local: bool

    def model_info(self) -> ModelInfo: ...

    def recognize(self, image_path: Path) -> EngineResult: ...

    def warmup(self) -> None: ...