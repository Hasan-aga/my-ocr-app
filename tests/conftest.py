from collections.abc import Iterator
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.config import Settings
from app.engines.base import EngineResult, ParsedLine
from app.engines.registry import EngineRegistry
from app.main import create_app
from app.schemas import ModelInfo


class FakeEngine:
    provider = "fake"
    id = "fake"
    label: str | None = "Fake engine"
    is_local = True

    def __init__(
        self,
        lines: list[ParsedLine] | None = None,
        *,
        ident: str | None = None,
        provider: str | None = None,
        label: str | None = None,
    ) -> None:
        self.lines = lines or [
            ParsedLine(text="First line", confidence=0.99, box=[10, 10, 200, 50]),
            ParsedLine(text="Second line", confidence=0.40, box=[10, 60, 220, 100]),
        ]
        if ident is not None:
            self.id = ident
        if provider is not None:
            self.provider = provider
        if label is not None:
            self.label = label

    def model_info(self) -> ModelInfo:
        return ModelInfo(provider=self.provider, id=self.id, label=self.label)

    def recognize(self, image_path) -> EngineResult:
        return EngineResult(lines=self.lines)

    def warmup(self) -> None:
        return None


@pytest.fixture
def settings() -> Settings:
    return Settings(api_key="test-key-1234567890", warmup=False)


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def fake_registry(fake_engine: FakeEngine) -> EngineRegistry:
    return EngineRegistry({"fake": fake_engine}, default_id="fake")


@pytest.fixture
def client(fake_registry: EngineRegistry) -> Iterator[TestClient]:
    settings = Settings(api_key="test-key-1234567890", warmup=False)
    app = create_app(settings=settings, registry=fake_registry)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def image_bytes() -> bytes:
    image = Image.new("RGB", (800, 600), "white")
    ImageDraw.Draw(image).text((40, 80), "Test note", fill="black")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()