from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

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
def settings(tmp_path) -> Settings:
    return Settings(
        api_key="test-key-1234567890",
        warmup=False,
        data_dir=str(tmp_path / "data"),
        admin_username=None,
        admin_password=None,
    )


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def fake_registry(fake_engine: FakeEngine) -> EngineRegistry:
    return EngineRegistry({"fake": fake_engine}, default_id="fake")


@pytest.fixture
def client(fake_registry: EngineRegistry, tmp_path) -> Iterator[TestClient]:
    settings = Settings(
        api_key="test-key-1234567890",
        warmup=False,
        data_dir=str(tmp_path / "data"),
        admin_username=None,
        admin_password=None,
    )
    app = create_app(settings=settings, registry=fake_registry)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def built_static(tmp_path, monkeypatch) -> Path:
    """Simulate a built SPA bundle where app/static would live in the image."""
    static_dir = tmp_path / "built-static"
    (static_dir / "assets").mkdir(parents=True)
    (static_dir / "index.html").write_text(
        '<!doctype html><html><head><title>My OCR App</title></head>'
        '<body><div id="root"></div>'
        '<script type="module" src="/assets/index-abc123.js"></script></body></html>',
        encoding="utf-8",
    )
    (static_dir / "assets" / "index-abc123.js").write_text("console.log('app');", encoding="utf-8")
    monkeypatch.setattr("app.main.STATIC_DIR", static_dir)
    return static_dir


@pytest.fixture
def image_bytes() -> bytes:
    image = Image.new("RGB", (800, 600), "white")
    ImageDraw.Draw(image).text((40, 80), "Test note", fill="black")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()