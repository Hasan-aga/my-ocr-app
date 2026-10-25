from collections.abc import Iterator
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.config import Settings
from app.main import create_app
from app.ocr import ParsedLine
from app.schemas import ModelInfo


class FakeEngine:
    det_model = "fake-det"
    rec_model = "fake-rec"

    def __init__(self, lines: list[ParsedLine] | None = None) -> None:
        self.lines = lines or [
            ParsedLine(text="First line", confidence=0.99, box=[10, 10, 200, 50]),
            ParsedLine(text="Second line", confidence=0.40, box=[10, 60, 220, 100]),
        ]

    def model_info(self) -> ModelInfo:
        return ModelInfo(detection=self.det_model, recognition=self.rec_model, tier="medium")

    def recognize(self, image_path) -> list[ParsedLine]:
        return self.lines

    def warmup(self) -> None:
        return None


@pytest.fixture
def settings() -> Settings:
    return Settings(api_key="test-key-1234567890", warmup=False)


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def client(settings: Settings, fake_engine: FakeEngine) -> Iterator[TestClient]:
    app = create_app(settings=settings, engine=fake_engine)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def image_bytes() -> bytes:
    image = Image.new("RGB", (800, 600), "white")
    ImageDraw.Draw(image).text((40, 80), "Test note", fill="black")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
