from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.engines.registry import EngineRegistry
from app.main import create_app
from tests.conftest import FakeEngine

AUTH = {"Authorization": "Bearer test-key-1234567890"}


def test_root(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is True
    assert payload["model"]["id"] == "fake"


def test_models_requires_token(client: TestClient) -> None:
    response = client.get("/models")
    assert response.status_code == 401


def test_models_lists_registry(client: TestClient) -> None:
    response = client.get("/models", headers=AUTH)
    assert response.status_code == 200
    payload = response.json()
    assert [entry["id"] for entry in payload["models"]] == ["fake"]
    assert payload["models"][0]["provider"] == "fake"


def test_models_lists_local_and_cloud_engines() -> None:
    local = FakeEngine(ident="paddle", provider="paddle", label="PaddleOCR (local)")
    cloud = FakeEngine(ident="qwen3.5-ocr", provider="qwen", label="Qwen3.5 OCR")
    registry = EngineRegistry({"paddle": local, "qwen3.5-ocr": cloud}, default_id="paddle")
    settings = Settings(api_key="test-key-1234567890", warmup=False)
    app = create_app(settings=settings, registry=registry)
    with TestClient(app) as client:
        response = client.get("/models", headers=AUTH)
    assert response.status_code == 200
    payload = response.json()
    assert [entry["id"] for entry in payload["models"]] == ["paddle", "qwen3.5-ocr"]
    qwen_entry = payload["models"][1]
    assert qwen_entry["provider"] == "qwen"
    assert qwen_entry["label"] == "Qwen3.5 OCR"
    assert qwen_entry["pricing"]["input_usd_per_1m"] is None


def test_ocr_requires_token(client: TestClient, image_bytes: bytes) -> None:
    response = client.post("/ocr", files={"file": ("note.png", image_bytes, "image/png")})
    assert response.status_code == 401


def test_ocr_rejects_wrong_token(client: TestClient, image_bytes: bytes) -> None:
    response = client.post(
        "/ocr",
        files={"file": ("note.png", image_bytes, "image/png")},
        headers={"Authorization": "Bearer wrong-token-12345"},
    )
    assert response.status_code == 401


def test_ocr_rejects_unsupported_content_type(client: TestClient) -> None:
    response = client.post(
        "/ocr",
        files={"file": ("note.txt", b"hello", "text/plain")},
        headers=AUTH,
    )
    assert response.status_code == 415


def test_ocr_rejects_invalid_image(client: TestClient) -> None:
    response = client.post(
        "/ocr",
        files={"file": ("note.png", b"not an image", "image/png")},
        headers=AUTH,
    )
    assert response.status_code == 400


def test_ocr_success(client: TestClient, image_bytes: bytes) -> None:
    response = client.post(
        "/ocr",
        files={"file": ("note.png", image_bytes, "image/png")},
        headers=AUTH,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "First line\nSecond line"
    assert payload["line_count"] == 2
    assert payload["image"]["format"] == "PNG"
    assert payload["model"]["id"] == "fake"
    assert payload["usage"] is None


def test_ocr_selects_model(client: TestClient, image_bytes: bytes) -> None:
    response = client.post(
        "/ocr?model=fake",
        files={"file": ("note.png", image_bytes, "image/png")},
        headers=AUTH,
    )
    assert response.status_code == 200
    assert response.json()["model"]["id"] == "fake"


def test_ocr_rejects_unknown_model(client: TestClient, image_bytes: bytes) -> None:
    response = client.post(
        "/ocr?model=nope",
        files={"file": ("note.png", image_bytes, "image/png")},
        headers=AUTH,
    )
    assert response.status_code == 400


def test_ocr_filters_confidence(client: TestClient, image_bytes: bytes) -> None:
    response = client.post(
        "/ocr?min_confidence=0.9",
        files={"file": ("note.png", image_bytes, "image/png")},
        headers=AUTH,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "First line"
    assert payload["line_count"] == 1


def test_ocr_rejects_oversized_upload(fake_registry, image_bytes: bytes) -> None:
    settings = Settings(api_key="test-key-1234567890", warmup=False, max_upload_mb=1)
    app = create_app(settings=settings, registry=fake_registry)
    with TestClient(app) as client:
        response = client.post(
            "/ocr",
            files={"file": ("note.png", b"x" * (1024 * 1024 + 1), "image/png")},
            headers=AUTH,
        )
    assert response.status_code == 413


def test_image_within_resize_limit(client: TestClient) -> None:
    image = Image.new("RGB", (5000, 100), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    response = client.post(
        "/ocr",
        files={"file": ("note.png", buffer.getvalue(), "image/png")},
        headers=AUTH,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["image"]["width"] == 2000
    assert payload["image"]["height"] == 40