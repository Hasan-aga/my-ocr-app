from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app

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
    assert payload["model"]["detection"] == "fake-det"


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
    assert payload["model"]["recognition"] == "fake-rec"


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


def test_ocr_rejects_oversized_upload(fake_engine, image_bytes: bytes) -> None:
    settings = Settings(api_key="test-key-1234567890", warmup=False, max_upload_mb=1)
    app = create_app(settings=settings, engine=fake_engine)
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
