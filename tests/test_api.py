from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.engines.registry import EngineRegistry
from app.main import create_app
from tests.conftest import FakeEngine

AUTH = {"Authorization": "Bearer test-key-1234567890"}


def test_root_serves_frontend(client: TestClient, built_static: Path) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "My OCR App" in response.text
    assert 'id="root"' in response.text


def test_root_without_build_returns_hint(client: TestClient, tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    # Point both candidate locations at a directory with no index.html.
    app = create_app(
        settings=Settings(
            api_key="test-key-1234567890",
            warmup=False,
            data_dir=str(tmp_path / "d"),
            admin_username=None,
            admin_password=None,
        ),
        registry=EngineRegistry({"fake": FakeEngine()}, default_id="fake"),
    )
    import app.main as main_module

    original = (main_module.STATIC_DIR, main_module.FRONTEND_DIST)
    main_module.STATIC_DIR, main_module.FRONTEND_DIST = missing, missing / "dist"
    try:
        with TestClient(app) as test_client:
            response = test_client.get("/")
    finally:
        main_module.STATIC_DIR, main_module.FRONTEND_DIST = original
    assert response.status_code == 503
    assert "npm run build" in response.json()["detail"]


def test_static_assets_and_spa_fallback(client: TestClient, built_static: Path) -> None:
    response = client.get("/assets/index-abc123.js")
    assert response.status_code == 200
    assert "console.log" in response.text
    # Unknown non-API path serves the app shell for client-side routing.
    response = client.get("/some/spa/route")
    assert response.status_code == 200
    assert 'id="root"' in response.text
    # Unknown API path returns 404 instead of the shell.
    assert client.get("/ocr/nonexistent").status_code == 404


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


def test_ocr_rate_limit(fake_registry, image_bytes: bytes) -> None:
    settings = Settings(
        api_key="test-key-1234567890",
        warmup=False,
        rate_limit_ocr_per_minute=1,
    )
    app = create_app(settings=settings, registry=fake_registry)
    with TestClient(app) as client:
        assert (
            client.post(
                "/ocr",
                files={"file": ("note.png", image_bytes, "image/png")},
                headers=AUTH,
            ).status_code
            == 200
        )
        limited = client.post(
            "/ocr",
            files={"file": ("note.png", image_bytes, "image/png")},
            headers=AUTH,
        )
        assert limited.status_code == 429
        assert int(limited.headers["retry-after"]) > 0
        assert client.get("/healthz").status_code == 200


def test_models_rate_limit(fake_registry) -> None:
    settings = Settings(
        api_key="test-key-1234567890",
        warmup=False,
        rate_limit_models_per_minute=1,
    )
    app = create_app(settings=settings, registry=fake_registry)
    with TestClient(app) as client:
        assert client.get("/models", headers=AUTH).status_code == 200
        assert client.get("/models", headers=AUTH).status_code == 429


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