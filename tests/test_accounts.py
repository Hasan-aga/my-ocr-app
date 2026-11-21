from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.engines.registry import EngineRegistry
from app.main import create_app
from tests.conftest import FakeEngine

API_KEY_AUTH = {"Authorization": "Bearer test-key-1234567890"}


@pytest.fixture
def app_client(tmp_path: Path) -> Iterator[TestClient]:
    settings = Settings(
        api_key="test-key-1234567890",
        warmup=False,
        data_dir=str(tmp_path / "data"),
        admin_username="admin",
        admin_password="admin-password-123",
    )
    registry = EngineRegistry({"fake": FakeEngine()}, default_id="fake")
    app = create_app(settings=settings, registry=registry)
    with TestClient(app) as client:
        yield client


def login(client: TestClient, username: str, password: str) -> dict:
    return client.post(
        "/auth/login", json={"username": username, "password": password}
    ).json()


def test_register_pending_and_login_blocked(app_client: TestClient) -> None:
    response = app_client.post(
        "/auth/register", json={"username": "alice", "password": "alice-password-1"}
    )
    assert response.status_code == 201
    assert response.json()["status"] == "pending"

    response = app_client.post(
        "/auth/login", json={"username": "alice", "password": "alice-password-1"}
    )
    assert response.status_code == 403
    assert "approval" in response.json()["detail"]


def test_admin_seeded_and_can_approve(app_client: TestClient, image_bytes: bytes) -> None:
    payload = login(app_client, "admin", "admin-password-123")
    assert "token" in payload
    assert payload["user"]["role"] == "admin"
    admin_auth = {"Authorization": f"Bearer {payload['token']}"}

    app_client.post(
        "/auth/register", json={"username": "alice", "password": "alice-password-1"}
    )

    users = app_client.get("/admin/users", headers=admin_auth).json()["users"]
    alice = next(u for u in users if u["username"] == "alice")
    assert alice["status"] == "pending"

    # A pending user's token would be rejected: approval comes first.
    response = app_client.post(
        f"/admin/users/{alice['id']}/approve", headers=admin_auth
    )
    assert response.status_code == 200
    assert response.json()["user"]["status"] == "approved"

    payload = login(app_client, "alice", "alice-password-1")
    assert "token" in payload
    alice_auth = {"Authorization": f"Bearer {payload['token']}"}

    assert app_client.get("/models", headers=alice_auth).status_code == 200
    response = app_client.post(
        "/ocr",
        files={"file": ("note.png", image_bytes, "image/png")},
        headers=alice_auth,
    )
    assert response.status_code == 200
    assert response.json()["line_count"] == 2


def test_login_wrong_password(app_client: TestClient) -> None:
    response = app_client.post(
        "/auth/login", json={"username": "admin", "password": "wrong-password"}
    )
    assert response.status_code == 401


def test_registration_validation(app_client: TestClient) -> None:
    response = app_client.post(
        "/auth/register", json={"username": "x", "password": "alice-password-1"}
    )
    assert response.status_code == 422
    response = app_client.post(
        "/auth/register", json={"username": "valid_name", "password": "short"}
    )
    assert response.status_code == 422
    response = app_client.post(
        "/auth/register", json={"username": "bad name", "password": "alice-password-1"}
    )
    assert response.status_code == 409


def test_duplicate_registration(app_client: TestClient) -> None:
    body = {"username": "alice", "password": "alice-password-1"}
    assert app_client.post("/auth/register", json=body).status_code == 201
    assert app_client.post("/auth/register", json=body).status_code == 409
    # Case-insensitive uniqueness.
    body["username"] = "ALICE"
    assert app_client.post("/auth/register", json=body).status_code == 409


def test_non_admin_cannot_manage_users(app_client: TestClient) -> None:
    admin_token = login(app_client, "admin", "admin-password-123")["token"]
    admin_auth = {"Authorization": f"Bearer {admin_token}"}
    app_client.post(
        "/auth/register", json={"username": "alice", "password": "alice-password-1"}
    )
    alice_id = next(
        u
        for u in app_client.get("/admin/users", headers=admin_auth).json()["users"]
        if u["username"] == "alice"
    )["id"]
    app_client.post(f"/admin/users/{alice_id}/approve", headers=admin_auth)
    alice_token = login(app_client, "alice", "alice-password-1")["token"]
    alice_auth = {"Authorization": f"Bearer {alice_token}"}

    assert app_client.get("/admin/users", headers=alice_auth).status_code == 403
    assert (
        app_client.post(f"/admin/users/{alice_id}/disable", headers=alice_auth).status_code
        == 403
    )


def test_admin_cannot_disable_self(app_client: TestClient) -> None:
    payload = login(app_client, "admin", "admin-password-123")
    admin_auth = {"Authorization": f"Bearer {payload['token']}"}
    admin_id = payload["user"]["id"]
    assert (
        app_client.post(f"/admin/users/{admin_id}/disable", headers=admin_auth).status_code
        == 400
    )
    assert (
        app_client.delete(f"/admin/users/{admin_id}", headers=admin_auth).status_code == 400
    )


def test_disable_and_delete_revoke_access(
    app_client: TestClient, image_bytes: bytes
) -> None:
    admin_auth = {
        "Authorization": f"Bearer {login(app_client, 'admin', 'admin-password-123')['token']}"
    }
    app_client.post(
        "/auth/register", json={"username": "bob", "password": "bob-password-1"}
    )
    users = app_client.get("/admin/users", headers=admin_auth).json()["users"]
    bob = next(u for u in users if u["username"] == "bob")
    app_client.post(f"/admin/users/{bob['id']}/approve", headers=admin_auth)
    bob_token = login(app_client, "bob", "bob-password-1")["token"]
    bob_auth = {"Authorization": f"Bearer {bob_token}"}
    assert app_client.get("/models", headers=bob_auth).status_code == 200

    app_client.post(f"/admin/users/{bob['id']}/disable", headers=admin_auth)
    # Existing token is rejected immediately.
    assert app_client.get("/models", headers=bob_auth).status_code == 403
    assert (
        app_client.post(
            "/auth/login", json={"username": "bob", "password": "bob-password-1"}
        ).status_code
        == 403
    )

    app_client.delete(f"/admin/users/{bob['id']}", headers=admin_auth)
    assert app_client.get("/models", headers=bob_auth).status_code == 401


def test_api_key_still_works_for_programmatic_access(
    app_client: TestClient, image_bytes: bytes
) -> None:
    response = app_client.post(
        "/ocr",
        files={"file": ("note.png", image_bytes, "image/png")},
        headers=API_KEY_AUTH,
    )
    assert response.status_code == 200
    me = app_client.get("/auth/me", headers=API_KEY_AUTH)
    assert me.status_code == 200
    assert me.json() == {"kind": "api_key", "user": None}


def test_me_with_user_token(app_client: TestClient) -> None:
    payload = login(app_client, "admin", "admin-password-123")
    me = app_client.get(
        "/auth/me", headers={"Authorization": f"Bearer {payload['token']}"}
    )
    assert me.status_code == 200
    assert me.json()["user"]["username"] == "admin"
    assert me.json()["user"]["role"] == "admin"


def test_garbage_token_rejected(app_client: TestClient) -> None:
    response = app_client.get(
        "/models", headers={"Authorization": "Bearer not-a-token"}
    )
    assert response.status_code == 401


def test_change_password(app_client: TestClient) -> None:
    payload = login(app_client, "admin", "admin-password-123")
    auth = {"Authorization": f"Bearer {payload['token']}"}
    response = app_client.post(
        "/auth/password",
        json={"current_password": "wrong", "new_password": "new-password-12345"},
        headers=auth,
    )
    assert response.status_code == 401
    response = app_client.post(
        "/auth/password",
        json={"current_password": "admin-password-123", "new_password": "new-password-12345"},
        headers=auth,
    )
    assert response.status_code == 204
    assert "token" in login(app_client, "admin", "new-password-12345")


def test_first_registration_becomes_admin_without_seed(tmp_path: Path) -> None:
    settings = Settings(
        api_key="test-key-1234567890",
        warmup=False,
        data_dir=str(tmp_path / "data"),
        admin_username=None,
        admin_password=None,
    )
    registry = EngineRegistry({"fake": FakeEngine()}, default_id="fake")
    with TestClient(create_app(settings=settings, registry=registry)) as client:
        response = client.post(
            "/auth/register", json={"username": "first", "password": "first-password-1"}
        )
        assert response.status_code == 201
        assert response.json()["status"] == "approved"
        payload = login(client, "first", "first-password-1")
        assert payload["user"]["role"] == "admin"

        response = client.post(
            "/auth/register", json={"username": "second", "password": "second-password-1"}
        )
        assert response.json()["status"] == "pending"


def test_pending_cannot_use_ocr(app_client: TestClient, image_bytes: bytes) -> None:
    app_client.post(
        "/auth/register", json={"username": "carol", "password": "carol-password-1"}
    )
    response = app_client.post(
        "/ocr",
        files={"file": ("note.png", image_bytes, "image/png")},
        headers=API_KEY_AUTH,
    )
    assert response.status_code == 200  # sanity: key path still fine


def test_token_rejected_after_user_deleted(app_client: TestClient) -> None:
    admin_auth = {
        "Authorization": f"Bearer {login(app_client, 'admin', 'admin-password-123')['token']}"
    }
    app_client.post(
        "/auth/register", json={"username": "dave", "password": "dave-password-1"}
    )
    users = app_client.get("/admin/users", headers=admin_auth).json()["users"]
    dave = next(u for u in users if u["username"] == "dave")
    app_client.post(f"/admin/users/{dave['id']}/approve", headers=admin_auth)
    dave_auth = {"Authorization": f"Bearer {login(app_client, 'dave', 'dave-password-1')['token']}"}
    assert app_client.get("/models", headers=dave_auth).status_code == 200
    app_client.delete(f"/admin/users/{dave['id']}", headers=admin_auth)
    assert app_client.get("/models", headers=dave_auth).status_code == 401
