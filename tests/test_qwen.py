import contextlib
import json
import tempfile
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.engines.base import EngineError
from app.engines.paddle import parse_result
from app.engines.qwen import QwenEngine, estimate_cost_usd, parse_qwen_content
from app.engines.qwen_catalog import QwenCatalogEntry
from app.engines.registry import create_registry


def _make_engine(handler, *, model_id: str = "qwen3.5-ocr") -> QwenEngine:
    transport = httpx.MockTransport(handler)
    return QwenEngine(
        model_id,
        api_key="test-key-1234567890",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        transport=transport,
    )


@contextlib.contextmanager
def _temp_image() -> Iterator[Path]:
    handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    path = Path(handle.name)
    handle.write(b"fake-png-bytes")
    handle.close()
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def test_engine_parses_json_lines_and_usage() -> None:
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "```json\n"
                            '[{"text": "Hello there", "box": [1, 2, 3, 4]},'
                            ' {"text": "Line two"}]'
                            "\n```"
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1500, "completion_tokens": 30, "total_tokens": 1530},
            },
        )

    engine = _make_engine(handler)
    try:
        with _temp_image() as image_path:
            result = engine.recognize(image_path)
    finally:
        engine.close()

    assert [line.text for line in result.lines] == ["Hello there", "Line two"]
    assert result.lines[0].box == [1.0, 2.0, 3.0, 4.0]
    assert result.lines[1].box is None
    assert result.usage is not None
    assert result.usage.input_tokens == 1500
    assert result.usage.output_tokens == 30
    assert result.usage.estimated_cost_usd is None

    body = calls[0]
    assert body["model"] == "qwen3.5-ocr"
    content = body["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_engine_raises_on_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "invalid api key"}},
            request=request,
        )

    engine = _make_engine(handler)
    try:
        with _temp_image() as image_path:
            with pytest.raises(EngineError):
                engine.recognize(image_path)
    finally:
        engine.close()


def test_engine_retries_on_429_then_succeeds() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(429, json={"error": "rate limit"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '[{"text": "ok"}]'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
        )

    engine = _make_engine(handler)
    try:
        with _temp_image() as image_path:
            result = engine.recognize(image_path)
    finally:
        engine.close()

    assert [line.text for line in result.lines] == ["ok"]
    assert attempts["count"] == 2


def test_parse_json_array_plain() -> None:
    lines = parse_qwen_content('[{"text": "alpha"}, {"text": "beta", "box": [0, 0, 10, 10]}]')
    assert [line.text for line in lines] == ["alpha", "beta"]
    assert lines[1].box == [0.0, 0.0, 10.0, 10.0]


def test_parse_empty_array_returns_no_lines() -> None:
    assert parse_qwen_content("[]") == []


def test_parse_falls_back_to_plain_text() -> None:
    lines = parse_qwen_content("This is the entire note text.")
    assert len(lines) == 1
    assert lines[0].text == "This is the entire note text."
    assert lines[0].box is None


def test_parse_content_parts_list() -> None:
    content = [{"type": "text", "text": '[{"text": "from parts"}]'}]
    lines = parse_qwen_content(content)
    assert [line.text for line in lines] == ["from parts"]


def test_parses_empty_content() -> None:
    assert parse_qwen_content("") == []
    assert parse_qwen_content(None) == []


def test_estimate_cost_usd() -> None:
    entry = QwenCatalogEntry(label="test", input_usd_per_1m=0.1, output_usd_per_1m=0.5)
    assert estimate_cost_usd(entry, 1_000_000, 2_000_000) == pytest.approx(1.1)
    unknown = QwenCatalogEntry(label="test", input_usd_per_1m=None, output_usd_per_1m=None)
    assert estimate_cost_usd(unknown, 1, 1) is None


def test_paddle_parse_result_smoke() -> None:
    lines = parse_result({"rec_texts": ["x"], "rec_scores": [0.9], "rec_boxes": [[0, 0, 1, 1]]})
    assert lines[0].text == "x"


def test_create_registry_with_qwen_models() -> None:
    settings = Settings(
        api_key="test-key-1234567890",
        warmup=False,
        enable_paddle=False,
        qwen_api_key="sk-1234567890abcdef",
        qwen_models="qwen3.5-ocr,qwen-custom",
        default_model="qwen3.5-ocr",
    )
    registry = create_registry(settings)
    try:
        assert registry.ids() == ["qwen3.5-ocr", "qwen-custom"]
        assert registry.default_id == "qwen3.5-ocr"
        catalog = {entry.id: entry for entry in registry.catalog()}
        assert catalog["qwen3.5-ocr"].provider == "qwen"
        assert catalog["qwen-custom"].label is None
        assert catalog["qwen-custom"].pricing is not None
        assert catalog["qwen-custom"].pricing.input_usd_per_1m is None
    finally:
        registry.close()


def test_create_registry_fails_without_engines() -> None:
    settings = Settings(
        api_key="test-key-1234567890",
        warmup=False,
        enable_paddle=False,
        qwen_api_key=None,
    )
    with pytest.raises(EngineError):
        create_registry(settings)