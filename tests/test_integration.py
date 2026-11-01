import os
from pathlib import Path

import pytest

from app.config import Settings
from app.engines.paddle import PaddleEngine
from app.images import prepare_image


@pytest.mark.integration
def test_real_engine_on_sample() -> None:
    if os.environ.get("RUN_OCR_INTEGRATION") != "1":
        pytest.skip("Set RUN_OCR_INTEGRATION=1 to run")
    sample = Path(__file__).resolve().parents[1] / "example-note-sm.jpg"
    if not sample.exists():
        pytest.skip("Sample image is not available")
    settings = Settings(api_key="integration-key-123456", warmup=False)
    engine = PaddleEngine(settings)
    prepared = prepare_image(sample.read_bytes(), settings)
    try:
        result = engine.recognize(prepared.path)
    finally:
        prepared.path.unlink(missing_ok=True)
    lines = result.lines
    assert lines
    assert any(line.text.strip() for line in lines)
