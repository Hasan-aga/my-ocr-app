import base64
import json
import logging
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from app.engines.base import EngineError, EngineResult, EngineUsage, ParsedLine
from app.engines.qwen_catalog import QwenCatalogEntry, catalog_entry
from app.schemas import ModelInfo

logger = logging.getLogger("my_ocr_app.qwen")

PROMPT = (
    "You are an OCR engine for images of handwritten notes. "
    "Extract every line of written text exactly as it appears, in reading order "
    "(top to bottom, then left to right). Preserve spelling, punctuation, and line breaks. "
    "Do not summarize, translate, or add commentary. "
    'Return ONLY a JSON array of objects, one per text line, with shape '
    '[{"text": "<the line text>", "box": [x1, y1, x2, y2]}], where box is the '
    "axis-aligned bounding box of the line in pixel coordinates of the provided image. "
    "If a line was not recognized, omit it. If nothing is recognized, return []."
)

RETRY_STATUSES = {408, 429, 500, 502, 503, 504}


def _endpoint(base_url: str, path: str = "") -> str:
    """Host/path for logging; never includes credentials or query strings."""
    split = urlsplit(base_url)
    return f"{split.netloc}{path}"


class QwenEngine:
    provider = "qwen"
    is_local = False

    def __init__(
        self,
        model_id: str,
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.id = model_id
        self.label: str | None = None
        self._entry: QwenCatalogEntry | None = None
        entry = catalog_entry(model_id)
        if entry is not None:
            self.label = entry.label
            self._entry = entry
        self._max_retries = max_retries
        self._client = httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def model_info(self) -> ModelInfo:
        return ModelInfo(provider=self.provider, id=self.id, label=self.label)

    def recognize(self, image_path: Path) -> EngineResult:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "model": self.id,
            "temperature": 0,
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded}"},
                        },
                    ],
                }
            ],
        }
        data = self._request(payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EngineError("Invalid response from Qwen: no message content") from exc

        lines = parse_qwen_content(content)
        usage = self._usage(data)
        return EngineResult(lines=lines, usage=usage)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        attempt = 0
        model = payload.get("model", self.id)
        while True:
            try:
                response = self._client.post("/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                if attempt >= self._max_retries:
                    logger.error(
                        "qwen chat request network failure endpoint=%s model=%s "
                        "attempts=%d error=%s",
                        _endpoint(str(self._client.base_url), "/chat/completions"),
                        model,
                        attempt + 1,
                        exc,
                    )
                    raise EngineError(f"Qwen request failed: {exc}") from exc
                logger.warning(
                    "qwen chat request retrying endpoint=%s model=%s attempt=%d error=%s",
                    _endpoint(str(self._client.base_url), "/chat/completions"),
                    model,
                    attempt + 1,
                    exc,
                )
                time.sleep(self._backoff(attempt))
                attempt += 1
                continue

            if response.status_code in RETRY_STATUSES:
                if attempt >= self._max_retries:
                    logger.error(
                        "qwen chat request exhausted retries endpoint=%s model=%s "
                        "status=%d body=%s",
                        _endpoint(str(self._client.base_url), "/chat/completions"),
                        model,
                        response.status_code,
                        response.text[:300],
                    )
                    raise EngineError(
                        f"Qwen returned HTTP {response.status_code} after {attempt} retries"
                    )
                retry_after = _parse_retry_after(response.headers.get("retry-after"))
                logger.warning(
                    "qwen chat request retrying endpoint=%s model=%s status=%d "
                    "attempt=%d retry_after=%s",
                    _endpoint(str(self._client.base_url), "/chat/completions"),
                    model,
                    response.status_code,
                    attempt + 1,
                    retry_after,
                )
                time.sleep(retry_after or self._backoff(attempt))
                attempt += 1
                continue

            if response.status_code >= 400:
                logger.error(
                    "qwen chat request failed endpoint=%s model=%s status=%d body=%s",
                    _endpoint(str(self._client.base_url), "/chat/completions"),
                    model,
                    response.status_code,
                    response.text[:500],
                )
                raise EngineError(
                    f"Qwen returned HTTP {response.status_code}: {response.text[:500]}"
                )
            logger.info(
                "qwen chat request ok endpoint=%s model=%s attempts=%d",
                _endpoint(str(self._client.base_url), "/chat/completions"),
                model,
                attempt + 1,
            )
            return response.json()

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(0.5 * (2**attempt), 10.0)

    def _usage(self, data: dict[str, Any]) -> EngineUsage | None:
        usage = data.get("usage")
        if not isinstance(usage, dict):
            return None
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
            return None
        cost = None
        if self._entry is not None:
            cost = estimate_cost_usd(self._entry, input_tokens, output_tokens)
        return EngineUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
        )

    def warmup(self) -> None:
        return None

    def close(self) -> None:
        self._client.close()


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _model_sort_key(model_id: str) -> tuple[int, str]:
    lowered = model_id.lower()
    if lowered.startswith("qwen-vl-ocr"):
        return (0, lowered)
    if "vl" in lowered or "omni" in lowered:
        return (1, lowered)
    if lowered.startswith("qwen"):
        return (2, lowered)
    return (3, lowered)


def list_provider_models(
    *,
    api_key: str,
    base_url: str,
    timeout_seconds: float,
    transport: httpx.BaseTransport | None = None,
) -> list[str]:
    """List every model the API key can access, using the provider's /models endpoint."""
    endpoint = _endpoint(base_url, "/models")
    logger.info("qwen model discovery starting endpoint=%s", endpoint)
    with httpx.Client(
        base_url=base_url,
        timeout=httpx.Timeout(timeout_seconds),
        transport=transport,
        headers={"Authorization": f"Bearer {api_key}"},
    ) as client:
        try:
            response = client.get("/models")
        except httpx.HTTPError as exc:
            logger.error("qwen model discovery network failure endpoint=%s error=%s", endpoint, exc)
            raise EngineError(f"Qwen model discovery failed: {exc}") from exc
        if response.status_code != 200:
            hint = ""
            if response.status_code == 401:
                hint = (
                    " (api key rejected: check that OCR_QWEN_API_KEY matches the region of "
                    "OCR_QWEN_BASE_URL, e.g. keys from Alibaba Cloud International must use "
                    "dashscope-intl.aliyuncs.com)"
                )
            logger.error(
                "qwen model discovery failed endpoint=%s status=%d body=%s%s",
                endpoint,
                response.status_code,
                response.text[:300],
                hint,
            )
            raise EngineError(
                f"Qwen model discovery returned HTTP {response.status_code}: "
                f"{response.text[:300]}{hint}"
            )
        data = response.json()
    ids: list[str] = [
        str(item.get("id"))
        for item in data.get("data", [])
        if isinstance(item, dict) and item.get("id")
    ]
    ids = sorted(set(ids), key=_model_sort_key)
    logger.info(
        "qwen model discovery succeeded endpoint=%s count=%d models=%s",
        endpoint,
        len(ids),
        ",".join(ids),
    )
    return ids


class QwenProvider:
    """Builds Qwen engines on demand and discovers the available model list."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._transport = transport

    def discover(self) -> list[str]:
        return list_provider_models(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout_seconds=self._timeout_seconds,
            transport=self._transport,
        )

    def build(self, model_id: str) -> QwenEngine:
        return QwenEngine(
            model_id,
            api_key=self._api_key,
            base_url=self._base_url,
            timeout_seconds=self._timeout_seconds,
            max_retries=self._max_retries,
            transport=self._transport,
        )


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return ""


def _extract_json_array(text: str) -> str | None:
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _box_from_item(item: dict[str, Any]) -> list[float] | None:
    box = item.get("box") or item.get("bbox") or item.get("box_xyxy")
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        values = [float(value) for value in box]
    except (TypeError, ValueError):
        return None
    return values


def parse_qwen_content(content: Any) -> list[ParsedLine]:
    text = _content_to_text(content).strip()
    if not text:
        return []

    array = _extract_json_array(text)
    if array is not None:
        try:
            data = json.loads(array)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list):
            lines = []
            for item in data:
                if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                    continue
                text = item["text"].strip()
                if not text:
                    continue
                lines.append(ParsedLine(text=text, confidence=1.0, box=_box_from_item(item)))
            if lines or not data:
                return lines

    cleaned = text.strip()
    for marker in ("```json", "```"):
        if cleaned.startswith(marker):
            cleaned = cleaned[len(marker) :].lstrip()
        if cleaned.endswith("```"):
            cleaned = cleaned[: -len("```")].rstrip()
    return [ParsedLine(text=cleaned, confidence=1.0, box=None)]


def estimate_cost_usd(
    entry: QwenCatalogEntry,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    if entry.input_usd_per_1m is None or entry.output_usd_per_1m is None:
        return None
    return (
        input_tokens / 1_000_000 * entry.input_usd_per_1m
        + output_tokens / 1_000_000 * entry.output_usd_per_1m
    )