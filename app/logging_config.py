import logging
import os
import sys
from contextvars import ContextVar

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


class RequestContextFilter(logging.Filter):
    """Injects the current request id (or '-') into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "-"
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        import json
        import time

        payload: dict[str, object] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    """Configure root logging once, honoring LOG_LEVEL and LOG_FORMAT env vars."""
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = logging.getLevelName(level_name)
    if not isinstance(level, int):
        level = logging.INFO

    root = logging.getLogger()
    if any(getattr(handler, "_my_ocr_app", False) for handler in root.handlers):
        return

    handler = logging.StreamHandler(sys.stdout)
    handler._my_ocr_app = True  # type: ignore[attr-defined]
    handler.addFilter(RequestContextFilter())
    if os.environ.get("LOG_FORMAT", "text").lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(funcName)s: %(message)s"
            )
        )
    root.handlers = [handler]
    root.setLevel(level)


def redact_secret(value: str | None, *, visible: int = 6) -> str:
    """Return a safe representation of a secret for logging."""
    if not value:
        return "<unset>"
    if len(value) <= visible:
        return "<redacted>"
    return f"{value[:visible]}***({len(value)} chars)"
