import logging
import secrets

from fastapi import HTTPException, Request

logger = logging.getLogger("my_ocr_app")


def verify_bearer(request: Request, api_key: str) -> None:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token or not secrets.compare_digest(token, api_key):
        logger.warning(
            "authentication failed path=%s header_present=%s scheme=%s token_present=%s",
            request.url.path,
            bool(header),
            scheme or "<none>",
            bool(token),
        )
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
