from typing import Optional

# Try to import security packages, but don't fail if they're not available
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from fastapi_csrf_protect import CsrfProtect
    from pydantic import BaseModel
    
    class CsrfSettings(BaseModel):
        secret_key: str = "my-secret-key-change-in-production"
    
    # Initialize rate limiter
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=["100/day;50/hour;10/minute"],
    )
    
    # Initialize CSRF protection
    @CsrfProtect.load_config
    def get_csrf_config():
        return CsrfSettings()
    
except ImportError:
    # Security packages not available
    limiter = None
    CsrfProtect = None