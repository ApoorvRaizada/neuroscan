import secrets

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

from app.config import Settings, get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Depends(api_key_header),
                    settings: Settings = Depends(get_settings)) -> None:
    ok = key is not None and secrets.compare_digest(key.encode(), settings.api_key.encode())
    if not ok:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")