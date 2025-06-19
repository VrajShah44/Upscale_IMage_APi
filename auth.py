from fastapi import Request, HTTPException
import os

# Load from .env or os environment (comma-separated if multiple)
raw_keys = os.getenv("API_KEYS", "my-secret-key")
API_KEYS = set(k.strip() for k in raw_keys.split(","))

async def verify_api_key(request: Request, call_next):
    token = request.headers.get("Authorization")
    if not token or token.replace("Bearer", "").strip() not in API_KEYS:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return await call_next(request)
