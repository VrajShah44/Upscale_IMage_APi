
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Configure Limiter with default limits and IP-based key function
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["5/minute"]
)

# Exception handler for rate limit errors
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded"}
    )

# Example FastAPI app with endpoint-specific rate limits
app = FastAPI()
app.state.limiter = limiter
app.exception_handler(RateLimitExceeded)(rate_limit_handler)

@app.get("/general")
@limiter.limit("10/minute")
async def general_endpoint(request: Request):
    return {"message": "This endpoint has a custom rate limit of 10 requests per minute."}

@app.get("/restricted")
@limiter.limit("2/minute")
async def restricted_endpoint(request: Request):
    return {"message": "This endpoint has a stricter rate limit of 2 requests per minute."}